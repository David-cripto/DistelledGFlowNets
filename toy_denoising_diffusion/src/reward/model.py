from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from ..diffusion.model import DenoiserMLP, TimeEmbedding
from ..diffusion.schedules import DDPMSchedule


AVAILABLE_DETAILED_BALANCE_MODELS = (
    "direct",
    "target_factored",
    "denoised_target_factored",
)


def available_detailed_balance_models() -> tuple[str, ...]:
    return AVAILABLE_DETAILED_BALANCE_MODELS


def _prepare_inputs(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError("x must have shape [batch, 2]")
    if t.ndim == 0:
        t = t.expand(x.shape[0])
    if t.ndim != 1:
        t = t.reshape(-1)
    if t.shape[0] != x.shape[0]:
        raise ValueError("t must have shape [batch]")
    return t.to(device=x.device, dtype=torch.float32)


def _prepare_timesteps(
    timesteps: Optional[torch.Tensor],
    t: torch.Tensor,
    num_steps: int,
) -> torch.Tensor:
    if timesteps is None:
        inferred = torch.round(t * float(num_steps) - 1.0)
        return inferred.clamp(0, num_steps - 1).to(dtype=torch.long)
    if timesteps.ndim == 0:
        timesteps = timesteps.expand(t.shape[0])
    if timesteps.ndim != 1:
        timesteps = timesteps.reshape(-1)
    if timesteps.shape[0] != t.shape[0]:
        raise ValueError("timesteps must have shape [batch]")
    return timesteps.to(device=t.device, dtype=torch.long)


class DetailedBalanceModel(nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError


class _TimeConditionedScalarMLP(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        input_dim: int = 2,
        depth: int = 3,
        num_time_frequencies: int = 8,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be at least 1")

        self.time_embedding = TimeEmbedding(num_frequencies=num_time_frequencies)
        expanded_input_dim = input_dim + self.time_embedding.output_dim

        layers = []
        width = expanded_input_dim
        for _ in range(depth):
            layers.append(nn.Linear(width, hidden_dim))
            layers.append(nn.SiLU())
            width = hidden_dim

        self.backbone = nn.Sequential(*layers)
        self.output = nn.Linear(width, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        features = torch.cat([x, self.time_embedding(t)], dim=-1)
        hidden = self.backbone(features)
        return self.output(hidden).squeeze(-1)


class _StateScalarMLP(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        input_dim: int = 2,
        depth: int = 3,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be at least 1")

        layers = []
        width = input_dim
        for _ in range(depth):
            layers.append(nn.Linear(width, hidden_dim))
            layers.append(nn.ReLU())
            width = hidden_dim

        self.backbone = nn.Sequential(*layers)
        self.output = nn.Linear(width, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(x)
        return self.output(hidden).squeeze(-1)


class DetailedBalanceMLP(DetailedBalanceModel):
    def __init__(
        self,
        hidden_dim: int = 128,
        input_dim: int = 2,
        depth: int = 3,
        num_time_frequencies: int = 8,
    ):
        super().__init__()
        self.scalar_field = _TimeConditionedScalarMLP(
            hidden_dim=hidden_dim,
            input_dim=input_dim,
            depth=depth,
            num_time_frequencies=num_time_frequencies,
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del timesteps
        t = _prepare_inputs(x, t)
        return self.scalar_field(x, t)


class TargetFactoredDetailedBalanceMLP(DetailedBalanceModel):
    def __init__(
        self,
        hidden_dim: int = 128,
        input_dim: int = 2,
        depth: int = 3,
        num_time_frequencies: int = 8,
    ):
        super().__init__()
        self.log_r = _StateScalarMLP(
            hidden_dim=hidden_dim,
            input_dim=input_dim,
            depth=depth,
        )
        self.log_tilde = _TimeConditionedScalarMLP(
            hidden_dim=hidden_dim,
            input_dim=input_dim,
            depth=depth,
            num_time_frequencies=num_time_frequencies,
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del timesteps
        t = _prepare_inputs(x, t)
        return self.log_r(x) + t * self.log_tilde(x, t)


class DenoisedTargetFactoredDetailedBalanceMLP(DetailedBalanceModel):
    def __init__(
        self,
        denoiser: DenoiserMLP,
        schedule: DDPMSchedule,
        hidden_dim: int = 128,
        input_dim: int = 2,
        depth: int = 3,
        num_time_frequencies: int = 8,
    ):
        super().__init__()
        self.denoiser = denoiser.eval()
        self.schedule = schedule
        for parameter in self.denoiser.parameters():
            parameter.requires_grad_(False)

        self.log_r = _StateScalarMLP(
            hidden_dim=hidden_dim,
            input_dim=input_dim,
            depth=depth,
        )
        self.log_tilde = _TimeConditionedScalarMLP(
            hidden_dim=hidden_dim,
            input_dim=input_dim,
            depth=depth,
            num_time_frequencies=num_time_frequencies,
        )

    @torch.no_grad()
    def _predict_x0(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        t = _prepare_inputs(x, t)
        output = x.clone()
        nonzero_mask = t > 0.0
        if not bool(nonzero_mask.any()):
            return output

        resolved_timesteps = _prepare_timesteps(
            timesteps=timesteps,
            t=t,
            num_steps=self.schedule.num_steps,
        )
        masked_x = x[nonzero_mask]
        masked_timesteps = resolved_timesteps[nonzero_mask]
        noise_prediction = self.denoiser(masked_x, self.schedule.time_values(masked_timesteps))
        output[nonzero_mask] = self.schedule.predict_x0(masked_x, masked_timesteps, noise_prediction)
        return output

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        t = _prepare_inputs(x, t)
        raw_output = self.log_tilde(x, t)
        with torch.no_grad():
            x_hat = self._predict_x0(x, t, timesteps=timesteps)
        return self.log_r(x_hat.detach()) + t * raw_output


def build_detailed_balance_model(
    model_type: str,
    hidden_dim: int,
    depth: int,
    num_time_frequencies: int,
    denoiser: DenoiserMLP,
    schedule: DDPMSchedule,
) -> DetailedBalanceModel:
    if model_type == "direct":
        return DetailedBalanceMLP(
            hidden_dim=hidden_dim,
            depth=depth,
            num_time_frequencies=num_time_frequencies,
        )
    if model_type == "target_factored":
        return TargetFactoredDetailedBalanceMLP(
            hidden_dim=hidden_dim,
            depth=depth,
            num_time_frequencies=num_time_frequencies,
        )
    if model_type == "denoised_target_factored":
        return DenoisedTargetFactoredDetailedBalanceMLP(
            denoiser=denoiser,
            schedule=schedule,
            hidden_dim=hidden_dim,
            depth=depth,
            num_time_frequencies=num_time_frequencies,
        )
    raise ValueError(
        f"Unknown detailed-balance model type '{model_type}'. "
        f"Expected one of {', '.join(AVAILABLE_DETAILED_BALANCE_MODELS)}."
    )
