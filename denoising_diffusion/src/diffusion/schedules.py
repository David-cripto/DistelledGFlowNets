from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch


AVAILABLE_BETA_SCHEDULES = ("linear", "cosine")


def available_beta_schedules() -> tuple[str, ...]:
    return AVAILABLE_BETA_SCHEDULES


@dataclass
class DDPMSchedule:
    num_steps: int = 128
    beta_schedule: str = "linear"
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    cosine_s: float = 0.008
    max_beta: float = 0.999
    epsilon: float = 1e-8
    betas: torch.Tensor = field(init=False, repr=False)
    alphas: torch.Tensor = field(init=False, repr=False)
    alpha_bars: torch.Tensor = field(init=False, repr=False)
    alpha_bars_prev: torch.Tensor = field(init=False, repr=False)
    sqrt_alpha_bars: torch.Tensor = field(init=False, repr=False)
    sqrt_one_minus_alpha_bars: torch.Tensor = field(init=False, repr=False)
    posterior_variances: torch.Tensor = field(init=False, repr=False)
    posterior_mean_coef1: torch.Tensor = field(init=False, repr=False)
    posterior_mean_coef2: torch.Tensor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.num_steps < 1:
            raise ValueError("num_steps must be at least 1")
        if self.beta_schedule not in AVAILABLE_BETA_SCHEDULES:
            raise ValueError(
                f"Unknown beta_schedule '{self.beta_schedule}'. "
                f"Expected one of {', '.join(AVAILABLE_BETA_SCHEDULES)}."
            )
        if not (0.0 < self.max_beta < 1.0):
            raise ValueError("Expected 0 < max_beta < 1.")
        if self.beta_schedule == "linear":
            if not (0.0 < self.beta_start < self.beta_end < 1.0):
                raise ValueError("Expected 0 < beta_start < beta_end < 1 for a linear schedule.")
            self.betas = torch.linspace(self.beta_start, self.beta_end, self.num_steps, dtype=torch.float32)
        else:
            if self.cosine_s < 0.0:
                raise ValueError("Expected cosine_s >= 0.")
            self.betas = self._cosine_betas()

        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.alpha_bars_prev = torch.cat([torch.ones(1, dtype=torch.float32), self.alpha_bars[:-1]], dim=0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt((1.0 - self.alpha_bars).clamp_min(self.epsilon))
        self.posterior_variances = (
            self.betas * (1.0 - self.alpha_bars_prev) / (1.0 - self.alpha_bars).clamp_min(self.epsilon)
        )
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alpha_bars_prev) / (1.0 - self.alpha_bars).clamp_min(self.epsilon)
        )
        self.posterior_mean_coef2 = (
            torch.sqrt(self.alphas) * (1.0 - self.alpha_bars_prev) / (1.0 - self.alpha_bars).clamp_min(self.epsilon)
        )

    def _cosine_betas(self) -> torch.Tensor:
        steps = torch.arange(self.num_steps + 1, dtype=torch.float32)
        normalized_time = steps / float(self.num_steps)
        angles = ((normalized_time + self.cosine_s) / (1.0 + self.cosine_s)) * (0.5 * math.pi)
        alpha_bar = torch.cos(angles).pow(2)
        alpha_bar = alpha_bar / alpha_bar[0].clamp_min(self.epsilon)
        betas = 1.0 - (alpha_bar[1:] / alpha_bar[:-1].clamp_min(self.epsilon))
        return betas.clamp(self.epsilon, self.max_beta)

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randint(0, self.num_steps, (batch_size,), device=device, dtype=torch.long)

    def time_values(self, timesteps: torch.Tensor) -> torch.Tensor:
        return (timesteps.to(dtype=torch.float32) + 1.0) / float(self.num_steps)

    def extract(self, values: torch.Tensor, timesteps: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        gathered = values.to(reference.device)[timesteps]
        return gathered.view(reference.shape[0], *([1] * (reference.ndim - 1)))

    def q_sample(self, x0: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        sqrt_alpha_bar = self.extract(self.sqrt_alpha_bars, timesteps, x0)
        sqrt_one_minus_alpha_bar = self.extract(self.sqrt_one_minus_alpha_bars, timesteps, x0)
        return sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise

    def predict_x0(self, x_t: torch.Tensor, timesteps: torch.Tensor, noise_prediction: torch.Tensor) -> torch.Tensor:
        sqrt_alpha_bar = self.extract(self.sqrt_alpha_bars, timesteps, x_t)
        sqrt_one_minus_alpha_bar = self.extract(self.sqrt_one_minus_alpha_bars, timesteps, x_t)
        return (x_t - sqrt_one_minus_alpha_bar * noise_prediction) / sqrt_alpha_bar.clamp_min(self.epsilon)

    def posterior_mean(self, x0_prediction: torch.Tensor, x_t: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        coef1 = self.extract(self.posterior_mean_coef1, timesteps, x_t)
        coef2 = self.extract(self.posterior_mean_coef2, timesteps, x_t)
        return coef1 * x0_prediction + coef2 * x_t

    def posterior_variance(self, timesteps: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        return self.extract(self.posterior_variances, timesteps, reference)
