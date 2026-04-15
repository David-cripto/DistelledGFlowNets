from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from ..diffusion.model import TimeEmbedding


AVAILABLE_DETAILED_BALANCE_MODELS = ("direct",)


def available_detailed_balance_models() -> tuple[str, ...]:
    return AVAILABLE_DETAILED_BALANCE_MODELS


def _group_norm_groups(num_channels: int) -> int:
    for groups in (8, 4, 2):
        if num_channels % groups == 0:
            return groups
    return 1


def _prepare_inputs(x: torch.Tensor, t: torch.Tensor, input_shape: tuple[int, int, int]) -> torch.Tensor:
    expected_shape = (x.shape[0], *input_shape)
    if x.ndim != 4 or tuple(x.shape) != expected_shape:
        raise ValueError(
            "x must have shape [batch, channels, height, width] "
            f"with image shape {input_shape}, got {tuple(x.shape)}."
        )
    if t.ndim == 0:
        t = t.expand(x.shape[0])
    if t.ndim != 1:
        t = t.reshape(-1)
    if t.shape[0] != x.shape[0]:
        raise ValueError("t must have shape [batch].")
    return t.to(device=x.device, dtype=torch.float32)


class DetailedBalanceModel(nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError


class _ResidualTimeConvBlock(nn.Module):
    def __init__(self, channels: int, time_dim: int):
        super().__init__()
        groups = _group_norm_groups(channels)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.time_projection = nn.Linear(time_dim, channels)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, time_features: torch.Tensor) -> torch.Tensor:
        residual = x
        hidden = self.conv1(F.silu(self.norm1(x)))
        hidden = hidden + self.time_projection(time_features).view(x.shape[0], -1, 1, 1)
        hidden = self.conv2(F.silu(self.norm2(hidden)))
        return residual + hidden


class DetailedBalanceMLP(DetailedBalanceModel):
    def __init__(
        self,
        input_shape: tuple[int, int, int],
        hidden_dim: int = 512,
        depth: int = 3,
        num_time_frequencies: int = 16,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be at least 1")

        self.input_shape = input_shape
        feature_channels = max(32, min(hidden_dim // 8, 128))
        self.time_embedding = TimeEmbedding(num_frequencies=num_time_frequencies)
        time_dim = feature_channels * 4

        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_embedding.output_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
        )
        self.input_projection = nn.Conv2d(input_shape[0], feature_channels, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList(
            [_ResidualTimeConvBlock(channels=feature_channels, time_dim=time_dim) for _ in range(depth)]
        )
        self.feature_projection = nn.Sequential(
            nn.GroupNorm(_group_norm_groups(feature_channels), feature_channels),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Sequential(
            nn.Linear(feature_channels + time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del timesteps
        t = _prepare_inputs(x, t, self.input_shape)
        time_features = self.time_mlp(self.time_embedding(t))
        hidden = self.input_projection(x)
        for block in self.blocks:
            hidden = block(hidden, time_features)
        pooled_features = self.feature_projection(hidden).flatten(start_dim=1)
        score_features = torch.cat([pooled_features, time_features], dim=-1)
        return self.head(score_features).squeeze(-1)


def build_detailed_balance_model(
    model_type: str,
    input_shape: tuple[int, int, int],
    hidden_dim: int,
    depth: int,
    num_time_frequencies: int,
) -> DetailedBalanceModel:
    if model_type != "direct":
        raise ValueError(
            f"Unknown detailed-balance model type '{model_type}'. "
            f"Expected one of {', '.join(AVAILABLE_DETAILED_BALANCE_MODELS)}."
        )
    return DetailedBalanceMLP(
        input_shape=input_shape,
        hidden_dim=hidden_dim,
        depth=depth,
        num_time_frequencies=num_time_frequencies,
    )
