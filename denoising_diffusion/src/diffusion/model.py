from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def _group_norm_groups(num_channels: int) -> int:
    for groups in (8, 4, 2):
        if num_channels % groups == 0:
            return groups
    return 1


class TimeEmbedding(nn.Module):
    def __init__(self, num_frequencies: int = 16):
        super().__init__()
        if num_frequencies < 1:
            raise ValueError("num_frequencies must be at least 1")

        frequencies = (2.0 ** torch.arange(num_frequencies, dtype=torch.float32)) * math.pi
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.output_dim = 1 + 2 * num_frequencies

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.view(-1, 1).to(dtype=torch.float32)
        angles = t * self.frequencies.view(1, -1)
        return torch.cat([t, torch.sin(angles), torch.cos(angles)], dim=-1)


class ResidualTimeBlock(nn.Module):
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


class DenoiserCNN(nn.Module):
    def __init__(
        self,
        image_channels: int = 1,
        hidden_channels: int = 64,
        depth: int = 4,
        num_time_frequencies: int = 16,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be at least 1")

        self.image_channels = image_channels
        self.time_embedding = TimeEmbedding(num_frequencies=num_time_frequencies)
        time_dim = hidden_channels * 4

        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_embedding.output_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
        )
        self.input_projection = nn.Conv2d(image_channels, hidden_channels, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList(
            [ResidualTimeBlock(channels=hidden_channels, time_dim=time_dim) for _ in range(depth)]
        )
        self.output_projection = nn.Sequential(
            nn.GroupNorm(_group_norm_groups(hidden_channels), hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, image_channels, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.output_projection[-1].weight)
        nn.init.zeros_(self.output_projection[-1].bias)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if x_t.ndim != 4:
            raise ValueError("x_t must have shape [batch, channels, height, width].")
        if x_t.shape[1] != self.image_channels:
            raise ValueError(f"Expected {self.image_channels} channels, got {x_t.shape[1]}.")
        if t.ndim == 0:
            t = t.expand(x_t.shape[0])
        if t.ndim != 1:
            t = t.reshape(-1)
        if t.shape[0] != x_t.shape[0]:
            raise ValueError("t must have shape [batch].")

        time_features = self.time_mlp(self.time_embedding(t))
        hidden = self.input_projection(x_t)
        for block in self.blocks:
            hidden = block(hidden, time_features)
        return self.output_projection(hidden)
