from __future__ import annotations

import math

import torch
from torch import nn


class TimeEmbedding(nn.Module):
    def __init__(self, num_frequencies: int = 8):
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


class DenoiserMLP(nn.Module):
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
        self.output = nn.Linear(width, input_dim)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if x_t.ndim != 2 or x_t.shape[1] != 2:
            raise ValueError("x_t must have shape [batch, 2]")
        if t.ndim == 0:
            t = t.expand(x_t.shape[0])
        if t.ndim != 1:
            t = t.reshape(-1)
        if t.shape[0] != x_t.shape[0]:
            raise ValueError("t must have shape [batch]")

        features = torch.cat([x_t, self.time_embedding(t)], dim=-1)
        hidden = self.backbone(features)
        return self.output(hidden)
