from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def _group_norm_groups(num_channels: int) -> int:
    for groups in (32, 16, 8, 4, 2):
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


class _TimeConditionedResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(_group_norm_groups(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_projection = nn.Linear(time_dim, out_channels)
        self.norm2 = nn.GroupNorm(_group_norm_groups(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor, time_features: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        hidden = self.conv1(F.silu(self.norm1(x)))
        hidden = hidden + self.time_projection(time_features).view(x.shape[0], -1, 1, 1)
        hidden = self.conv2(F.silu(self.norm2(hidden)))
        return residual + hidden


class _DownsampleBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        feature_channels: int,
        downsample_out_channels: int,
        time_dim: int,
    ):
        super().__init__()
        self.block1 = _TimeConditionedResBlock(in_channels, feature_channels, time_dim)
        self.block2 = _TimeConditionedResBlock(feature_channels, feature_channels, time_dim)
        self.downsample = nn.Conv2d(feature_channels, downsample_out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor, time_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.block1(x, time_features)
        x = self.block2(x, time_features)
        skip = x
        x = self.downsample(x)
        return x, skip


class _UpsampleBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        time_dim: int,
    ):
        super().__init__()
        merged_channels = in_channels + skip_channels
        self.block1 = _TimeConditionedResBlock(merged_channels, out_channels, time_dim)
        self.block2 = _TimeConditionedResBlock(out_channels, out_channels, time_dim)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, time_features: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.block1(x, time_features)
        x = self.block2(x, time_features)
        return x


class DenoiserUNet(nn.Module):
    def __init__(
        self,
        image_channels: int = 1,
        hidden_channels: int = 64,
        depth: int = 4,
        num_time_frequencies: int = 16,
    ):
        super().__init__()
        if depth < 2:
            raise ValueError("depth must be at least 2 for a UNet noise predictor.")
        if hidden_channels < 8:
            raise ValueError("hidden_channels must be at least 8.")

        self.image_channels = image_channels
        self.hidden_channels = hidden_channels
        self.depth = depth
        self.time_embedding = TimeEmbedding(num_frequencies=num_time_frequencies)
        time_dim = hidden_channels * 4
        self.channel_levels = [hidden_channels * (2**level) for level in range(depth)]

        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_embedding.output_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
        )
        self.input_projection = nn.Conv2d(image_channels, self.channel_levels[0], kernel_size=3, padding=1)

        self.encoder_blocks = nn.ModuleList(
            [
                _DownsampleBlock(
                    in_channels=self.channel_levels[level],
                    feature_channels=self.channel_levels[level],
                    downsample_out_channels=self.channel_levels[level + 1],
                    time_dim=time_dim,
                )
                for level in range(depth - 1)
            ]
        )
        bottleneck_channels = self.channel_levels[-1]
        self.middle_blocks = nn.ModuleList(
            [
                _TimeConditionedResBlock(bottleneck_channels, bottleneck_channels, time_dim),
                _TimeConditionedResBlock(bottleneck_channels, bottleneck_channels, time_dim),
            ]
        )
        self.decoder_blocks = nn.ModuleList(
            [
                _UpsampleBlock(
                    in_channels=self.channel_levels[level + 1],
                    skip_channels=self.channel_levels[level],
                    out_channels=self.channel_levels[level],
                    time_dim=time_dim,
                )
                for level in range(depth - 2, -1, -1)
            ]
        )
        self.output_projection = nn.Sequential(
            nn.GroupNorm(_group_norm_groups(self.channel_levels[0]), self.channel_levels[0]),
            nn.SiLU(),
            nn.Conv2d(self.channel_levels[0], image_channels, kernel_size=3, padding=1),
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
        x = self.input_projection(x_t)
        skips: list[torch.Tensor] = []

        for encoder_block in self.encoder_blocks:
            x, skip = encoder_block(x, time_features)
            skips.append(skip)

        for block in self.middle_blocks:
            x = block(x, time_features)

        for decoder_block, skip in zip(self.decoder_blocks, reversed(skips)):
            x = decoder_block(x, skip, time_features)

        return self.output_projection(x)


def build_denoiser(
    image_channels: int,
    hidden_channels: int,
    depth: int,
    num_time_frequencies: int,
) -> DenoiserUNet:
    return DenoiserUNet(
        image_channels=image_channels,
        hidden_channels=hidden_channels,
        depth=depth,
        num_time_frequencies=num_time_frequencies,
    )


# Backward-compatible alias for the old name used across the package.
DenoiserCNN = DenoiserUNet
