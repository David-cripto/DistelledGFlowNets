from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple, Union

import torch

from .base import Density2D


TensorLike = Union[torch.Tensor, Sequence[float], float]


def _as_point(value: TensorLike) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if tensor.ndim == 0:
        tensor = tensor.repeat(2)
    if tensor.shape != (2,):
        raise ValueError("Expected a scalar or a 2D vector")
    return tensor


class DiagonalGaussian2D(Density2D):
    def __init__(
        self,
        mean: TensorLike = (0.0, 0.0),
        std: TensorLike = (1.0, 1.0),
        name: str = "gaussian",
    ):
        self.mean = _as_point(mean)
        self.std = _as_point(std).clamp_min(1e-4)
        self.name = name

    def sample(
        self,
        num_samples: int,
        device: Union[str, torch.device] = "cpu",
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        mean = self.mean.to(device)
        std = self.std.to(device)
        noise = torch.randn((num_samples, 2), device=device, generator=generator)
        return mean.unsqueeze(0) + std.unsqueeze(0) * noise

    def log_prob(self, points: torch.Tensor) -> torch.Tensor:
        points = points.to(dtype=torch.float32)
        mean = self.mean.to(points.device)
        std = self.std.to(points.device)
        normalized = (points - mean.unsqueeze(0)) / std.unsqueeze(0)
        log_scale = torch.log(std).sum()
        return -0.5 * normalized.pow(2).sum(dim=-1) - log_scale - math.log(2.0 * math.pi)

    def default_limits(self) -> Tuple[float, float, float, float]:
        padding = 4.0 * float(self.std.max())
        x0, y0 = self.mean.tolist()
        return (x0 - padding, x0 + padding, y0 - padding, y0 + padding)


class GaussianMixture2D(Density2D):
    def __init__(
        self,
        centers: Sequence[Sequence[float]],
        std: TensorLike,
        weights: Optional[Sequence[float]] = None,
        name: str = "gaussian_mixture",
    ):
        self.centers = torch.as_tensor(centers, dtype=torch.float32)
        if self.centers.ndim != 2 or self.centers.shape[1] != 2:
            raise ValueError("centers must have shape [num_components, 2]")

        std_tensor = torch.as_tensor(std, dtype=torch.float32)
        if std_tensor.ndim == 0:
            std_tensor = std_tensor.repeat(self.centers.shape[0], 2)
        elif std_tensor.ndim == 1:
            if std_tensor.shape[0] == 2:
                std_tensor = std_tensor.unsqueeze(0).repeat(self.centers.shape[0], 1)
            elif std_tensor.shape[0] == self.centers.shape[0]:
                std_tensor = std_tensor.unsqueeze(-1).repeat(1, 2)
            else:
                raise ValueError("std must broadcast to [num_components, 2]")
        elif std_tensor.ndim == 2 and std_tensor.shape != self.centers.shape:
            raise ValueError("std must broadcast to [num_components, 2]")

        self.std = std_tensor.clamp_min(1e-4)

        if weights is None:
            probs = torch.ones(self.centers.shape[0], dtype=torch.float32)
        else:
            probs = torch.as_tensor(weights, dtype=torch.float32)
            if probs.ndim != 1 or probs.shape[0] != self.centers.shape[0]:
                raise ValueError("weights must have shape [num_components]")
        self.weights = probs / probs.sum()
        self.log_weights = torch.log(self.weights)
        self.name = name

    def sample(
        self,
        num_samples: int,
        device: Union[str, torch.device] = "cpu",
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        weights = self.weights.to(device)
        indices = torch.multinomial(weights, num_samples, replacement=True, generator=generator)
        centers = self.centers.to(device)[indices]
        std = self.std.to(device)[indices]
        noise = torch.randn((num_samples, 2), device=device, generator=generator)
        return centers + std * noise

    def log_prob(self, points: torch.Tensor) -> torch.Tensor:
        points = points.to(dtype=torch.float32)
        centers = self.centers.to(points.device)
        std = self.std.to(points.device)
        log_weights = self.log_weights.to(points.device)

        normalized = (points.unsqueeze(1) - centers.unsqueeze(0)) / std.unsqueeze(0)
        component_log_prob = (
            -0.5 * normalized.pow(2).sum(dim=-1)
            - torch.log(std).sum(dim=-1).unsqueeze(0)
            - math.log(2.0 * math.pi)
        )
        return torch.logsumexp(component_log_prob + log_weights.unsqueeze(0), dim=1)

    def default_limits(self) -> Tuple[float, float, float, float]:
        padding = 4.0 * float(self.std.max())
        minimum = self.centers.min(dim=0).values
        maximum = self.centers.max(dim=0).values
        return (
            float(minimum[0] - padding),
            float(maximum[0] + padding),
            float(minimum[1] - padding),
            float(maximum[1] + padding),
        )
