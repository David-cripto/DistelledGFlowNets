from __future__ import annotations

import math
from typing import Iterable, Tuple

from .base import Density2D
from .gaussians import DiagonalGaussian2D, GaussianMixture2D


def _ring_centers(num_components: int, radius: float) -> Iterable[tuple[float, float]]:
    for idx in range(num_components):
        angle = 2.0 * math.pi * idx / num_components
        yield (radius * math.cos(angle), radius * math.sin(angle))


def _two_moons_centers(num_components: int, radius: float) -> Iterable[tuple[float, float]]:
    half = num_components // 2
    for idx in range(half):
        angle = math.pi * idx / max(1, half - 1)
        yield (radius * math.cos(angle), radius * math.sin(angle))
    for idx in range(num_components - half):
        angle = math.pi * idx / max(1, num_components - half - 1)
        yield (radius - radius * math.cos(angle), -0.55 - radius * math.sin(angle))


def build_distribution(name: str) -> Density2D:
    key = name.lower()
    if key == "gaussian":
        return DiagonalGaussian2D(std=(1.0, 1.0), name="gaussian")
    if key == "wide_gaussian":
        return DiagonalGaussian2D(std=(1.8, 1.8), name="wide_gaussian")
    if key == "shifted_gaussian":
        return DiagonalGaussian2D(mean=(-1.5, 1.0), std=(0.6, 0.9), name="shifted_gaussian")
    if key == "mixture4":
        return GaussianMixture2D(
            centers=[(-2.0, -2.0), (-2.0, 2.0), (2.0, -2.0), (2.0, 2.0)],
            std=0.35,
            name="mixture4",
        )
    if key == "grid9":
        return GaussianMixture2D(
            centers=[
                (-2.0, -2.0),
                (-2.0, 0.0),
                (-2.0, 2.0),
                (0.0, -2.0),
                (0.0, 0.0),
                (0.0, 2.0),
                (2.0, -2.0),
                (2.0, 0.0),
                (2.0, 2.0),
            ],
            std=0.28,
            name="grid9",
        )
    if key == "eight_gaussians":
        return GaussianMixture2D(
            centers=list(_ring_centers(num_components=8, radius=2.4)),
            std=0.22,
            name="eight_gaussians",
        )
    if key == "ring":
        return GaussianMixture2D(
            centers=list(_ring_centers(num_components=24, radius=2.6)),
            std=0.16,
            name="ring",
        )
    if key == "two_moons":
        return GaussianMixture2D(
            centers=list(_two_moons_centers(num_components=32, radius=1.8)),
            std=0.12,
            name="two_moons",
        )
    raise ValueError(f"Unknown distribution '{name}'")


def available_distributions() -> Tuple[str, ...]:
    return (
        "gaussian",
        "wide_gaussian",
        "shifted_gaussian",
        "mixture4",
        "grid9",
        "eight_gaussians",
        "ring",
        "two_moons",
    )
