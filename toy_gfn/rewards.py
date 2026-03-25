from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np
import torch


RewardBuilder = Callable[[int, float], torch.Tensor]


def _coordinate_grid(grid_size: int):
    xs = np.linspace(0.0, 1.0, grid_size + 1)
    ys = np.linspace(0.0, 1.0, grid_size + 1)
    return np.meshgrid(xs, ys, indexing="ij")


def _gaussian(
    x: np.ndarray,
    y: np.ndarray,
    mean_x: float,
    mean_y: float,
    sigma_x: float,
    sigma_y: float,
    scale: float = 1.0,
):
    return scale * np.exp(
        -0.5 * (((x - mean_x) / sigma_x) ** 2 + ((y - mean_y) / sigma_y) ** 2)
    )


def mixture_reward(grid_size: int, reward_floor: float = 1e-4):
    x, y = _coordinate_grid(grid_size)
    reward = _gaussian(x, y, 0.22, 0.25, 0.10, 0.10, 1.0)
    reward += _gaussian(x, y, 0.76, 0.23, 0.12, 0.08, 1.35)
    reward += _gaussian(x, y, 0.58, 0.80, 0.10, 0.10, 0.95)
    reward += reward_floor
    return torch.tensor(reward, dtype=torch.float32)


def ring_reward(grid_size: int, reward_floor: float = 1e-4):
    x, y = _coordinate_grid(grid_size)
    radius = np.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2)
    reward = np.exp(-0.5 * ((radius - 0.32) / 0.07) ** 2)
    reward += reward_floor
    return torch.tensor(reward, dtype=torch.float32)


def corners_reward(grid_size: int, reward_floor: float = 1e-4):
    x, y = _coordinate_grid(grid_size)
    reward = np.zeros_like(x)
    for mean_x, mean_y in [(0.08, 0.08), (0.08, 0.92), (0.92, 0.08), (0.92, 0.92)]:
        reward += _gaussian(x, y, mean_x, mean_y, 0.08, 0.08, 1.0)
    reward += reward_floor
    return torch.tensor(reward, dtype=torch.float32)


_REWARD_BUILDERS: Dict[str, RewardBuilder] = {
    "mixture": mixture_reward,
    "ring": ring_reward,
    "corners": corners_reward,
}


def available_rewards():
    return tuple(_REWARD_BUILDERS.keys())


def build_reward_grid(name: str, grid_size: int, reward_floor: float = 1e-4):
    if name not in _REWARD_BUILDERS:
        known = ", ".join(available_rewards())
        raise ValueError(f"Unknown reward '{name}'. Expected one of: {known}")
    return _REWARD_BUILDERS[name](grid_size=grid_size, reward_floor=reward_floor)
