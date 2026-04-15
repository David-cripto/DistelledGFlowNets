from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from .plots import _draw_density_panel, _grid_points, _save_figure


@torch.no_grad()
def reward_log_density_on_grid(
    reward_model: torch.nn.Module,
    limits: Tuple[float, float, float, float],
    resolution: int = 220,
    time_value: float = 0.0,
    device: str = "cpu",
    batch_size: int = 8192,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs, ys, points = _grid_points(limits, resolution)
    flat_values = np.zeros(points.shape[0], dtype=np.float32)
    tensor_points = torch.from_numpy(points).to(device=device, dtype=torch.float32)

    reward_model = reward_model.to(device).eval()
    for start in range(0, tensor_points.shape[0], batch_size):
        stop = start + batch_size
        batch = tensor_points[start:stop]
        times = torch.full((batch.shape[0],), time_value, device=batch.device)
        flat_values[start:stop] = reward_model(batch, times).cpu().numpy()

    return xs, ys, flat_values.reshape(resolution, resolution)


def normalize_log_density_grid(
    log_density: np.ndarray,
    limits: Tuple[float, float, float, float],
) -> np.ndarray:
    shifted = log_density - float(np.max(log_density))
    density = np.exp(shifted)
    dx = (limits[1] - limits[0]) / max(log_density.shape[0] - 1, 1)
    dy = (limits[3] - limits[2]) / max(log_density.shape[1] - 1, 1)
    mass = density.sum() * dx * dy
    return density / max(mass, 1e-12)


def density_grid_metrics(
    target_density: np.ndarray,
    estimate_density: np.ndarray,
    limits: Tuple[float, float, float, float],
) -> Dict[str, float]:
    dx = (limits[1] - limits[0]) / max(target_density.shape[0] - 1, 1)
    dy = (limits[3] - limits[2]) / max(target_density.shape[1] - 1, 1)
    target = target_density / max(target_density.sum() * dx * dy, 1e-12)
    estimate = estimate_density / max(estimate_density.sum() * dx * dy, 1e-12)
    eps = 1e-12
    target_safe = np.clip(target, eps, None)
    estimate_safe = np.clip(estimate, eps, None)
    return {
        "l1": float(np.sum(np.abs(target_safe - estimate_safe)) * dx * dy),
        "kl_target_to_estimate": float(
            np.sum(target_safe * (np.log(target_safe) - np.log(estimate_safe))) * dx * dy
        ),
        "max_abs": float(np.max(np.abs(target_safe - estimate_safe))),
        "estimate_mass": float(estimate_density.sum() * dx * dy),
    }


def plot_reward_density_comparison(
    target_density: np.ndarray,
    kde_density: np.ndarray,
    reward_density: np.ndarray,
    limits: Tuple[float, float, float, float],
    out_path: Path,
    estimate_title: str = "Density from exp(R)",
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    _draw_density_panel(axes[0], target_density, limits, "Ground-truth density")
    _draw_density_panel(axes[1], kde_density, limits, "Diffusion sample KDE")
    _draw_density_panel(axes[2], reward_density, limits, estimate_title)
    _save_figure(fig, out_path)


def plot_reward_value_field(
    reward_log_density: np.ndarray,
    limits: Tuple[float, float, float, float],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    image = ax.imshow(
        reward_log_density.T,
        origin="lower",
        extent=limits,
        interpolation="bilinear",
        aspect="equal",
    )
    ax.set_title("Reward field R(x)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    _save_figure(fig, out_path)


def plot_reward_training_curves(history: List[Dict[str, float]], out_path: Path) -> None:
    steps = [row["step"] for row in history]
    loss = [row["loss"] for row in history]
    consistency = [row["consistency_loss"] for row in history]
    gradient_penalty = [row["gradient_penalty"] for row in history]
    reward_l1 = [row["reward_density_l1"] for row in history]

    fig, axes = plt.subplots(1, 4, figsize=(17, 4))
    axes[0].plot(steps, loss)
    axes[0].set_title("Total loss")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("loss")

    axes[1].plot(steps, consistency)
    axes[1].set_title("Transition consistency")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("MSE")

    axes[2].plot(steps, gradient_penalty)
    axes[2].set_title("Gradient penalty")
    axes[2].set_xlabel("step")
    axes[2].set_ylabel("penalty")

    axes[3].plot(steps, reward_l1)
    axes[3].set_title("Grid L1 vs target")
    axes[3].set_xlabel("step")
    axes[3].set_ylabel("L1")

    _save_figure(fig, out_path)
