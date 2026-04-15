from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from ..distributions import Density2D


def _save_figure(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def combine_plot_limits(*limits: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    xmin = min(limit[0] for limit in limits)
    xmax = max(limit[1] for limit in limits)
    ymin = min(limit[2] for limit in limits)
    ymax = max(limit[3] for limit in limits)
    padding = 0.08 * max(xmax - xmin, ymax - ymin)
    return (xmin - padding, xmax + padding, ymin - padding, ymax + padding)


def _grid_points(
    limits: Tuple[float, float, float, float],
    resolution: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.linspace(limits[0], limits[1], resolution)
    ys = np.linspace(limits[2], limits[3], resolution)
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")
    points = np.stack([grid_x.reshape(-1), grid_y.reshape(-1)], axis=-1)
    return xs, ys, points


@torch.no_grad()
def density_on_grid(
    distribution: Density2D,
    limits: Tuple[float, float, float, float],
    resolution: int = 220,
    device: str = "cpu",
    batch_size: int = 8192,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs, ys, points = _grid_points(limits, resolution)
    flat_density = np.zeros(points.shape[0], dtype=np.float32)
    tensor_points = torch.from_numpy(points).to(device=device, dtype=torch.float32)

    for start in range(0, tensor_points.shape[0], batch_size):
        stop = start + batch_size
        batch = tensor_points[start:stop]
        flat_density[start:stop] = torch.exp(distribution.log_prob(batch)).cpu().numpy()

    return xs, ys, flat_density.reshape(resolution, resolution)


def _default_bandwidth(samples: np.ndarray) -> float:
    sample_std = float(np.std(samples, axis=0, ddof=1).mean())
    sample_std = max(sample_std, 1e-2)
    return max(1e-2, 1.06 * sample_std * samples.shape[0] ** (-1.0 / 6.0))


def kde_on_grid(
    samples: np.ndarray,
    limits: Tuple[float, float, float, float],
    resolution: int = 220,
    bandwidth: Optional[float] = None,
    chunk_size: int = 1024,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if samples.ndim != 2 or samples.shape[1] != 2:
        raise ValueError("samples must have shape [num_samples, 2]")

    xs, ys, points = _grid_points(limits, resolution)
    bandwidth = bandwidth or _default_bandwidth(samples)
    normalization = 1.0 / (samples.shape[0] * 2.0 * np.pi * bandwidth * bandwidth)
    flat_density = np.zeros(points.shape[0], dtype=np.float32)

    for start in range(0, points.shape[0], chunk_size):
        stop = start + chunk_size
        chunk = points[start:stop]
        diff = chunk[:, None, :] - samples[None, :, :]
        squared_distance = np.sum(diff * diff, axis=-1)
        kernel_values = np.exp(-0.5 * squared_distance / (bandwidth * bandwidth))
        flat_density[start:stop] = normalization * kernel_values.sum(axis=1)

    return xs, ys, flat_density.reshape(resolution, resolution)


def _draw_density_panel(
    ax: plt.Axes,
    density: np.ndarray,
    limits: Tuple[float, float, float, float],
    title: str,
) -> None:
    image = ax.imshow(
        density.T,
        origin="lower",
        extent=limits,
        interpolation="bilinear",
        aspect="equal",
    )
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)


def plot_exact_density(
    distribution: Density2D,
    title: str,
    out_path: Path,
    limits: Tuple[float, float, float, float],
    resolution: int = 220,
) -> None:
    _, _, density = density_on_grid(distribution, limits=limits, resolution=resolution)
    fig, ax = plt.subplots(figsize=(5, 5))
    _draw_density_panel(ax, density, limits, title)
    _save_figure(fig, out_path)


def plot_sample_kde(
    samples: np.ndarray,
    title: str,
    out_path: Path,
    limits: Tuple[float, float, float, float],
    resolution: int = 220,
    bandwidth: Optional[float] = None,
) -> None:
    _, _, density = kde_on_grid(samples, limits=limits, resolution=resolution, bandwidth=bandwidth)
    fig, ax = plt.subplots(figsize=(5, 5))
    _draw_density_panel(ax, density, limits, title)
    _save_figure(fig, out_path)


def plot_density_triptych(
    reference_distribution: Density2D,
    target_distribution: Density2D,
    model_samples: np.ndarray,
    out_path: Path,
    limits: Tuple[float, float, float, float],
    resolution: int = 220,
) -> None:
    _, _, ref_density = density_on_grid(reference_distribution, limits=limits, resolution=resolution)
    _, _, target_density = density_on_grid(target_distribution, limits=limits, resolution=resolution)
    _, _, sample_density = kde_on_grid(model_samples, limits=limits, resolution=resolution)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    _draw_density_panel(axes[0], ref_density, limits, "Reference density")
    _draw_density_panel(axes[1], target_density, limits, "Target density")
    _draw_density_panel(axes[2], sample_density, limits, "Model sample KDE")
    _save_figure(fig, out_path)


def plot_trajectories(
    trajectory: np.ndarray,
    times: np.ndarray,
    out_path: Path,
    limits: Tuple[float, float, float, float],
    max_paths: int = 160,
) -> None:
    if trajectory.ndim != 3 or trajectory.shape[-1] != 2:
        raise ValueError("trajectory must have shape [num_times, num_samples, 2]")
    if trajectory.shape[0] != times.shape[0]:
        raise ValueError("times and trajectory length must match")

    num_samples = trajectory.shape[1]
    count = min(max_paths, num_samples)
    indices = np.linspace(0, num_samples - 1, count, dtype=int)
    subset = trajectory[:, indices, :]

    fig, ax = plt.subplots(figsize=(6, 6))
    for path in np.transpose(subset, (1, 0, 2)):
        ax.plot(path[:, 0], path[:, 1], color="#4C72B0", alpha=0.22, linewidth=0.9)

    ax.scatter(subset[0, :, 0], subset[0, :, 1], s=16, color="#DD8452", alpha=0.85, label="t=1.0")
    ax.scatter(subset[-1, :, 0], subset[-1, :, 1], s=16, color="#55A868", alpha=0.85, label="t=0.0")
    ax.set_xlim(limits[0], limits[1])
    ax.set_ylim(limits[2], limits[3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Reverse trajectories across {len(times)} time points")
    ax.legend(loc="upper right")
    _save_figure(fig, out_path)
