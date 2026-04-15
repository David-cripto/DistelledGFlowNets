from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from ..diffusion.model import DenoiserMLP
from ..diffusion.schedules import DDPMSchedule
from ..diffusion.training import TrainConfig
from ..distributions import Density2D, build_distribution
from ..visualization import combine_plot_limits, density_grid_metrics, density_on_grid
from ..visualization.plots import _draw_density_panel, _grid_points, _save_figure

@dataclass
class CompatibilityCheckConfig:
    grid_resolution: int = 80
    kernel_chunk_size: int = 256
    model_batch_size: int = 4096
    num_plot_steps: int = 4
    limit_scale: float = 1.25
    device: str = "cpu"


@dataclass
class CompatibilityCheckResult:
    history: List[Dict[str, float]]
    times: np.ndarray
    selected_steps: np.ndarray
    limits: Tuple[float, float, float, float]
    grid_x: np.ndarray
    grid_y: np.ndarray
    target_density: np.ndarray
    backward_densities: np.ndarray
    roundtrip_densities: np.ndarray


def _expand_limits(
    limits: Tuple[float, float, float, float],
    scale: float,
) -> Tuple[float, float, float, float]:
    if scale <= 0.0:
        raise ValueError("limit_scale must be positive")
    center_x = 0.5 * (limits[0] + limits[1])
    center_y = 0.5 * (limits[2] + limits[3])
    half_width = 0.5 * (limits[1] - limits[0]) * scale
    half_height = 0.5 * (limits[3] - limits[2]) * scale
    return (
        center_x - half_width,
        center_x + half_width,
        center_y - half_height,
        center_y + half_height,
    )


def _grid_area(limits: Tuple[float, float, float, float], resolution: int) -> float:
    dx = (limits[1] - limits[0]) / max(resolution - 1, 1)
    dy = (limits[3] - limits[2]) / max(resolution - 1, 1)
    return dx * dy


def _grid_mass(density: np.ndarray, limits: Tuple[float, float, float, float]) -> float:
    return float(density.sum() * _grid_area(limits, density.shape[0]))


def _raw_density_metrics(
    reference_density: np.ndarray,
    estimate_density: np.ndarray,
    limits: Tuple[float, float, float, float],
) -> Dict[str, float]:
    area = _grid_area(limits, reference_density.shape[0])
    difference = estimate_density - reference_density
    return {
        "l1_raw": float(np.abs(difference).sum() * area),
        "l2_raw": float(np.sqrt((difference * difference).sum() * area)),
        "max_abs_raw": float(np.abs(difference).max()),
        "reference_mass": _grid_mass(reference_density, limits),
        "estimate_mass": _grid_mass(estimate_density, limits),
        "mass_abs_diff": abs(_grid_mass(estimate_density, limits) - _grid_mass(reference_density, limits)),
    }


def _cell_edges(values: np.ndarray) -> np.ndarray:
    if values.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if values.shape[0] == 1:
        step = 1.0
        return np.array([values[0] - 0.5 * step, values[0] + 0.5 * step], dtype=np.float32)
    midpoints = 0.5 * (values[:-1] + values[1:])
    left_edge = values[0] - 0.5 * (values[1] - values[0])
    right_edge = values[-1] + 0.5 * (values[-1] - values[-2])
    return np.concatenate(
        [
            np.array([left_edge], dtype=np.float32),
            midpoints.astype(np.float32),
            np.array([right_edge], dtype=np.float32),
        ]
    )


def _normal_cdf(z: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))


@torch.no_grad()
def _reverse_kernel_means(
    denoiser: DenoiserMLP,
    points: torch.Tensor,
    step: int,
    schedule: DDPMSchedule,
    batch_size: int,
) -> torch.Tensor:
    means = []
    for start in range(0, points.shape[0], batch_size):
        stop = start + batch_size
        batch = points[start:stop]
        timesteps = torch.full((batch.shape[0],), step, device=batch.device, dtype=torch.long)
        noise_prediction = denoiser(batch, schedule.time_values(timesteps))
        x0_prediction = schedule.predict_x0(batch, timesteps, noise_prediction)
        means.append(schedule.posterior_mean(x0_prediction, batch, timesteps))
    return torch.cat(means, dim=0)


def _forward_kernel_means(
    points: torch.Tensor,
    step: int,
    schedule: DDPMSchedule,
) -> torch.Tensor:
    alpha = float(schedule.alphas[step].cpu())
    return math.sqrt(alpha) * points


@torch.no_grad()
def _propagate_density(
    source_density: np.ndarray,
    source_means: torch.Tensor,
    variance: float,
    x_edges: torch.Tensor,
    y_edges: torch.Tensor,
    limits: Tuple[float, float, float, float],
    resolution: int,
    chunk_size: int,
) -> np.ndarray:
    if variance <= 0.0:
        raise ValueError("variance must be positive")

    device = source_means.device
    source_weights = torch.from_numpy(source_density.reshape(-1)).to(device=device, dtype=torch.float32)
    source_weights = source_weights * _grid_area(limits, resolution)
    target_mass = torch.zeros((resolution, resolution), device=device, dtype=torch.float32)
    std = math.sqrt(variance)
    if std <= 0.0:
        raise ValueError("standard deviation must be positive")
    x_edges = x_edges.to(device=device, dtype=torch.float32)
    y_edges = y_edges.to(device=device, dtype=torch.float32)

    for start in range(0, source_means.shape[0], chunk_size):
        stop = start + chunk_size
        mean_chunk = source_means[start:stop]
        weight_chunk = source_weights[start:stop]
        if float(weight_chunk.abs().sum().cpu()) == 0.0:
            continue

        mean_x = mean_chunk[:, 0].unsqueeze(0)
        mean_y = mean_chunk[:, 1].unsqueeze(0)
        x_probs = _normal_cdf((x_edges[1:].unsqueeze(1) - mean_x) / std) - _normal_cdf(
            (x_edges[:-1].unsqueeze(1) - mean_x) / std
        )
        y_probs = _normal_cdf((y_edges[1:].unsqueeze(1) - mean_y) / std) - _normal_cdf(
            (y_edges[:-1].unsqueeze(1) - mean_y) / std
        )
        target_mass += torch.einsum("im,jm,m->ij", x_probs, y_probs, weight_chunk)

    return (target_mass / _grid_area(limits, resolution)).cpu().numpy().astype(np.float32)


def _selected_steps(num_steps: int, num_plot_steps: int) -> np.ndarray:
    count = max(1, min(num_steps, num_plot_steps))
    steps = np.linspace(1, num_steps, count)
    rounded = np.unique(np.round(steps).astype(np.int64))
    if rounded[0] != 1:
        rounded[0] = 1
    if rounded[-1] != num_steps:
        rounded[-1] = num_steps
    return rounded


def run_compatibility_check(
    config: CompatibilityCheckConfig,
    denoiser: DenoiserMLP,
    reference_distribution: Density2D,
    target_distribution: Density2D,
    num_sample_steps: int,
) -> CompatibilityCheckResult:
    device = torch.device(config.device)
    schedule = DDPMSchedule(num_steps=num_sample_steps)
    denoiser = denoiser.to(device).eval()
    for parameter in denoiser.parameters():
        parameter.requires_grad_(False)

    base_limits = combine_plot_limits(
        reference_distribution.default_limits(),
        target_distribution.default_limits(),
    )
    limits = _expand_limits(base_limits, config.limit_scale)
    grid_x, grid_y, points_np = _grid_points(limits, config.grid_resolution)
    x_edges = torch.from_numpy(_cell_edges(grid_x))
    y_edges = torch.from_numpy(_cell_edges(grid_y))
    points = torch.from_numpy(points_np).to(device=device, dtype=torch.float32)

    _, _, reference_density = density_on_grid(
        reference_distribution,
        limits=limits,
        resolution=config.grid_resolution,
        device=str(device),
    )
    _, _, target_density = density_on_grid(
        target_distribution,
        limits=limits,
        resolution=config.grid_resolution,
        device=str(device),
    )

    backward_densities = np.zeros((num_sample_steps + 1, config.grid_resolution, config.grid_resolution), dtype=np.float32)
    roundtrip_densities = np.full_like(backward_densities, np.nan)
    backward_densities[num_sample_steps] = reference_density.astype(np.float32)

    history: List[Dict[str, float]] = []

    for step in range(num_sample_steps - 1, -1, -1):
        reverse_means = _reverse_kernel_means(
            denoiser=denoiser,
            points=points,
            step=step,
            schedule=schedule,
            batch_size=config.model_batch_size,
        )
        reverse_variance = float(schedule.betas[step].cpu())
        backward_densities[step] = _propagate_density(
            source_density=backward_densities[step + 1],
            source_means=reverse_means,
            variance=reverse_variance,
            x_edges=x_edges,
            y_edges=y_edges,
            limits=limits,
            resolution=config.grid_resolution,
            chunk_size=config.kernel_chunk_size,
        ).astype(np.float32)

        forward_means = _forward_kernel_means(
            points=points,
            step=step,
            schedule=schedule,
        )
        forward_variance = float(schedule.betas[step].cpu())
        roundtrip_densities[step + 1] = _propagate_density(
            source_density=backward_densities[step],
            source_means=forward_means,
            variance=forward_variance,
            x_edges=x_edges,
            y_edges=y_edges,
            limits=limits,
            resolution=config.grid_resolution,
            chunk_size=config.kernel_chunk_size,
        ).astype(np.float32)

        raw_metrics = _raw_density_metrics(
            reference_density=backward_densities[step + 1],
            estimate_density=roundtrip_densities[step + 1],
            limits=limits,
        )
        normalized_metrics = density_grid_metrics(
            target_density=backward_densities[step + 1],
            estimate_density=roundtrip_densities[step + 1],
            limits=limits,
        )
        row = {
            "step": float(step + 1),
            "time": float(step + 1) / float(num_sample_steps),
            **raw_metrics,
            "l1_normalized": normalized_metrics["l1"],
            "kl_normalized": normalized_metrics["kl_target_to_estimate"],
            "max_abs_normalized": normalized_metrics["max_abs"],
        }
        history.append(row)
        print(
            f"compat_step={step + 1:5d} "
            f"time={row['time']:.3f} "
            f"l1_raw={row['l1_raw']:.6f} "
            f"l1_norm={row['l1_normalized']:.6f} "
            f"mass={row['reference_mass']:.6f}->{row['estimate_mass']:.6f}"
        )

    times = np.arange(num_sample_steps + 1, dtype=np.float32) / float(num_sample_steps)
    selected_steps = _selected_steps(num_sample_steps, config.num_plot_steps)
    return CompatibilityCheckResult(
        history=history,
        times=times,
        selected_steps=selected_steps,
        limits=limits,
        grid_x=grid_x,
        grid_y=grid_y,
        target_density=target_density.astype(np.float32),
        backward_densities=backward_densities,
        roundtrip_densities=roundtrip_densities,
    )


def _plot_step_metrics(history: List[Dict[str, float]], out_path: Path) -> None:
    steps = [row["step"] for row in history]
    l1_raw = [row["l1_raw"] for row in history]
    l1_normalized = [row["l1_normalized"] for row in history]
    kl_normalized = [row["kl_normalized"] for row in history]
    original_mass = [row["reference_mass"] for row in history]
    roundtrip_mass = [row["estimate_mass"] for row in history]
    mass_abs_diff = [row["mass_abs_diff"] for row in history]
    max_abs_raw = [row["max_abs_raw"] for row in history]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(steps, l1_raw, label="raw L1")
    axes[0, 0].plot(steps, max_abs_raw, label="raw max abs")
    axes[0, 0].set_title("Raw Density Errors")
    axes[0, 0].set_xlabel("time index")
    axes[0, 0].legend()

    axes[0, 1].plot(steps, l1_normalized, label="normalized L1")
    axes[0, 1].plot(steps, kl_normalized, label="normalized KL")
    axes[0, 1].set_title("Normalized Density Errors")
    axes[0, 1].set_xlabel("time index")
    axes[0, 1].legend()

    axes[1, 0].plot(steps, original_mass, label="pi_t")
    axes[1, 0].plot(steps, roundtrip_mass, label="q # pi_{t-1}")
    axes[1, 0].set_title("Mass on Grid")
    axes[1, 0].set_xlabel("time index")
    axes[1, 0].legend()

    axes[1, 1].plot(steps, mass_abs_diff)
    axes[1, 1].set_title("Mass Absolute Difference")
    axes[1, 1].set_xlabel("time index")
    axes[1, 1].set_ylabel("|m_q - m_pi|")

    _save_figure(fig, out_path)


def _plot_selected_steps(
    result: CompatibilityCheckResult,
    out_path: Path,
) -> None:
    steps = result.selected_steps.tolist()
    fig, axes = plt.subplots(len(steps), 3, figsize=(12, 4 * len(steps)))
    axes = np.atleast_2d(axes)

    for row, step in enumerate(steps):
        original = result.backward_densities[step]
        roundtrip = result.roundtrip_densities[step]
        difference = np.abs(roundtrip - original)
        time_value = result.times[step]

        _draw_density_panel(
            axes[row, 0],
            original,
            result.limits,
            title=f"$\\pi_t$ at t={time_value:.3f}",
        )
        _draw_density_panel(
            axes[row, 1],
            roundtrip,
            result.limits,
            title=f"$q_\\# \\pi_{{t-1}}$ at t={time_value:.3f}",
        )
        _draw_density_panel(
            axes[row, 2],
            difference,
            result.limits,
            title=f"|difference| at t={time_value:.3f}",
        )

    _save_figure(fig, out_path)


def _plot_terminal_vs_target(
    result: CompatibilityCheckResult,
    out_path: Path,
) -> None:
    terminal = result.backward_densities[0]
    difference = np.abs(terminal - result.target_density)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    _draw_density_panel(axes[0], result.target_density, result.limits, "Exact target density")
    _draw_density_panel(axes[1], terminal, result.limits, "Backward terminal density")
    _draw_density_panel(axes[2], difference, result.limits, "|terminal - target|")
    _save_figure(fig, out_path)


def save_compatibility_artifacts(
    result: CompatibilityCheckResult,
    config: CompatibilityCheckConfig,
    diffusion_config: Dict[str, object],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "compatibility_config": asdict(config),
        "diffusion_config": diffusion_config,
        "history": result.history,
    }

    np.save(out_dir / "grid_x.npy", result.grid_x)
    np.save(out_dir / "grid_y.npy", result.grid_y)
    np.save(out_dir / "times.npy", result.times)
    np.save(out_dir / "selected_steps.npy", result.selected_steps)
    np.save(out_dir / "target_density.npy", result.target_density)
    np.save(out_dir / "backward_densities.npy", result.backward_densities)
    np.save(out_dir / "roundtrip_densities.npy", result.roundtrip_densities)

    with open(out_dir / "compatibility_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    _plot_step_metrics(result.history, out_dir / "compatibility_step_metrics.png")
    _plot_selected_steps(result, out_dir / "compatibility_selected_steps.png")
    _plot_terminal_vs_target(result, out_dir / "terminal_vs_target.png")


def load_diffusion_checkpoint(
    checkpoint_path: Path,
    device: str = "cpu",
) -> tuple[DenoiserMLP, Density2D, Density2D, TrainConfig]:
    payload = torch.load(checkpoint_path, map_location=device)
    config = TrainConfig(**payload["config"])
    model = DenoiserMLP(
        hidden_dim=config.hidden_dim,
        depth=config.depth,
        num_time_frequencies=config.time_frequencies,
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    reference_distribution = build_distribution(config.reference)
    target_distribution = build_distribution(config.target)
    return model, reference_distribution, target_distribution, config
