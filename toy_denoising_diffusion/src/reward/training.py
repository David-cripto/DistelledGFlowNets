from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from ..diffusion.model import DenoiserMLP
from ..diffusion.schedules import DDPMSchedule
from ..diffusion.training import sample_model_samples
from ..distributions import Density2D
from ..visualization import (
    combine_plot_limits,
    density_grid_metrics,
    density_on_grid,
    kde_on_grid,
    normalize_log_density_grid,
    plot_reward_density_comparison,
)
from ..visualization.plots import _grid_points, _save_figure
from .model import DetailedBalanceModel, build_detailed_balance_model


torch.set_num_threads(1)

_LOG_2PI = math.log(2.0 * math.pi)


@dataclass
class DetailedBalanceTrainConfig:
    train_steps: int = 2000
    batch_num_trajectories: int = 128
    lr: float = 1e-3
    seed: int = 0
    model_type: str = "direct"
    hidden_dim: int = 128
    depth: int = 3
    time_frequencies: int = 8
    boundary_penalty_weight: float = 1.0
    eval_every: int = 100
    num_sample_steps: int = 64
    num_kde_samples: int = 4096
    density_resolution: int = 220
    device: str = "cpu"


@dataclass
class DetailedBalanceTrainResult:
    model: DetailedBalanceModel
    history: List[Dict[str, float]]
    limits: Tuple[float, float, float, float]
    grid_x: np.ndarray
    grid_y: np.ndarray
    reference_density: np.ndarray
    target_density: np.ndarray
    kde_density: np.ndarray
    diffusion_samples: np.ndarray
    boundary_log_score: np.ndarray
    boundary_score: np.ndarray
    terminal_log_score: np.ndarray
    terminal_score: np.ndarray
    terminal_score_density: np.ndarray


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def _sample_transition_pairs(
    denoiser: DenoiserMLP,
    reference_distribution: Density2D,
    num_trajectories: int,
    schedule: DDPMSchedule,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    current = reference_distribution.sample(num_trajectories, device=device)

    denoiser.eval()
    x_t_list = []
    x_prev_list = []
    timestep_list = []
    t_now_list = []
    t_prev_list = []

    for step in range(schedule.num_steps - 1, -1, -1):
        timesteps = torch.full((num_trajectories,), step, device=device, dtype=torch.long)
        t_now = schedule.time_values(timesteps)
        t_prev = torch.full((num_trajectories,), float(step) / float(schedule.num_steps), device=device)
        reverse_mean, reverse_variance = _reverse_kernel_statistics(
            denoiser=denoiser,
            x_t=current,
            timesteps=timesteps,
            schedule=schedule,
        )
        reverse_std = torch.sqrt(reverse_variance.clamp_min(schedule.epsilon))
        previous = reverse_mean + reverse_std * torch.randn_like(current)

        x_t_list.append(current.detach())
        x_prev_list.append(previous.detach())
        timestep_list.append(timesteps.detach())
        t_now_list.append(t_now.detach())
        t_prev_list.append(t_prev.detach())
        current = previous

    return (
        torch.cat(x_t_list, dim=0),
        torch.cat(x_prev_list, dim=0),
        torch.cat(timestep_list, dim=0),
        torch.cat(t_now_list, dim=0),
        torch.cat(t_prev_list, dim=0),
    )


@torch.no_grad()
def _reverse_kernel_statistics(
    denoiser: DenoiserMLP,
    x_t: torch.Tensor,
    timesteps: torch.Tensor,
    schedule: DDPMSchedule,
) -> tuple[torch.Tensor, torch.Tensor]:
    noise_prediction = denoiser(x_t, schedule.time_values(timesteps))
    x0_prediction = schedule.predict_x0(x_t, timesteps, noise_prediction)
    mean = schedule.posterior_mean(x0_prediction, x_t, timesteps)
    variance = schedule.extract(schedule.betas, timesteps, x_t)
    return mean, variance


def _forward_kernel_statistics(
    x_prev: torch.Tensor,
    timesteps: torch.Tensor,
    schedule: DDPMSchedule,
) -> tuple[torch.Tensor, torch.Tensor]:
    alpha = schedule.extract(schedule.alphas, timesteps, x_prev)
    beta = schedule.extract(schedule.betas, timesteps, x_prev)
    mean = torch.sqrt(alpha) * x_prev
    return mean, beta


def _isotropic_gaussian_log_prob(
    samples: torch.Tensor,
    mean: torch.Tensor,
    variance: torch.Tensor,
) -> torch.Tensor:
    variance = variance.clamp_min(1e-12)
    squared_error = (samples - mean).pow(2).sum(dim=-1)
    variance_flat = variance.view(samples.shape[0], -1)[:, 0]
    return -0.5 * (
        squared_error / variance_flat
        + samples.shape[-1] * (_LOG_2PI + torch.log(variance_flat))
    )


@torch.no_grad()
def _scalar_field_on_grid(
    evaluator: Callable[[torch.Tensor], torch.Tensor],
    limits: Tuple[float, float, float, float],
    resolution: int,
    device: torch.device,
    batch_size: int = 8192,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs, ys, points = _grid_points(limits, resolution)
    flat_values = np.zeros(points.shape[0], dtype=np.float32)
    tensor_points = torch.from_numpy(points).to(device=device, dtype=torch.float32)

    for start in range(0, tensor_points.shape[0], batch_size):
        stop = start + batch_size
        batch = tensor_points[start:stop]
        flat_values[start:stop] = evaluator(batch).cpu().numpy()

    return xs, ys, flat_values.reshape(resolution, resolution)


@torch.no_grad()
def _score_on_grid(
    model: DetailedBalanceModel,
    time_value: float,
    limits: Tuple[float, float, float, float],
    resolution: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model = model.to(device).eval()

    def _evaluator(points: torch.Tensor) -> torch.Tensor:
        times = torch.full((points.shape[0],), time_value, device=points.device)
        return model(points, times)

    return _scalar_field_on_grid(
        evaluator=_evaluator,
        limits=limits,
        resolution=resolution,
        device=device,
    )


def _plot_scalar_field(
    values: np.ndarray,
    limits: Tuple[float, float, float, float],
    title: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    image = ax.imshow(
        values.T,
        origin="lower",
        extent=limits,
        interpolation="bilinear",
        aspect="equal",
    )
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    _save_figure(fig, out_path)


def _plot_training_curves(history: List[Dict[str, float]], out_path: Path) -> None:
    steps = [row["step"] for row in history]
    loss = [row["loss"] for row in history]
    boundary_penalty = [row["boundary_penalty"] for row in history]
    residual_abs_mean = [row["residual_abs_mean"] for row in history]
    residual_std = [row["residual_std"] for row in history]
    terminal_score_l1 = [row["terminal_score_l1"] for row in history]

    fig, axes = plt.subplots(1, 5, figsize=(22, 4))

    axes[0].plot(steps, loss)
    axes[0].set_title("Detailed-balance loss")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("MSE")

    axes[1].plot(steps, residual_abs_mean)
    axes[1].set_title("|residual| mean")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("absolute mean")

    axes[2].plot(steps, boundary_penalty)
    axes[2].set_title("Boundary penalty")
    axes[2].set_xlabel("step")
    axes[2].set_ylabel("MSE")

    axes[3].plot(steps, residual_std)
    axes[3].set_title("Residual std")
    axes[3].set_xlabel("step")
    axes[3].set_ylabel("std")

    axes[4].plot(steps, terminal_score_l1)
    axes[4].set_title("Normalized exp(f(x, 0)) L1")
    axes[4].set_xlabel("step")
    axes[4].set_ylabel("L1 vs target")

    _save_figure(fig, out_path)


@torch.no_grad()
def _evaluate_model(
    model: DetailedBalanceModel,
    reference_density: np.ndarray,
    target_density: np.ndarray,
    limits: Tuple[float, float, float, float],
    resolution: int,
    device: torch.device,
) -> tuple[
    Dict[str, float],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    grid_x, grid_y, boundary_log_score = _score_on_grid(
        model=model,
        time_value=1.0,
        limits=limits,
        resolution=resolution,
        device=device,
    )
    _, _, terminal_log_score = _score_on_grid(
        model=model,
        time_value=0.0,
        limits=limits,
        resolution=resolution,
        device=device,
    )

    boundary_score = np.exp(np.clip(boundary_log_score, -60.0, 60.0))
    terminal_score = np.exp(np.clip(terminal_log_score, -60.0, 60.0))
    boundary_metrics = density_grid_metrics(reference_density, boundary_score, limits=limits)
    terminal_score_density = normalize_log_density_grid(terminal_log_score, limits=limits)
    terminal_metrics = density_grid_metrics(target_density, terminal_score_density, limits=limits)

    metrics = {
        "boundary_score_l1": boundary_metrics["l1"],
        "boundary_score_kl": boundary_metrics["kl_target_to_estimate"],
        "terminal_score_l1": terminal_metrics["l1"],
        "terminal_score_kl": terminal_metrics["kl_target_to_estimate"],
    }
    return metrics, grid_x, grid_y, boundary_log_score, boundary_score, terminal_log_score, terminal_score, terminal_score_density


def train_detailed_balance_model(
    config: DetailedBalanceTrainConfig,
    denoiser: DenoiserMLP,
    reference_distribution: Density2D,
    target_distribution: Density2D,
) -> DetailedBalanceTrainResult:
    set_seed(config.seed)
    device = torch.device(config.device)
    schedule = DDPMSchedule(num_steps=config.num_sample_steps)
    denoiser = denoiser.to(device).eval()
    for parameter in denoiser.parameters():
        parameter.requires_grad_(False)

    model = build_detailed_balance_model(
        model_type=config.model_type,
        hidden_dim=config.hidden_dim,
        depth=config.depth,
        num_time_frequencies=config.time_frequencies,
        denoiser=denoiser,
        schedule=schedule,
    ).to(device)
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.lr,
    )

    limits = combine_plot_limits(reference_distribution.default_limits(), target_distribution.default_limits())
    _, _, reference_density = density_on_grid(
        reference_distribution,
        limits=limits,
        resolution=config.density_resolution,
        device=str(device),
    )
    _, _, target_density = density_on_grid(
        target_distribution,
        limits=limits,
        resolution=config.density_resolution,
        device=str(device),
    )

    history: List[Dict[str, float]] = []

    for step in range(config.train_steps + 1):
        model.train()
        x_t, x_prev, timesteps, t_now, t_prev = _sample_transition_pairs(
            denoiser=denoiser,
            reference_distribution=reference_distribution,
            num_trajectories=config.batch_num_trajectories,
            schedule=schedule,
            device=device,
        )

        with torch.no_grad():
            reverse_mean, reverse_variance = _reverse_kernel_statistics(
                denoiser=denoiser,
                x_t=x_t,
                timesteps=timesteps,
                schedule=schedule,
            )
            forward_mean, forward_variance = _forward_kernel_statistics(
                x_prev=x_prev,
                timesteps=timesteps,
                schedule=schedule,
            )
            log_p = _isotropic_gaussian_log_prob(x_prev, reverse_mean, reverse_variance)
            log_q = _isotropic_gaussian_log_prob(x_t, forward_mean, forward_variance)

        previous_timesteps = (timesteps - 1).clamp_min(0)
        log_f_t = model(x_t, t_now, timesteps=timesteps)
        log_f_prev = model(x_prev, t_prev, timesteps=previous_timesteps)
        residual = log_f_prev + log_q - log_f_t - log_p
        detailed_balance_loss = residual.pow(2).mean()
        boundary_states = x_t[: config.batch_num_trajectories]
        boundary_times = t_now[: config.batch_num_trajectories]
        boundary_timesteps = timesteps[: config.batch_num_trajectories]
        boundary_targets = reference_distribution.log_prob(boundary_states).detach()
        boundary_penalty = (
            model(boundary_states, boundary_times, timesteps=boundary_timesteps) - boundary_targets
        ).pow(2).mean()
        loss = detailed_balance_loss + config.boundary_penalty_weight * boundary_penalty

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        should_eval = (step % config.eval_every == 0) or (step == config.train_steps)
        if should_eval:
            eval_metrics, _, _, boundary_log_score, _, terminal_log_score, _, terminal_score_density = _evaluate_model(
                model=model,
                reference_density=reference_density,
                target_density=target_density,
                limits=limits,
                resolution=config.density_resolution,
                device=device,
            )
            row = {
                "step": float(step),
                "loss": float(loss.detach().cpu()),
                "detailed_balance_loss": float(detailed_balance_loss.detach().cpu()),
                "boundary_penalty": float(boundary_penalty.detach().cpu()),
                "residual_abs_mean": float(residual.detach().abs().mean().cpu()),
                "residual_std": float(residual.detach().std().cpu()),
                **eval_metrics,
                "terminal_log_score_std": float(np.std(terminal_log_score)),
                "terminal_density_mass": float(
                    terminal_score_density.sum()
                    * (limits[1] - limits[0])
                    / max(config.density_resolution - 1, 1)
                    * (limits[3] - limits[2])
                    / max(config.density_resolution - 1, 1)
                ),
                "boundary_log_score_std": float(np.std(boundary_log_score)),
            }
            history.append(row)
            print(
                f"db_step={step:5d} "
                f"loss={row['loss']:.6f} "
                f"db={row['detailed_balance_loss']:.6f} "
                f"boundary={row['boundary_penalty']:.6f} "
                f"|res|={row['residual_abs_mean']:.6f} "
                f"res_std={row['residual_std']:.6f} "
                f"t0_l1={row['terminal_score_l1']:.6f} "
                f"t1_l1={row['boundary_score_l1']:.6f}"
            )

    diffusion_samples = (
        sample_model_samples(
            model=denoiser,
            reference_distribution=reference_distribution,
            num_samples=config.num_kde_samples,
            num_steps=config.num_sample_steps,
            device=device,
            schedule=schedule,
        )
        .cpu()
        .numpy()
    )
    grid_x, grid_y, kde_density = kde_on_grid(
        diffusion_samples,
        limits=limits,
        resolution=config.density_resolution,
    )
    (
        _,
        _,
        _,
        boundary_log_score,
        boundary_score,
        terminal_log_score,
        terminal_score,
        terminal_score_density,
    ) = _evaluate_model(
        model=model,
        reference_density=reference_density,
        target_density=target_density,
        limits=limits,
        resolution=config.density_resolution,
        device=device,
    )

    return DetailedBalanceTrainResult(
        model=model.cpu(),
        history=history,
        limits=limits,
        grid_x=grid_x,
        grid_y=grid_y,
        reference_density=reference_density,
        target_density=target_density,
        kde_density=kde_density,
        diffusion_samples=diffusion_samples,
        boundary_log_score=boundary_log_score,
        boundary_score=boundary_score,
        terminal_log_score=terminal_log_score,
        terminal_score=terminal_score,
        terminal_score_density=terminal_score_density,
    )


def save_detailed_balance_run_artifacts(
    result: DetailedBalanceTrainResult,
    config: DetailedBalanceTrainConfig,
    diffusion_config: Dict[str, object],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "reward_config": asdict(config),
        "diffusion_config": diffusion_config,
        "history": result.history,
        "final_metrics": {
            "boundary": density_grid_metrics(
                result.reference_density,
                result.boundary_score,
                limits=result.limits,
            ),
            "terminal": density_grid_metrics(
                result.target_density,
                result.terminal_score_density,
                limits=result.limits,
            ),
        },
    }

    torch.save(
        {
            "state_dict": result.model.state_dict(),
            "reward_config": asdict(config),
            "diffusion_config": diffusion_config,
        },
        out_dir / "detailed_balance_checkpoint.pt",
    )

    np.save(out_dir / "grid_x.npy", result.grid_x)
    np.save(out_dir / "grid_y.npy", result.grid_y)
    np.save(out_dir / "reference_density.npy", result.reference_density)
    np.save(out_dir / "target_density.npy", result.target_density)
    np.save(out_dir / "kde_density.npy", result.kde_density)
    np.save(out_dir / "diffusion_samples.npy", result.diffusion_samples)
    np.save(out_dir / "boundary_log_score.npy", result.boundary_log_score)
    np.save(out_dir / "boundary_score.npy", result.boundary_score)
    np.save(out_dir / "terminal_log_score.npy", result.terminal_log_score)
    np.save(out_dir / "terminal_score.npy", result.terminal_score)
    np.save(out_dir / "terminal_score_density.npy", result.terminal_score_density)

    with open(out_dir / "detailed_balance_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    plot_reward_density_comparison(
        target_density=result.target_density,
        kde_density=result.kde_density,
        reward_density=result.terminal_score_density,
        limits=result.limits,
        out_path=out_dir / "terminal_score_density_comparison.png",
        estimate_title="Normalized exp(f(x, 0))",
    )
    _plot_scalar_field(
        values=result.boundary_score,
        limits=result.limits,
        title="exp(f(x, 1))",
        out_path=out_dir / "exp_f_t1.png",
    )
    _plot_scalar_field(
        values=result.terminal_score,
        limits=result.limits,
        title="exp(f(x, 0))",
        out_path=out_dir / "exp_f_t0.png",
    )
    _plot_training_curves(result.history, out_dir / "detailed_balance_training_curves.png")


RewardLearningTrainConfig = DetailedBalanceTrainConfig
RewardLearningTrainResult = DetailedBalanceTrainResult
train_reward_learning_model = train_detailed_balance_model
save_reward_learning_run_artifacts = save_detailed_balance_run_artifacts
