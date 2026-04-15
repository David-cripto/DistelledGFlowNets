from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ..distributions import Density2D, build_distribution
from ..distributions.gaussians import DiagonalGaussian2D
from ..visualization import combine_plot_limits, plot_density_triptych, plot_exact_density, plot_sample_kde, plot_trajectories
from .model import DenoiserMLP
from .schedules import DDPMSchedule


torch.set_num_threads(1)


@dataclass
class TrainConfig:
    target: str = "mixture4"
    reference: str = "gaussian"
    train_steps: int = 3000
    batch_size: int = 512
    lr: float = 1e-3
    seed: int = 0
    hidden_dim: int = 128
    depth: int = 3
    time_frequencies: int = 8
    eval_every: int = 200
    num_eval_samples: int = 2048
    num_sample_steps: int = 64
    num_visualization_samples: int = 4096
    num_trajectory_samples: int = 256
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    device: str = "cpu"


@dataclass
class TrainResult:
    model: DenoiserMLP
    reference_distribution: Density2D
    target_distribution: Density2D
    history: List[Dict[str, float]]
    sample_times: np.ndarray
    sample_trajectory: np.ndarray
    model_samples: np.ndarray


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _require_standard_normal_reference(reference_distribution: Density2D) -> None:
    if not isinstance(reference_distribution, DiagonalGaussian2D):
        raise ValueError("Standard DDPM requires a standard normal reference distribution; use reference='gaussian'.")
    zero = torch.zeros_like(reference_distribution.mean)
    one = torch.ones_like(reference_distribution.std)
    if not torch.allclose(reference_distribution.mean, zero) or not torch.allclose(reference_distribution.std, one):
        raise ValueError("Standard DDPM requires the reference distribution to be N(0, I); use reference='gaussian'.")


def _sample_training_batch(
    target_distribution: Density2D,
    schedule: DDPMSchedule,
    batch_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x0 = target_distribution.sample(batch_size, device=device)
    noise = torch.randn_like(x0)
    timesteps = schedule.sample_timesteps(batch_size, device=device)
    t = schedule.time_values(timesteps)
    x_t = schedule.q_sample(x0, timesteps, noise)
    return x0, noise, timesteps, t, x_t


@torch.no_grad()
def predict_x0(
    model: DenoiserMLP,
    x_t: torch.Tensor,
    timesteps: torch.Tensor,
    schedule: DDPMSchedule,
) -> torch.Tensor:
    noise_prediction = model(x_t, schedule.time_values(timesteps))
    return schedule.predict_x0(x_t, timesteps, noise_prediction)


@torch.no_grad()
def reverse_diffusion_step(
    model: DenoiserMLP,
    x_t: torch.Tensor,
    timesteps: torch.Tensor,
    schedule: DDPMSchedule,
) -> Tuple[torch.Tensor, torch.Tensor]:
    noise_prediction = model(x_t, schedule.time_values(timesteps))
    x0_prediction = schedule.predict_x0(x_t, timesteps, noise_prediction)
    posterior_mean = schedule.posterior_mean(x0_prediction, x_t, timesteps)
    reverse_std = torch.sqrt(schedule.extract(schedule.betas, timesteps, x_t).clamp_min(schedule.epsilon))
    noise = torch.randn_like(x_t)
    x_prev = posterior_mean + reverse_std * noise
    return x_prev, x0_prediction


@torch.no_grad()
def sample_model_samples(
    model: DenoiserMLP,
    reference_distribution: Density2D,
    num_samples: int,
    num_steps: int,
    device: torch.device,
    schedule: DDPMSchedule | None = None,
) -> torch.Tensor:
    schedule = schedule or DDPMSchedule(num_steps=num_steps)
    if schedule.num_steps != num_steps:
        raise ValueError("num_steps must match the DDPM schedule used for sampling.")
    _require_standard_normal_reference(reference_distribution)

    current = reference_distribution.sample(num_samples, device=device)
    model.eval()
    for step in range(schedule.num_steps - 1, -1, -1):
        timesteps = torch.full((num_samples,), step, device=device, dtype=torch.long)
        current, _ = reverse_diffusion_step(
            model=model,
            x_t=current,
            timesteps=timesteps,
            schedule=schedule,
        )

    return current


@torch.no_grad()
def sample_trajectory(
    model: DenoiserMLP,
    reference_distribution: Density2D,
    num_samples: int,
    num_steps: int,
    device: str = "cpu",
    schedule: DDPMSchedule | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    device_obj = torch.device(device)
    schedule = schedule or DDPMSchedule(num_steps=num_steps)
    if schedule.num_steps != num_steps:
        raise ValueError("num_steps must match the DDPM schedule used for sampling.")
    _require_standard_normal_reference(reference_distribution)

    current = reference_distribution.sample(num_samples, device=device_obj)
    states = [current.detach().cpu()]
    sample_times = [1.0]

    model.eval()
    for step in range(schedule.num_steps - 1, -1, -1):
        timesteps = torch.full((num_samples,), step, device=device_obj, dtype=torch.long)
        current, _ = reverse_diffusion_step(
            model=model,
            x_t=current,
            timesteps=timesteps,
            schedule=schedule,
        )
        states.append(current.detach().cpu())
        sample_times.append(float(step) / float(schedule.num_steps))

    trajectory = torch.stack(states, dim=0).numpy()
    return np.asarray(sample_times, dtype=np.float32), trajectory


def _covariance(samples: torch.Tensor) -> torch.Tensor:
    centered = samples - samples.mean(dim=0, keepdim=True)
    denom = max(samples.shape[0] - 1, 1)
    return centered.T @ centered / float(denom)


@torch.no_grad()
def _evaluate_model(
    model: DenoiserMLP,
    target_distribution: Density2D,
    reference_distribution: Density2D,
    config: TrainConfig,
    device: torch.device,
    schedule: DDPMSchedule,
) -> Dict[str, float]:
    model_samples = sample_model_samples(
        model=model,
        reference_distribution=reference_distribution,
        num_samples=config.num_eval_samples,
        num_steps=config.num_sample_steps,
        device=device,
        schedule=schedule,
    )
    target_samples = target_distribution.sample(config.num_eval_samples, device=device)

    sample_mean = model_samples.mean(dim=0)
    target_mean = target_samples.mean(dim=0)
    sample_cov = _covariance(model_samples)
    target_cov = _covariance(target_samples)

    metrics = {
        "target_log_prob": float(target_distribution.log_prob(model_samples).mean().cpu()),
        "mean_l2": float(torch.linalg.norm(sample_mean - target_mean).cpu()),
        "cov_l1": float(torch.mean(torch.abs(sample_cov - target_cov)).cpu()),
    }
    return metrics


def train_bridge_diffusion(config: TrainConfig) -> TrainResult:
    set_seed(config.seed)
    device = torch.device(config.device)
    schedule = DDPMSchedule(
        num_steps=config.num_sample_steps,
        beta_start=config.beta_start,
        beta_end=config.beta_end,
    )
    target_distribution = build_distribution(config.target)
    reference_distribution = build_distribution(config.reference)
    _require_standard_normal_reference(reference_distribution)

    model = DenoiserMLP(
        hidden_dim=config.hidden_dim,
        depth=config.depth,
        num_time_frequencies=config.time_frequencies,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    history: List[Dict[str, float]] = []

    for step in range(config.train_steps + 1):
        model.train()
        _, noise, _, t, x_t = _sample_training_batch(
            target_distribution=target_distribution,
            schedule=schedule,
            batch_size=config.batch_size,
            device=device,
        )
        noise_prediction = model(x_t, t)
        loss = F.mse_loss(noise_prediction, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        should_eval = (step % config.eval_every == 0) or (step == config.train_steps)
        if should_eval:
            metrics = _evaluate_model(
                model=model,
                target_distribution=target_distribution,
                reference_distribution=reference_distribution,
                config=config,
                device=device,
                schedule=schedule,
            )
            row = {
                "step": float(step),
                "loss": float(loss.detach().cpu()),
                **metrics,
            }
            history.append(row)
            print(
                f"step={step:5d} "
                f"loss={row['loss']:.6f} "
                f"logp={row['target_log_prob']:.4f} "
                f"mean_l2={row['mean_l2']:.4f} "
                f"cov_l1={row['cov_l1']:.4f}"
            )

    times, trajectory = sample_trajectory(
        model=model,
        reference_distribution=reference_distribution,
        num_samples=config.num_trajectory_samples,
        num_steps=config.num_sample_steps,
        device=config.device,
        schedule=schedule,
    )
    model_samples = (
        sample_model_samples(
            model=model,
            reference_distribution=reference_distribution,
            num_samples=config.num_visualization_samples,
            num_steps=config.num_sample_steps,
            device=device,
            schedule=schedule,
        )
        .cpu()
        .numpy()
    )

    return TrainResult(
        model=model.cpu(),
        reference_distribution=reference_distribution,
        target_distribution=target_distribution,
        history=history,
        sample_times=times,
        sample_trajectory=trajectory,
        model_samples=model_samples,
    )


def save_run_artifacts(result: TrainResult, config: TrainConfig, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "config": asdict(config),
        "history": result.history,
    }

    torch.save(
        {
            "state_dict": result.model.state_dict(),
            "config": asdict(config),
            "target": config.target,
            "reference": config.reference,
        },
        out_dir / "checkpoint.pt",
    )

    np.save(out_dir / "sample_times.npy", result.sample_times)
    np.save(out_dir / "sample_trajectory.npy", result.sample_trajectory)
    np.save(out_dir / "model_samples.npy", result.model_samples)

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    limits = combine_plot_limits(
        result.reference_distribution.default_limits(),
        result.target_distribution.default_limits(),
    )

    plot_exact_density(
        distribution=result.reference_distribution,
        title=f"Reference density: {result.reference_distribution.name}",
        out_path=out_dir / "reference_density.png",
        limits=limits,
    )
    plot_exact_density(
        distribution=result.target_distribution,
        title=f"Target density: {result.target_distribution.name}",
        out_path=out_dir / "target_density.png",
        limits=limits,
    )
    plot_sample_kde(
        samples=result.model_samples,
        title="Model sample KDE",
        out_path=out_dir / "model_kde.png",
        limits=limits,
    )
    plot_density_triptych(
        reference_distribution=result.reference_distribution,
        target_distribution=result.target_distribution,
        model_samples=result.model_samples,
        out_path=out_dir / "density_triptych.png",
        limits=limits,
    )
    plot_trajectories(
        trajectory=result.sample_trajectory,
        times=result.sample_times,
        out_path=out_dir / "trajectories.png",
        limits=limits,
    )


train_diffusion = train_bridge_diffusion
save_diffusion_artifacts = save_run_artifacts
