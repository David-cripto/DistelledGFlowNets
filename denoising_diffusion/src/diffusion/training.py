from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..data import DatasetInfo, StandardNormalReference, build_dataset, extract_images, get_dataset_info
from ..visualization import plot_training_curves, save_image_grid, save_trajectory_grid
from .model import DenoiserCNN
from .schedules import DDPMSchedule


torch.set_num_threads(1)


@dataclass
class TrainConfig:
    dataset: str = "mnist"
    data_dir: str = "data"
    download: bool = True
    train_steps: int = 10000
    batch_size: int = 128
    lr: float = 2e-4
    seed: int = 0
    hidden_channels: int = 64
    depth: int = 4
    time_frequencies: int = 16
    eval_every: int = 500
    num_eval_samples: int = 64
    num_sample_steps: int = 128
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    num_workers: int = 0
    device: str = "cpu"


@dataclass
class TrainResult:
    model: DenoiserCNN
    reference_distribution: StandardNormalReference
    dataset_info: DatasetInfo
    history: List[Dict[str, float]]
    sample_times: np.ndarray
    sample_trajectory: np.ndarray
    model_samples: np.ndarray
    real_samples: np.ndarray


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _build_loader(
    dataset_name: str,
    data_dir: str | Path,
    *,
    batch_size: int,
    train: bool,
    download: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    dataset = build_dataset(
        dataset_name,
        root=data_dir,
        train=train,
        download=download,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        drop_last=train,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def _cycle(loader: DataLoader) -> Iterator[torch.Tensor]:
    while True:
        for batch in loader:
            yield extract_images(batch)


@torch.no_grad()
def predict_x0(
    model: DenoiserCNN,
    x_t: torch.Tensor,
    timesteps: torch.Tensor,
    schedule: DDPMSchedule,
) -> torch.Tensor:
    noise_prediction = model(x_t, schedule.time_values(timesteps))
    return schedule.predict_x0(x_t, timesteps, noise_prediction)


@torch.no_grad()
def reverse_diffusion_step(
    model: DenoiserCNN,
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
    model: DenoiserCNN,
    reference_distribution: StandardNormalReference,
    num_samples: int,
    num_steps: int,
    device: torch.device,
    schedule: DDPMSchedule | None = None,
) -> torch.Tensor:
    schedule = schedule or DDPMSchedule(num_steps=num_steps)
    if schedule.num_steps != num_steps:
        raise ValueError("num_steps must match the DDPM schedule used for sampling.")

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
    model: DenoiserCNN,
    reference_distribution: StandardNormalReference,
    num_samples: int,
    num_steps: int,
    *,
    device: str = "cpu",
    schedule: DDPMSchedule | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    device_obj = torch.device(device)
    schedule = schedule or DDPMSchedule(num_steps=num_steps)
    if schedule.num_steps != num_steps:
        raise ValueError("num_steps must match the DDPM schedule used for sampling.")

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


@torch.no_grad()
def _evaluate_model(
    model: DenoiserCNN,
    eval_batch: torch.Tensor,
    reference_distribution: StandardNormalReference,
    config: TrainConfig,
    device: torch.device,
    schedule: DDPMSchedule,
) -> Dict[str, float]:
    eval_batch = eval_batch.to(device)
    noise = torch.randn_like(eval_batch)
    timesteps = schedule.sample_timesteps(eval_batch.shape[0], device=device)
    x_t = schedule.q_sample(eval_batch, timesteps, noise)
    noise_prediction = model(x_t, schedule.time_values(timesteps))
    eval_loss = F.mse_loss(noise_prediction, noise)

    model_samples = sample_model_samples(
        model=model,
        reference_distribution=reference_distribution,
        num_samples=config.num_eval_samples,
        num_steps=config.num_sample_steps,
        device=device,
        schedule=schedule,
    )

    return {
        "eval_loss": float(eval_loss.cpu()),
        "sample_mean": float(model_samples.mean().cpu()),
        "sample_std": float(model_samples.std().cpu()),
        "sample_abs_mean": float(model_samples.abs().mean().cpu()),
    }


def train_image_diffusion(config: TrainConfig) -> TrainResult:
    set_seed(config.seed)
    device = torch.device(config.device)
    dataset_info = get_dataset_info(config.dataset)
    reference_distribution = StandardNormalReference(image_shape=dataset_info.image_shape)
    schedule = DDPMSchedule(
        num_steps=config.num_sample_steps,
        beta_start=config.beta_start,
        beta_end=config.beta_end,
    )

    train_loader = _build_loader(
        config.dataset,
        config.data_dir,
        batch_size=config.batch_size,
        train=True,
        download=config.download,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )
    eval_loader = _build_loader(
        config.dataset,
        config.data_dir,
        batch_size=config.num_eval_samples,
        train=False,
        download=config.download,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )
    train_batches = _cycle(train_loader)
    eval_batches = _cycle(eval_loader)

    model = DenoiserCNN(
        image_channels=dataset_info.image_shape[0],
        hidden_channels=config.hidden_channels,
        depth=config.depth,
        num_time_frequencies=config.time_frequencies,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    history: List[Dict[str, float]] = []

    for step in range(config.train_steps + 1):
        model.train()
        x0 = next(train_batches).to(device)
        noise = torch.randn_like(x0)
        timesteps = schedule.sample_timesteps(x0.shape[0], device=device)
        t = schedule.time_values(timesteps)
        x_t = schedule.q_sample(x0, timesteps, noise)
        noise_prediction = model(x_t, t)
        loss = F.mse_loss(noise_prediction, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        should_eval = (step % config.eval_every == 0) or (step == config.train_steps)
        if should_eval:
            metrics = _evaluate_model(
                model=model,
                eval_batch=next(eval_batches),
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
                f"eval_loss={row['eval_loss']:.6f} "
                f"sample_mean={row['sample_mean']:.4f} "
                f"sample_std={row['sample_std']:.4f}"
            )

    real_samples = next(eval_batches)[: config.num_eval_samples].cpu().numpy()
    times, trajectory = sample_trajectory(
        model=model,
        reference_distribution=reference_distribution,
        num_samples=min(config.num_eval_samples, 16),
        num_steps=config.num_sample_steps,
        device=config.device,
        schedule=schedule,
    )
    model_samples = (
        sample_model_samples(
            model=model,
            reference_distribution=reference_distribution,
            num_samples=config.num_eval_samples,
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
        dataset_info=dataset_info,
        history=history,
        sample_times=times,
        sample_trajectory=trajectory,
        model_samples=model_samples,
        real_samples=real_samples,
    )


def load_diffusion_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: str = "cpu",
) -> tuple[DenoiserCNN, StandardNormalReference, DatasetInfo, TrainConfig]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = TrainConfig(**checkpoint["config"])
    dataset_info = get_dataset_info(config.dataset)
    model = DenoiserCNN(
        image_channels=dataset_info.image_shape[0],
        hidden_channels=config.hidden_channels,
        depth=config.depth,
        num_time_frequencies=config.time_frequencies,
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    reference_distribution = StandardNormalReference(image_shape=dataset_info.image_shape)
    return model, reference_distribution, dataset_info, config


def save_run_artifacts(result: TrainResult, config: TrainConfig, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "config": asdict(config),
        "history": result.history,
        "dataset": {
            "name": result.dataset_info.name,
            "image_shape": list(result.dataset_info.image_shape),
        },
    }

    torch.save(
        {
            "state_dict": result.model.state_dict(),
            "config": asdict(config),
            "dataset": result.dataset_info.name,
            "image_shape": result.dataset_info.image_shape,
        },
        out_dir / "checkpoint.pt",
    )

    np.save(out_dir / "sample_times.npy", result.sample_times)
    np.save(out_dir / "sample_trajectory.npy", result.sample_trajectory)
    np.save(out_dir / "model_samples.npy", result.model_samples)
    np.save(out_dir / "real_samples.npy", result.real_samples)

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    save_image_grid(
        torch.from_numpy(result.real_samples),
        out_dir / "real_samples.png",
        dataset_info=result.dataset_info,
        nrow=8,
    )
    save_image_grid(
        torch.from_numpy(result.model_samples),
        out_dir / "generated_samples.png",
        dataset_info=result.dataset_info,
        nrow=8,
    )
    save_trajectory_grid(
        torch.from_numpy(result.sample_trajectory),
        out_dir / "sample_trajectory.png",
        dataset_info=result.dataset_info,
    )
    plot_training_curves(
        result.history,
        keys=("loss", "eval_loss", "sample_std"),
        out_path=out_dir / "diffusion_training_curves.png",
    )


train_diffusion = train_image_diffusion
save_diffusion_artifacts = save_run_artifacts
