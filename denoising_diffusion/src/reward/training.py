from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterator, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data import DatasetInfo, StandardNormalReference, build_dataset, extract_images, get_dataset_info
from ..diffusion.model import DenoiserCNN
from ..diffusion.schedules import DDPMSchedule
from ..diffusion.training import sample_model_samples
from ..visualization import plot_histograms, plot_training_curves, save_image_grid
from .model import DetailedBalanceModel, build_detailed_balance_model


torch.set_num_threads(1)


@dataclass
class DetailedBalanceTrainConfig:
    dataset: str = "mnist"
    data_dir: str = "data"
    download: bool = True
    train_steps: int = 4000
    pretrain_steps: int = 1000
    pretrain_eval_every: int = 250
    batch_num_trajectories: int = 64
    eval_batch_size: int = 256
    lr: float = 1e-4
    seed: int = 0
    model_type: str = "direct"
    hidden_dim: int = 512
    depth: int = 3
    time_frequencies: int = 16
    boundary_penalty_weight: float = 0.1
    eval_every: int = 250
    num_sample_steps: int = 128
    num_preview_samples: int = 64
    num_workers: int = 0
    device: str = "cpu"


@dataclass
class DetailedBalanceTrainResult:
    model: DetailedBalanceModel
    history: List[Dict[str, float]]
    real_samples: np.ndarray
    diffusion_samples: np.ndarray
    boundary_log_scores: np.ndarray
    boundary_target_log_scores: np.ndarray
    real_terminal_scores: np.ndarray
    generated_terminal_scores: np.ndarray


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
def _sample_transition_pairs(
    denoiser: DenoiserCNN,
    x0: torch.Tensor,
    schedule: DDPMSchedule,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = x0.shape[0]
    x0 = x0.to(device)
    timesteps = torch.randint(1, schedule.num_steps, (batch_size,), device=device, dtype=torch.long)
    noise = torch.randn_like(x0)
    x_t = schedule.q_sample(x0, timesteps, noise)

    reverse_mean, reverse_variance = _reverse_kernel_statistics(
        denoiser=denoiser,
        x_t=x_t,
        timesteps=timesteps,
        schedule=schedule,
    )
    reverse_variance = reverse_variance.clamp_min(schedule.epsilon)
    reverse_std = torch.sqrt(reverse_variance)
    x_prev = reverse_mean + reverse_std * torch.randn_like(x_t)

    t_now = schedule.time_values(timesteps)
    previous_timesteps = timesteps - 1
    t_prev = schedule.time_values(previous_timesteps)
    return x_t, x_prev, timesteps, t_now, t_prev, reverse_mean, reverse_variance


@torch.no_grad()
def _reverse_kernel_statistics(
    denoiser: DenoiserCNN,
    x_t: torch.Tensor,
    timesteps: torch.Tensor,
    schedule: DDPMSchedule,
) -> tuple[torch.Tensor, torch.Tensor]:
    noise_prediction = denoiser(x_t, schedule.time_values(timesteps), timesteps=timesteps)
    x0_prediction = schedule.predict_x0(x_t, timesteps, noise_prediction)
    # This mean matches the DDPM epsilon-parameterized reverse mean used in
    # the paper; we compute it through x0_hat only as an equivalent form.
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


def _matched_variance_transition_log_ratio(
    x_t: torch.Tensor,
    x_prev: torch.Tensor,
    reverse_mean: torch.Tensor,
    timesteps: torch.Tensor,
    schedule: DDPMSchedule,
) -> torch.Tensor:
    forward_mean, beta = _forward_kernel_statistics(
        x_prev=x_prev,
        timesteps=timesteps,
        schedule=schedule,
    )
    beta = beta.clamp_min(schedule.epsilon)
    reverse_error = (x_prev - reverse_mean).reshape(x_prev.shape[0], -1).pow(2).sum(dim=-1)
    forward_error = (x_t - forward_mean).reshape(x_t.shape[0], -1).pow(2).sum(dim=-1)
    beta_flat = beta.reshape(beta.shape[0], -1)[:, 0]
    # With sigma_t^2 = beta_t for both kernels, the Gaussian normalizers cancel,
    # so log q(x_t | x_{t-1}) - log p_theta(x_{t-1} | x_t) is just this quadratic gap.
    return 0.5 * (reverse_error - forward_error) / beta_flat


@torch.no_grad()
def _sample_reference_supervision_batch(
    reference_distribution: StandardNormalReference,
    batch_size: int,
    schedule: DDPMSchedule,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    states = reference_distribution.sample(batch_size, device=device)
    timesteps = torch.randint(0, schedule.num_steps, (batch_size,), device=device, dtype=torch.long)
    times = schedule.time_values(timesteps)
    targets = reference_distribution.log_prob(states)
    return states, timesteps, times, targets


@torch.no_grad()
def _evaluate_model(
    model: DetailedBalanceModel,
    denoiser: DenoiserCNN,
    reference_distribution: StandardNormalReference,
    eval_batch: torch.Tensor,
    config: DetailedBalanceTrainConfig,
    device: torch.device,
    schedule: DDPMSchedule,
) -> tuple[Dict[str, float], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    eval_batch = eval_batch.to(device)
    num_samples = eval_batch.shape[0]

    diffusion_samples = sample_model_samples(
        model=denoiser,
        reference_distribution=reference_distribution,
        num_samples=num_samples,
        num_steps=config.num_sample_steps,
        device=device,
        schedule=schedule,
    )

    boundary_states = reference_distribution.sample(num_samples, device=device)
    boundary_timesteps = torch.full((num_samples,), schedule.num_steps - 1, device=device, dtype=torch.long)
    boundary_times = schedule.time_values(boundary_timesteps)
    boundary_predictions = model(boundary_states, boundary_times, timesteps=boundary_timesteps)
    boundary_targets = reference_distribution.log_prob(boundary_states)

    terminal_timesteps = torch.zeros((num_samples,), device=device, dtype=torch.long)
    terminal_times = torch.zeros((num_samples,), device=device)
    real_scores = model(eval_batch, terminal_times, timesteps=terminal_timesteps)
    generated_scores = model(diffusion_samples, terminal_times, timesteps=terminal_timesteps)

    metrics = {
        "boundary_rmse": float(torch.sqrt((boundary_predictions - boundary_targets).pow(2).mean()).cpu()),
        "boundary_l1": float(torch.mean(torch.abs(boundary_predictions - boundary_targets)).cpu()),
        "terminal_real_score_mean": float(real_scores.mean().cpu()),
        "terminal_generated_score_mean": float(generated_scores.mean().cpu()),
        "terminal_score_gap": float((real_scores.mean() - generated_scores.mean()).cpu()),
    }
    return metrics, diffusion_samples, boundary_predictions, boundary_targets, real_scores, generated_scores


def train_detailed_balance_model(
    config: DetailedBalanceTrainConfig,
    denoiser: DenoiserCNN,
    reference_distribution: StandardNormalReference,
    dataset_info: DatasetInfo,
    schedule: DDPMSchedule,
) -> DetailedBalanceTrainResult:
    set_seed(config.seed)
    device = torch.device(config.device)
    denoiser = denoiser.to(device).eval()
    for parameter in denoiser.parameters():
        parameter.requires_grad_(False)

    eval_loader = _build_loader(
        config.dataset,
        config.data_dir,
        batch_size=config.eval_batch_size,
        train=False,
        download=config.download,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )
    train_loader = _build_loader(
        config.dataset,
        config.data_dir,
        batch_size=config.batch_num_trajectories,
        train=True,
        download=config.download,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )
    train_batches = _cycle(train_loader)
    eval_batches = _cycle(eval_loader)

    model = build_detailed_balance_model(
        model_type=config.model_type,
        input_shape=dataset_info.image_shape,
        hidden_dim=config.hidden_dim,
        depth=config.depth,
        num_time_frequencies=config.time_frequencies,
        num_train_timesteps=schedule.num_steps,
    ).to(device)

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.lr,
    )

    if config.pretrain_steps > 0:
        pretrain_eval_every = max(config.pretrain_eval_every, 1)
        for step in range(1, config.pretrain_steps + 1):
            model.train()
            states, timesteps, times, targets = _sample_reference_supervision_batch(
                reference_distribution=reference_distribution,
                batch_size=config.batch_num_trajectories,
                schedule=schedule,
                device=device,
            )
            predictions = model(states, times, timesteps=timesteps)
            pretrain_loss = (predictions - targets).pow(2).mean()

            optimizer.zero_grad()
            pretrain_loss.backward()
            optimizer.step()

            should_log = (step % pretrain_eval_every == 0) or (step == config.pretrain_steps)
            if should_log:
                abs_error = torch.mean(torch.abs(predictions.detach() - targets)).cpu()
                print(
                    f"pretrain_step={step:5d} "
                    f"loss={float(pretrain_loss.detach().cpu()):.6f} "
                    f"abs_err={float(abs_error):.6f}"
                )

    history: List[Dict[str, float]] = []

    for step in range(config.train_steps + 1):
        model.train()
        x0 = next(train_batches).to(device)
        x_t, x_prev, timesteps, t_now, t_prev, reverse_mean, reverse_variance = _sample_transition_pairs(
            denoiser=denoiser,
            x0=x0,
            schedule=schedule,
            device=device,
        )

        with torch.no_grad():
            transition_log_ratio = _matched_variance_transition_log_ratio(
                x_t=x_t,
                x_prev=x_prev,
                reverse_mean=reverse_mean,
                timesteps=timesteps,
                schedule=schedule,
            )

        previous_timesteps = (timesteps - 1).clamp_min(0)
        log_f_t = model(x_t, t_now, timesteps=timesteps)
        log_f_prev = model(x_prev, t_prev, timesteps=previous_timesteps)
        residual = log_f_prev - log_f_t + transition_log_ratio
        detailed_balance_loss = residual.pow(2).mean()

        boundary_states = reference_distribution.sample(config.batch_num_trajectories, device=device)
        boundary_timesteps = torch.full(
            (config.batch_num_trajectories,),
            schedule.num_steps - 1,
            device=device,
            dtype=torch.long,
        )
        boundary_times = schedule.time_values(boundary_timesteps)
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
            (
                eval_metrics,
                diffusion_eval_samples,
                boundary_eval_predictions,
                boundary_eval_targets,
                real_scores,
                generated_scores,
            ) = _evaluate_model(
                model=model,
                denoiser=denoiser,
                reference_distribution=reference_distribution,
                eval_batch=next(eval_batches),
                config=config,
                device=device,
                schedule=schedule,
            )
            row = {
                "step": float(step),
                "loss": float(loss.detach().cpu()),
                "detailed_balance_loss": float(detailed_balance_loss.detach().cpu()),
                "boundary_penalty": float(boundary_penalty.detach().cpu()),
                "residual_abs_mean": float(residual.detach().abs().mean().cpu()),
                "residual_std": float(residual.detach().std().cpu()),
                **eval_metrics,
            }
            history.append(row)
            print(
                f"db_step={step:5d} "
                f"loss={row['loss']:.6f} "
                f"db={row['detailed_balance_loss']:.6f} "
                f"boundary={row['boundary_penalty']:.6f} "
                f"|res|={row['residual_abs_mean']:.6f} "
                f"boundary_rmse={row['boundary_rmse']:.6f} "
                f"score_gap={row['terminal_score_gap']:.6f}"
            )

    preview_batch = next(eval_batches).to(device)
    diffusion_samples = sample_model_samples(
        model=denoiser,
        reference_distribution=reference_distribution,
        num_samples=config.num_preview_samples,
        num_steps=config.num_sample_steps,
        device=device,
        schedule=schedule,
    )
    preview_batch = preview_batch[: config.num_preview_samples]
    preview_times = torch.zeros((preview_batch.shape[0],), device=device)
    preview_timesteps = torch.zeros((preview_batch.shape[0],), device=device, dtype=torch.long)
    generated_times = torch.zeros((diffusion_samples.shape[0],), device=device)
    generated_timesteps = torch.zeros((diffusion_samples.shape[0],), device=device, dtype=torch.long)

    boundary_states = reference_distribution.sample(config.eval_batch_size, device=device)
    boundary_timesteps = torch.full((config.eval_batch_size,), schedule.num_steps - 1, device=device, dtype=torch.long)
    boundary_times = schedule.time_values(boundary_timesteps)

    model.eval()
    with torch.no_grad():
        real_terminal_scores = model(preview_batch, preview_times, timesteps=preview_timesteps)
        generated_terminal_scores = model(diffusion_samples, generated_times, timesteps=generated_timesteps)
        boundary_log_scores = model(boundary_states, boundary_times, timesteps=boundary_timesteps)
        boundary_target_log_scores = reference_distribution.log_prob(boundary_states)

    return DetailedBalanceTrainResult(
        model=model.cpu(),
        history=history,
        real_samples=preview_batch.cpu().numpy(),
        diffusion_samples=diffusion_samples.cpu().numpy(),
        boundary_log_scores=boundary_log_scores.cpu().numpy(),
        boundary_target_log_scores=boundary_target_log_scores.cpu().numpy(),
        real_terminal_scores=real_terminal_scores.cpu().numpy(),
        generated_terminal_scores=generated_terminal_scores.cpu().numpy(),
    )


def save_detailed_balance_run_artifacts(
    result: DetailedBalanceTrainResult,
    config: DetailedBalanceTrainConfig,
    diffusion_config: Dict[str, object],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_info = get_dataset_info(config.dataset)

    payload = {
        "reward_config": asdict(config),
        "diffusion_config": diffusion_config,
        "history": result.history,
    }

    torch.save(
        {
            "state_dict": result.model.state_dict(),
            "reward_config": asdict(config),
            "diffusion_config": diffusion_config,
        },
        out_dir / "detailed_balance_checkpoint.pt",
    )

    np.save(out_dir / "real_samples.npy", result.real_samples)
    np.save(out_dir / "diffusion_samples.npy", result.diffusion_samples)
    np.save(out_dir / "boundary_log_scores.npy", result.boundary_log_scores)
    np.save(out_dir / "boundary_target_log_scores.npy", result.boundary_target_log_scores)
    np.save(out_dir / "real_terminal_scores.npy", result.real_terminal_scores)
    np.save(out_dir / "generated_terminal_scores.npy", result.generated_terminal_scores)

    with open(out_dir / "detailed_balance_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    save_image_grid(
        torch.from_numpy(result.real_samples),
        out_dir / "real_samples.png",
        dataset_info=dataset_info,
        nrow=8,
    )
    save_image_grid(
        torch.from_numpy(result.diffusion_samples),
        out_dir / "diffusion_samples.png",
        dataset_info=dataset_info,
        nrow=8,
    )
    plot_histograms(
        {
            "real terminal scores": torch.from_numpy(result.real_terminal_scores),
            "generated terminal scores": torch.from_numpy(result.generated_terminal_scores),
        },
        out_path=out_dir / "terminal_score_histogram.png",
    )
    plot_histograms(
        {
            "predicted boundary log-score": torch.from_numpy(result.boundary_log_scores),
            "gaussian target log-score": torch.from_numpy(result.boundary_target_log_scores),
        },
        out_path=out_dir / "boundary_alignment_histogram.png",
    )
    plot_training_curves(
        result.history,
        keys=(
            "loss",
            "detailed_balance_loss",
            "boundary_penalty",
            "residual_abs_mean",
            "boundary_rmse",
            "terminal_score_gap",
        ),
        out_path=out_dir / "detailed_balance_training_curves.png",
    )


RewardLearningTrainConfig = DetailedBalanceTrainConfig
RewardLearningTrainResult = DetailedBalanceTrainResult
train_reward_learning_model = train_detailed_balance_model
save_reward_learning_run_artifacts = save_detailed_balance_run_artifacts
