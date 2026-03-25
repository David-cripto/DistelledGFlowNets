from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from .model import TabularFlowGFlowNet
from .plotting import plot_distribution_comparison, plot_sample_histogram, plot_training_curves
from .rewards import build_reward_grid


torch.set_num_threads(1)


@dataclass
class TrainConfig:
    reward: str = "mixture"
    grid_size: int = 20
    steps: int = 1000
    lr: float = 0.1
    seed: int = 0
    reward_floor: float = 1e-4
    eval_every: int = 25
    num_terminal_samples: int = 5000
    device: str = "cpu"


@dataclass
class TrainResult:
    model: TabularFlowGFlowNet
    reward_grid: torch.Tensor
    target_distribution: torch.Tensor
    learned_distribution: torch.Tensor
    terminal_samples: np.ndarray
    history: List[Dict[str, float]]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def flow_matching_loss(
    model: TabularFlowGFlowNet,
    reward_grid: torch.Tensor,
):
    log_flows = model.masked_log_flows()
    log_reward = reward_grid.log()
    n = model.grid_size + 1

    stop_loss = (log_flows[:, :, 2] - log_reward).pow(2).mean()

    root_outgoing = torch.logsumexp(log_flows[0, 0], dim=-1)
    source_target = torch.log(reward_grid.sum())
    source_loss = (root_outgoing - source_target).pow(2)

    neg_inf = torch.full((n, n), -1e9, dtype=log_flows.dtype, device=log_flows.device)
    left_incoming = neg_inf.clone()
    down_incoming = neg_inf.clone()
    left_incoming[1:, :] = log_flows[:-1, :, 0]
    down_incoming[:, 1:] = log_flows[:, :-1, 1]

    log_incoming = torch.logsumexp(torch.stack([left_incoming, down_incoming], dim=0), dim=0)
    log_outgoing = torch.logsumexp(log_flows, dim=-1)

    non_root_mask = torch.ones((n, n), dtype=torch.bool, device=log_flows.device)
    non_root_mask[0, 0] = False
    conservation_loss = (log_incoming[non_root_mask] - log_outgoing[non_root_mask]).pow(2).mean()

    total_loss = stop_loss + source_loss + conservation_loss
    pieces = {
        "loss": float(total_loss.detach().cpu()),
        "stop_loss": float(stop_loss.detach().cpu()),
        "source_loss": float(source_loss.detach().cpu()),
        "conservation_loss": float(conservation_loss.detach().cpu()),
    }
    return total_loss, pieces


@torch.no_grad()
def exact_terminal_distribution(model: TabularFlowGFlowNet):
    """Compute the exact terminal distribution induced by the forward policy."""
    policy = model.forward_policy().detach().cpu()
    n = model.grid_size + 1

    state_mass = torch.zeros((n, n), dtype=torch.float32)
    terminal_mass = torch.zeros((n, n), dtype=torch.float32)
    state_mass[0, 0] = 1.0

    for depth in range(0, 2 * model.grid_size + 1):
        for x in range(n):
            y = depth - x
            if y < 0 or y >= n:
                continue

            mass = state_mass[x, y]
            terminal_mass[x, y] += mass * policy[x, y, 2]

            if x < model.grid_size:
                state_mass[x + 1, y] += mass * policy[x, y, 0]
            if y < model.grid_size:
                state_mass[x, y + 1] += mass * policy[x, y, 1]

    return terminal_mass / terminal_mass.sum()


@torch.no_grad()
def distribution_metrics(target: torch.Tensor, learned: torch.Tensor) -> Dict[str, float]:
    eps = 1e-12
    target = target.clamp_min(eps)
    learned = learned.clamp_min(eps)
    kl_target_to_model = torch.sum(target * (target.log() - learned.log())).item()
    l1 = torch.sum(torch.abs(target - learned)).item()
    max_abs = torch.max(torch.abs(target - learned)).item()
    return {
        "kl_target_to_model": float(kl_target_to_model),
        "l1": float(l1),
        "max_abs": float(max_abs),
    }


def train_gflownet(config: TrainConfig) -> TrainResult:
    set_seed(config.seed)

    if config.device != "cpu":
        raise ValueError("This toy project is intentionally CPU-only")

    reward_grid = build_reward_grid(
        name=config.reward,
        grid_size=config.grid_size,
        reward_floor=config.reward_floor,
    ).to(config.device)

    target_distribution = (reward_grid / reward_grid.sum()).detach().cpu()
    model = TabularFlowGFlowNet(grid_size=config.grid_size).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    history: List[Dict[str, float]] = []

    for step in range(config.steps + 1):
        optimizer.zero_grad()
        loss, pieces = flow_matching_loss(model, reward_grid)
        loss.backward()
        optimizer.step()

        should_eval = (step % config.eval_every == 0) or (step == config.steps)
        if should_eval:
            learned_distribution = exact_terminal_distribution(model)
            metrics = distribution_metrics(target_distribution, learned_distribution)
            row = {"step": float(step), **pieces, **metrics}
            history.append(row)
            print(
                f"step={step:4d} "
                f"loss={row['loss']:.6f} "
                f"kl={row['kl_target_to_model']:.6f} "
                f"l1={row['l1']:.6f}"
            )

    final_learned = exact_terminal_distribution(model)
    sample_batch = model.sample_terminal_points(config.num_terminal_samples)
    terminal_samples = torch.stack([sample_batch.terminal_x, sample_batch.terminal_y], dim=1).numpy()

    return TrainResult(
        model=model.cpu(),
        reward_grid=reward_grid.cpu(),
        target_distribution=target_distribution.cpu(),
        learned_distribution=final_learned.cpu(),
        terminal_samples=terminal_samples,
        history=history,
    )


def save_run_artifacts(result: TrainResult, config: TrainConfig, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    config_dict = asdict(config)
    metrics = distribution_metrics(result.target_distribution, result.learned_distribution)
    payload = {
        "config": config_dict,
        "final_metrics": metrics,
        "history": result.history,
    }

    torch.save(
        {
            "state_dict": result.model.state_dict(),
            "grid_size": config.grid_size,
            "reward": config.reward,
            "config": config_dict,
        },
        out_dir / "checkpoint.pt",
    )

    np.save(out_dir / "reward_grid.npy", result.reward_grid.numpy())
    np.save(out_dir / "target_distribution.npy", result.target_distribution.numpy())
    np.save(out_dir / "learned_distribution.npy", result.learned_distribution.numpy())
    np.save(out_dir / "terminal_samples.npy", result.terminal_samples)

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    plot_distribution_comparison(
        target=result.target_distribution.numpy(),
        learned=result.learned_distribution.numpy(),
        title=f"Reward='{config.reward}' on a {config.grid_size + 1}x{config.grid_size + 1} grid",
        out_path=out_dir / "distribution_comparison.png",
    )
    plot_training_curves(result.history, out_dir / "training_curves.png")
    plot_sample_histogram(
        samples=result.terminal_samples,
        grid_size=config.grid_size,
        out_path=out_dir / "sample_histogram.png",
    )
