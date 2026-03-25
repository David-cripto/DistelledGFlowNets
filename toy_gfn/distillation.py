from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn

from .model import SampleBatch, TabularFlowGFlowNet
from .plotting import (
    plot_distribution_comparison,
    plot_inverse_distillation_curves,
    plot_sample_histogram,
)
from .rewards import build_reward_grid
from .training import (
    distribution_metrics,
    exact_terminal_distribution,
    flow_matching_loss,
    set_seed,
)


def _set_requires_grad(module: nn.Module, flag: bool):
    for parameter in module.parameters():
        parameter.requires_grad_(flag)


class GumbelTerminalGenerator(nn.Module):
    def __init__(self, grid_size: int, init_scale: float = 1e-2):
        super().__init__()
        if grid_size < 1:
            raise ValueError("grid_size must be at least 1")
        if init_scale < 0:
            raise ValueError("init_scale must be non-negative")

        self.grid_size = int(grid_size)
        n = self.grid_size + 1
        init = init_scale * torch.randn(n, n, dtype=torch.float32)
        self.logits = nn.Parameter(init)

    @property
    def shape(self):
        n = self.grid_size + 1
        return (n, n)

    @property
    def num_states(self):
        n = self.grid_size + 1
        return n * n

    def terminal_distribution(self):
        flat_probs = torch.softmax(self.logits.reshape(-1), dim=0)
        return flat_probs.reshape(self.shape)

    @staticmethod
    def _sample_gumbels(shape: Tuple[int, ...], device: torch.device):
        uniform = torch.rand(shape, device=device).clamp_(1e-6, 1.0 - 1e-6)
        return -torch.log(-torch.log(uniform))

    @torch.no_grad()
    def sample_terminal_points(self, num_samples: int):
        if num_samples < 1:
            raise ValueError("num_samples must be positive")

        device = self.logits.device
        n = self.grid_size + 1
        flat_logits = self.logits.reshape(-1)
        gumbels = self._sample_gumbels((int(num_samples), self.num_states), device=device)
        flat_indices = (flat_logits[None, :] + gumbels).argmax(dim=1)

        x = torch.div(flat_indices, n, rounding_mode="floor")
        y = flat_indices % n
        return SampleBatch(terminal_x=x.cpu(), terminal_y=y.cpu())


@dataclass
class LoadedPretrainedGFlowNet:
    model: TabularFlowGFlowNet
    normalized_model: TabularFlowGFlowNet
    target_distribution: torch.Tensor
    pretrained_distribution: torch.Tensor
    metadata: Dict[str, object]


@dataclass
class DistillConfig:
    checkpoint_path: Path
    out_dir: Path = Path("outputs") / "distilled"
    steps: int = 300
    generator_lr: float = 0.5
    f_lr: float = 0.1
    f_updates_per_step: int = 25
    f_warmup_steps: int = 250
    eval_every: int = 10
    num_terminal_samples: int = 5000
    seed: int = 0
    init_generator_scale: float = 1e-2
    init_aux_from_pretrained: bool = True
    device: str = "cpu"
    target_distribution_path: Optional[Path] = None
    reward_grid_path: Optional[Path] = None
    clamp_negative_objective: bool = True


@dataclass
class DistillResult:
    pretrained_model: TabularFlowGFlowNet
    auxiliary_model: TabularFlowGFlowNet
    generator: GumbelTerminalGenerator
    target_distribution: torch.Tensor
    pretrained_distribution: torch.Tensor
    generator_distribution: torch.Tensor
    auxiliary_distribution: torch.Tensor
    terminal_samples: np.ndarray
    history: List[Dict[str, float]]
    metadata: Dict[str, object]


@torch.no_grad()
def _load_probability_grid(path: Path):
    array = np.load(path)
    tensor = torch.tensor(array, dtype=torch.float32)
    if tensor.ndim != 2:
        raise ValueError(f"Expected a 2D grid at '{path}', got shape {tuple(tensor.shape)}")
    if float(tensor.sum()) <= 0:
        raise ValueError(f"Grid at '{path}' must have positive total mass")
    return tensor


@torch.no_grad()
def _clone_model(model: TabularFlowGFlowNet):
    clone = TabularFlowGFlowNet(grid_size=model.grid_size)
    clone.load_state_dict(model.state_dict())
    return clone


@torch.no_grad()
def _resolve_target_distribution_and_log_source(
    checkpoint_path: Path,
    payload: Dict[str, object],
    model: TabularFlowGFlowNet,
    target_distribution_path: Optional[Path] = None,
    reward_grid_path: Optional[Path] = None,
):
    if reward_grid_path is not None:
        reward_grid = _load_probability_grid(reward_grid_path)
        total_mass = float(reward_grid.sum().item())
        return reward_grid / total_mass, float(np.log(total_mass)), str(reward_grid_path)

    sibling_reward_grid = checkpoint_path.parent / "reward_grid.npy"
    if sibling_reward_grid.exists():
        reward_grid = _load_probability_grid(sibling_reward_grid)
        total_mass = float(reward_grid.sum().item())
        return reward_grid / total_mass, float(np.log(total_mass)), str(sibling_reward_grid)

    if target_distribution_path is not None:
        target_distribution = _load_probability_grid(target_distribution_path)
        target_distribution = target_distribution / target_distribution.sum()
        source_log = float(torch.logsumexp(model.masked_log_flows()[0, 0], dim=-1).item())
        return target_distribution, source_log, str(target_distribution_path)

    sibling_target_distribution = checkpoint_path.parent / "target_distribution.npy"
    if sibling_target_distribution.exists():
        target_distribution = _load_probability_grid(sibling_target_distribution)
        target_distribution = target_distribution / target_distribution.sum()
        source_log = float(torch.logsumexp(model.masked_log_flows()[0, 0], dim=-1).item())
        return target_distribution, source_log, str(sibling_target_distribution)

    reward_name = payload.get("reward")
    config = payload.get("config", {})
    if isinstance(reward_name, str):
        reward_floor = float(config.get("reward_floor", 1e-4))
        reward_grid = build_reward_grid(
            name=reward_name,
            grid_size=model.grid_size,
            reward_floor=reward_floor,
        )
        total_mass = float(reward_grid.sum().item())
        return reward_grid / total_mass, float(np.log(total_mass)), f"reward_builder:{reward_name}"

    source_log = float(torch.logsumexp(model.masked_log_flows()[0, 0], dim=-1).item())
    pretrained_distribution = exact_terminal_distribution(model)
    return pretrained_distribution, source_log, "pretrained_forward_policy"


@torch.no_grad()
def load_pretrained_gflownet(
    checkpoint_path: Path,
    target_distribution_path: Optional[Path] = None,
    reward_grid_path: Optional[Path] = None,
):
    payload = torch.load(checkpoint_path, map_location="cpu")
    grid_size = int(payload["grid_size"])

    model = TabularFlowGFlowNet(grid_size=grid_size)
    model.load_state_dict(payload["state_dict"])
    model.eval()

    target_distribution, log_source_mass, target_source = _resolve_target_distribution_and_log_source(
        checkpoint_path=checkpoint_path,
        payload=payload,
        model=model,
        target_distribution_path=target_distribution_path,
        reward_grid_path=reward_grid_path,
    )
    target_distribution = target_distribution.to(dtype=torch.float32)
    target_distribution = target_distribution / target_distribution.sum()

    normalized_model = _clone_model(model)
    normalized_model.log_edge_flows.sub_(log_source_mass)
    normalized_model.eval()

    pretrained_distribution = exact_terminal_distribution(model)
    metadata = {
        "checkpoint_path": str(checkpoint_path),
        "grid_size": grid_size,
        "reward": payload.get("reward"),
        "config": payload.get("config", {}),
        "target_source": target_source,
        "normalization_log_source_mass": float(log_source_mass),
    }
    return LoadedPretrainedGFlowNet(
        model=model,
        normalized_model=normalized_model,
        target_distribution=target_distribution.cpu(),
        pretrained_distribution=pretrained_distribution.cpu(),
        metadata=metadata,
    )


def _evaluate_distillation(
    step: int,
    generator: GumbelTerminalGenerator,
    auxiliary_model: TabularFlowGFlowNet,
    pretrained_model: TabularFlowGFlowNet,
    target_distribution: torch.Tensor,
    clamp_negative_objective: bool,
):
    generator_distribution = generator.terminal_distribution().detach().cpu()
    auxiliary_distribution = exact_terminal_distribution(auxiliary_model)

    star_loss, _ = flow_matching_loss(pretrained_model, generator_distribution)
    aux_loss, _ = flow_matching_loss(auxiliary_model, generator_distribution)
    target_metrics = distribution_metrics(target_distribution, generator_distribution)
    aux_fit_metrics = distribution_metrics(generator_distribution, auxiliary_distribution)

    raw_gap = star_loss - aux_loss
    update_gap = torch.clamp_min(raw_gap, 0.0) if clamp_negative_objective else raw_gap

    return {
        "step": float(step),
        "generator_objective": float(update_gap.detach().cpu()),
        "generator_objective_raw": float(raw_gap.detach().cpu()),
        "star_loss": float(star_loss.detach().cpu()),
        "aux_loss": float(aux_loss.detach().cpu()),
        "kl_target_to_generator": float(target_metrics["kl_target_to_model"]),
        "l1_target_to_generator": float(target_metrics["l1"]),
        "max_abs_target_to_generator": float(target_metrics["max_abs"]),
        "aux_fit_kl": float(aux_fit_metrics["kl_target_to_model"]),
        "aux_fit_l1": float(aux_fit_metrics["l1"]),
    }


def train_inverse_distillation(config: DistillConfig):
    if config.device != "cpu":
        raise ValueError("This toy project is intentionally CPU-only")

    set_seed(config.seed)
    checkpoint_path = Path(config.checkpoint_path)
    loaded = load_pretrained_gflownet(
        checkpoint_path=checkpoint_path,
        target_distribution_path=config.target_distribution_path,
        reward_grid_path=config.reward_grid_path,
    )

    target_distribution = loaded.target_distribution.to(config.device)
    pretrained_model = loaded.normalized_model.to(config.device)
    _set_requires_grad(pretrained_model, False)

    generator = GumbelTerminalGenerator(
        grid_size=loaded.metadata["grid_size"],
        init_scale=config.init_generator_scale,
    ).to(config.device)

    if config.init_aux_from_pretrained:
        auxiliary_model = _clone_model(pretrained_model).to(config.device)
    else:
        auxiliary_model = TabularFlowGFlowNet(grid_size=loaded.metadata["grid_size"]).to(config.device)

    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=config.generator_lr)
    auxiliary_optimizer = torch.optim.Adam(auxiliary_model.parameters(), lr=config.f_lr)

    history: List[Dict[str, float]] = []

    def train_auxiliary(inner_steps: int):
        if inner_steps <= 0:
            return
        _set_requires_grad(auxiliary_model, True)
        for _ in range(inner_steps):
            current_distribution = generator.terminal_distribution().detach()
            auxiliary_optimizer.zero_grad()
            loss, _ = flow_matching_loss(auxiliary_model, current_distribution)
            loss.backward()
            auxiliary_optimizer.step()

    train_auxiliary(config.f_warmup_steps)

    for step in range(config.steps + 1):
        train_auxiliary(config.f_updates_per_step)

        _set_requires_grad(auxiliary_model, False)
        generator_optimizer.zero_grad()
        current_distribution = generator.terminal_distribution()
        star_loss, _ = flow_matching_loss(pretrained_model, current_distribution)
        aux_loss, _ = flow_matching_loss(auxiliary_model, current_distribution)
        raw_generator_loss = star_loss - aux_loss
        generator_loss = (
            torch.clamp_min(raw_generator_loss, 0.0)
            if config.clamp_negative_objective
            else raw_generator_loss
        )
        generator_loss.backward()
        generator_optimizer.step()
        _set_requires_grad(auxiliary_model, True)

        should_eval = (step % config.eval_every == 0) or (step == config.steps)
        if should_eval:
            row = _evaluate_distillation(
                step=step,
                generator=generator,
                auxiliary_model=auxiliary_model,
                pretrained_model=pretrained_model,
                target_distribution=target_distribution.cpu(),
                clamp_negative_objective=config.clamp_negative_objective,
            )
            history.append(row)
            print(
                f"outer_step={step:4d} "
                f"obj={row['generator_objective']:.6f} "
                f"star={row['star_loss']:.6f} "
                f"aux={row['aux_loss']:.6f} "
                f"kl={row['kl_target_to_generator']:.6f} "
                f"aux_l1={row['aux_fit_l1']:.6f}"
            )

    generator_distribution = generator.terminal_distribution().detach().cpu()
    auxiliary_distribution = exact_terminal_distribution(auxiliary_model)
    sample_batch = generator.sample_terminal_points(config.num_terminal_samples)
    terminal_samples = torch.stack([sample_batch.terminal_x, sample_batch.terminal_y], dim=1).numpy()

    return DistillResult(
        pretrained_model=pretrained_model.cpu(),
        auxiliary_model=auxiliary_model.cpu(),
        generator=generator.cpu(),
        target_distribution=loaded.target_distribution.cpu(),
        pretrained_distribution=loaded.pretrained_distribution.cpu(),
        generator_distribution=generator_distribution.cpu(),
        auxiliary_distribution=auxiliary_distribution.cpu(),
        terminal_samples=terminal_samples,
        history=history,
        metadata=loaded.metadata,
    )


def save_distillation_artifacts(result: DistillResult, config: DistillConfig, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    config_dict = asdict(config)
    config_dict["checkpoint_path"] = str(config_dict["checkpoint_path"])
    config_dict["out_dir"] = str(config_dict["out_dir"])
    if config_dict["target_distribution_path"] is not None:
        config_dict["target_distribution_path"] = str(config_dict["target_distribution_path"])
    if config_dict["reward_grid_path"] is not None:
        config_dict["reward_grid_path"] = str(config_dict["reward_grid_path"])

    final_target_metrics = distribution_metrics(result.target_distribution, result.generator_distribution)
    final_aux_fit_metrics = distribution_metrics(result.generator_distribution, result.auxiliary_distribution)
    final_pretrained_metrics = distribution_metrics(result.pretrained_distribution, result.generator_distribution)

    payload = {
        "config": config_dict,
        "metadata": result.metadata,
        "final_metrics": {
            "target": final_target_metrics,
            "auxiliary_fit": final_aux_fit_metrics,
            "pretrained": final_pretrained_metrics,
        },
        "history": result.history,
    }

    torch.save(
        {
            "grid_size": result.generator.grid_size,
            "state_dict": result.generator.state_dict(),
            "latent_distribution": "iid_gumbel",
            "config": config_dict,
            "metadata": result.metadata,
        },
        out_dir / "generator_checkpoint.pt",
    )
    torch.save(
        {
            "grid_size": result.auxiliary_model.grid_size,
            "state_dict": result.auxiliary_model.state_dict(),
            "config": config_dict,
            "metadata": result.metadata,
        },
        out_dir / "auxiliary_checkpoint.pt",
    )
    torch.save(
        {
            "grid_size": result.pretrained_model.grid_size,
            "state_dict": result.pretrained_model.state_dict(),
            "config": config_dict,
            "metadata": result.metadata,
        },
        out_dir / "pretrained_normalized_checkpoint.pt",
    )

    np.save(out_dir / "target_distribution.npy", result.target_distribution.numpy())
    np.save(out_dir / "pretrained_distribution.npy", result.pretrained_distribution.numpy())
    np.save(out_dir / "generator_distribution.npy", result.generator_distribution.numpy())
    np.save(out_dir / "auxiliary_distribution.npy", result.auxiliary_distribution.numpy())
    np.save(out_dir / "generator_samples.npy", result.terminal_samples)

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    title = (
        "Inverse distillation: target vs generator "
        f"on a {result.generator.grid_size + 1}x{result.generator.grid_size + 1} grid"
    )
    plot_distribution_comparison(
        target=result.target_distribution.numpy(),
        learned=result.generator_distribution.numpy(),
        title=title,
        out_path=out_dir / "generator_vs_target.png",
    )
    plot_distribution_comparison(
        target=result.generator_distribution.numpy(),
        learned=result.auxiliary_distribution.numpy(),
        title="Generator distribution vs auxiliary GFlowNet distribution",
        out_path=out_dir / "generator_vs_auxiliary.png",
    )
    plot_distribution_comparison(
        target=result.pretrained_distribution.numpy(),
        learned=result.generator_distribution.numpy(),
        title="Pretrained GFlowNet distribution vs generator distribution",
        out_path=out_dir / "generator_vs_pretrained.png",
    )
    plot_inverse_distillation_curves(result.history, out_dir / "inverse_distillation_curves.png")
    plot_sample_histogram(
        samples=result.terminal_samples,
        grid_size=result.generator.grid_size,
        out_path=out_dir / "generator_sample_histogram.png",
    )
