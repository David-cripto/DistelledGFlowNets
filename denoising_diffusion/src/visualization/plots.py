from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import torch
from torchvision.utils import save_image

from ..data import DatasetInfo


def _to_display_range(images: torch.Tensor, dataset_info: DatasetInfo) -> torch.Tensor:
    images = dataset_info.denormalize(images.detach().cpu())
    return images.clamp(0.0, 1.0)


def save_image_grid(
    images: torch.Tensor,
    out_path: Path,
    dataset_info: DatasetInfo,
    *,
    nrow: int = 8,
    padding: int = 2,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(_to_display_range(images, dataset_info), out_path, nrow=nrow, padding=padding)


def save_trajectory_grid(
    trajectory: torch.Tensor,
    out_path: Path,
    dataset_info: DatasetInfo,
    *,
    max_samples: int = 8,
    max_timepoints: int = 8,
) -> None:
    if trajectory.ndim != 5:
        raise ValueError("Expected trajectory to have shape [num_times, batch, channels, height, width].")

    num_times, batch_size = trajectory.shape[:2]
    sample_count = min(batch_size, max_samples)
    time_count = min(num_times, max_timepoints)

    sample_indices = torch.linspace(0, batch_size - 1, steps=sample_count, dtype=torch.long)
    time_indices = torch.linspace(0, num_times - 1, steps=time_count, dtype=torch.long)
    tiles = trajectory[time_indices][:, sample_indices]
    tiles = tiles.permute(1, 0, 2, 3, 4).reshape(sample_count * time_count, *trajectory.shape[2:])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(_to_display_range(tiles, dataset_info), out_path, nrow=time_count, padding=2)


def plot_training_curves(
    history: list[dict[str, float]],
    keys: Iterable[str],
    out_path: Path,
) -> None:
    keys = list(keys)
    if not history:
        return

    steps = [row["step"] for row in history]
    fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 4))
    if len(keys) == 1:
        axes = [axes]

    for axis, key in zip(axes, keys):
        values = [row[key] for row in history]
        axis.plot(steps, values)
        axis.set_title(key.replace("_", " "))
        axis.set_xlabel("step")
        axis.set_ylabel(key)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_histograms(
    series: dict[str, torch.Tensor],
    out_path: Path,
    *,
    bins: int = 40,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, values in series.items():
        ax.hist(values.detach().cpu().numpy(), bins=bins, alpha=0.55, label=label, density=True)

    ax.legend()
    ax.set_xlabel("value")
    ax.set_ylabel("density")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
