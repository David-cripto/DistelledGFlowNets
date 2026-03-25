from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np


def _save_figure(fig: plt.Figure, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_distribution_comparison(
    target: np.ndarray,
    learned: np.ndarray,
    title: str,
    out_path: Path,
):
    error = np.abs(target - learned)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    panels = [
        (target, "Target distribution"),
        (learned, "Learned distribution"),
        (error, "Absolute error"),
    ]

    for ax, (image, label) in zip(axes, panels):
        im = ax.imshow(image.T, origin="lower", interpolation="nearest")
        ax.set_title(label)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(title)
    _save_figure(fig, out_path)


def plot_training_curves(history: List[Dict[str, float]], out_path: Path):
    steps = [row["step"] for row in history]
    loss = [row["loss"] for row in history]
    kl = [row["kl_target_to_model"] for row in history]
    l1 = [row["l1"] for row in history]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].plot(steps, loss)
    axes[0].set_title("Training loss")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("loss")

    axes[1].plot(steps, kl)
    axes[1].set_title("KL(target || model)")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("KL")

    axes[2].plot(steps, l1)
    axes[2].set_title("L1 distance")
    axes[2].set_xlabel("step")
    axes[2].set_ylabel("L1")

    _save_figure(fig, out_path)


def plot_inverse_distillation_curves(history: List[Dict[str, float]], out_path: Path):
    steps = [row["step"] for row in history]
    gap = [row["generator_objective"] for row in history]
    raw_gap = [row.get("generator_objective_raw", row["generator_objective"]) for row in history]
    star_loss = [row["star_loss"] for row in history]
    aux_loss = [row["aux_loss"] for row in history]
    kl_target = [row["kl_target_to_generator"] for row in history]
    aux_fit_l1 = [row["aux_fit_l1"] for row in history]

    fig, axes = plt.subplots(1, 4, figsize=(17, 4))

    axes[0].plot(steps, gap, label="used for update")
    if any(abs(a - b) > 1e-12 for a, b in zip(gap, raw_gap)):
        axes[0].plot(steps, raw_gap, label="raw gap")
        axes[0].legend()
    axes[0].set_title("Generator objective")
    axes[0].set_xlabel("outer step")
    axes[0].set_ylabel("L(G)")

    axes[1].plot(steps, star_loss, label="L(f*, p_G)")
    axes[1].plot(steps, aux_loss, label="L(f, p_G)")
    axes[1].set_title("Fixed vs inner loss")
    axes[1].set_xlabel("outer step")
    axes[1].set_ylabel("loss")
    axes[1].legend()

    axes[2].plot(steps, kl_target)
    axes[2].set_title("KL(target || generator)")
    axes[2].set_xlabel("outer step")
    axes[2].set_ylabel("KL")

    axes[3].plot(steps, aux_fit_l1)
    axes[3].set_title("L1(generator, auxiliary)")
    axes[3].set_xlabel("outer step")
    axes[3].set_ylabel("L1")

    _save_figure(fig, out_path)


def plot_sample_histogram(samples: np.ndarray, grid_size: int, out_path: Path):
    bins = np.arange(grid_size + 2) - 0.5
    fig, ax = plt.subplots(figsize=(5, 5))
    hist = ax.hist2d(samples[:, 0], samples[:, 1], bins=[bins, bins])
    ax.set_title("Sampled terminal states")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(hist[3], ax=ax, fraction=0.046, pad=0.04)
    _save_figure(fig, out_path)


def plot_summary_grid(summary_rows: Iterable[Dict[str, object]], out_path: Path):
    rows = list(summary_rows)
    if not rows:
        raise ValueError("summary_rows must not be empty")

    fig, axes = plt.subplots(len(rows), 3, figsize=(12, 4 * len(rows)))
    if len(rows) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_idx, row in enumerate(rows):
        reward_name = str(row["reward"])
        target = np.asarray(row["target"])
        learned = np.asarray(row["learned"])
        samples = np.asarray(row["samples"])
        grid_size = int(row["grid_size"])

        axes[row_idx, 0].imshow(target.T, origin="lower", interpolation="nearest")
        axes[row_idx, 0].set_title(f"{reward_name}: target")
        axes[row_idx, 0].set_xlabel("x")
        axes[row_idx, 0].set_ylabel("y")

        axes[row_idx, 1].imshow(learned.T, origin="lower", interpolation="nearest")
        axes[row_idx, 1].set_title(f"{reward_name}: learned")
        axes[row_idx, 1].set_xlabel("x")
        axes[row_idx, 1].set_ylabel("y")

        bins = np.arange(grid_size + 2) - 0.5
        axes[row_idx, 2].hist2d(samples[:, 0], samples[:, 1], bins=[bins, bins])
        axes[row_idx, 2].set_title(f"{reward_name}: samples")
        axes[row_idx, 2].set_xlabel("x")
        axes[row_idx, 2].set_ylabel("y")

    _save_figure(fig, out_path)
