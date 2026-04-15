from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from .compatibility import (
    CompatibilityCheckConfig,
    load_diffusion_checkpoint,
    run_compatibility_check,
    save_compatibility_artifacts,
)
from .diffusion import TrainConfig, train_diffusion
from .distributions import available_distributions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check whether the learned reverse DDPM kernel is compatible with the forward diffusion kernel"
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--target", choices=available_distributions(), default="mixture4")
    parser.add_argument("--reference", choices=["gaussian"], default="gaussian")
    parser.add_argument("--train-steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--time-frequencies", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--num-eval-samples", type=int, default=2048)
    parser.add_argument("--num-sample-steps", type=int, default=64)
    parser.add_argument("--num-visualization-samples", type=int, default=4096)
    parser.add_argument("--num-trajectory-samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--grid-resolution", type=int, default=80)
    parser.add_argument("--kernel-chunk-size", type=int, default=256)
    parser.add_argument("--model-batch-size", type=int, default=4096)
    parser.add_argument("--num-plot-steps", type=int, default=4)
    parser.add_argument("--limit-scale", type=float, default=1.25)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs") / "denoising_diffusion" / "compatibility_run",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.checkpoint is None:
        diffusion_config = TrainConfig(
            target=args.target,
            reference=args.reference,
            train_steps=args.train_steps,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            hidden_dim=args.hidden_dim,
            depth=args.depth,
            time_frequencies=args.time_frequencies,
            eval_every=args.eval_every,
            num_eval_samples=args.num_eval_samples,
            num_sample_steps=args.num_sample_steps,
            num_visualization_samples=args.num_visualization_samples,
            num_trajectory_samples=args.num_trajectory_samples,
            device=args.device,
        )
        diffusion_result = train_diffusion(diffusion_config)
        denoiser = diffusion_result.model
        reference_distribution = diffusion_result.reference_distribution
        target_distribution = diffusion_result.target_distribution
    else:
        denoiser, reference_distribution, target_distribution, diffusion_config = load_diffusion_checkpoint(
            args.checkpoint,
            device=args.device,
        )

    compatibility_config = CompatibilityCheckConfig(
        grid_resolution=args.grid_resolution,
        kernel_chunk_size=args.kernel_chunk_size,
        model_batch_size=args.model_batch_size,
        num_plot_steps=args.num_plot_steps,
        limit_scale=args.limit_scale,
        device=args.device,
    )
    result = run_compatibility_check(
        config=compatibility_config,
        denoiser=denoiser,
        reference_distribution=reference_distribution,
        target_distribution=target_distribution,
        num_sample_steps=diffusion_config.num_sample_steps,
    )
    save_compatibility_artifacts(
        result=result,
        config=compatibility_config,
        diffusion_config=asdict(diffusion_config),
        out_dir=args.out_dir,
    )
    print(f"Saved compatibility check to: {args.out_dir}")


if __name__ == "__main__":
    main()
