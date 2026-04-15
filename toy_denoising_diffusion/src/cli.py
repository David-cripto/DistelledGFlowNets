from __future__ import annotations

import argparse
from pathlib import Path

from .diffusion import TrainConfig, save_diffusion_artifacts, train_diffusion
from .distributions import available_distributions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small 2D DDPM denoising model")
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
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs") / "denoising_diffusion" / "run",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainConfig(
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
    result = train_diffusion(config)
    save_diffusion_artifacts(result, config, args.out_dir)
    print(f"Saved run to: {args.out_dir}")


if __name__ == "__main__":
    main()
