from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from .diffusion import TrainConfig as DiffusionTrainConfig
from .diffusion import train_diffusion
from .distributions import available_distributions
from .reward import (
    DetailedBalanceTrainConfig,
    available_detailed_balance_models,
    save_detailed_balance_run_artifacts,
    train_detailed_balance_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train f(x, t) with a detailed-balance loss on DDPM transitions")
    parser.add_argument("--target", choices=available_distributions(), default="mixture4")
    parser.add_argument("--reference", choices=["gaussian"], default="gaussian")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs") / "denoising_diffusion" / "detailed_balance_run",
    )

    parser.add_argument("--diffusion-train-steps", type=int, default=3000)
    parser.add_argument("--diffusion-batch-size", type=int, default=512)
    parser.add_argument("--diffusion-lr", type=float, default=1e-3)
    parser.add_argument("--diffusion-hidden-dim", type=int, default=128)
    parser.add_argument("--diffusion-depth", type=int, default=3)
    parser.add_argument("--diffusion-time-frequencies", type=int, default=8)
    parser.add_argument("--diffusion-eval-every", type=int, default=500)
    parser.add_argument("--diffusion-eval-samples", type=int, default=2048)
    parser.add_argument("--num-sample-steps", type=int, default=128)

    parser.add_argument("--reward-train-steps", type=int, default=2000)
    parser.add_argument("--reward-lr", type=float, default=1e-3)
    parser.add_argument("--reward-model", choices=available_detailed_balance_models(), default="direct")
    parser.add_argument("--reward-hidden-dim", type=int, default=128)
    parser.add_argument("--reward-depth", type=int, default=3)
    parser.add_argument("--reward-time-frequencies", type=int, default=8)
    parser.add_argument("--boundary-penalty-weight", type=float, default=0.1)
    parser.add_argument("--reward-eval-every", type=int, default=500)
    parser.add_argument("--transition-trajectories", type=int, default=128)
    parser.add_argument("--num-kde-samples", type=int, default=4096)
    parser.add_argument("--density-resolution", type=int, default=220)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    diffusion_config = DiffusionTrainConfig(
        target=args.target,
        reference=args.reference,
        train_steps=args.diffusion_train_steps,
        batch_size=args.diffusion_batch_size,
        lr=args.diffusion_lr,
        seed=args.seed,
        hidden_dim=args.diffusion_hidden_dim,
        depth=args.diffusion_depth,
        time_frequencies=args.diffusion_time_frequencies,
        eval_every=args.diffusion_eval_every,
        num_eval_samples=args.diffusion_eval_samples,
        num_sample_steps=args.num_sample_steps,
        num_visualization_samples=args.num_kde_samples,
        device=args.device,
    )
    diffusion_result = train_diffusion(diffusion_config)

    reward_config = DetailedBalanceTrainConfig(
        train_steps=args.reward_train_steps,
        batch_num_trajectories=args.transition_trajectories,
        lr=args.reward_lr,
        seed=args.seed,
        model_type=args.reward_model,
        hidden_dim=args.reward_hidden_dim,
        depth=args.reward_depth,
        time_frequencies=args.reward_time_frequencies,
        boundary_penalty_weight=args.boundary_penalty_weight,
        eval_every=args.reward_eval_every,
        num_sample_steps=args.num_sample_steps,
        num_kde_samples=args.num_kde_samples,
        density_resolution=args.density_resolution,
        device=args.device,
    )
    reward_result = train_detailed_balance_model(
        config=reward_config,
        denoiser=diffusion_result.model,
        reference_distribution=diffusion_result.reference_distribution,
        target_distribution=diffusion_result.target_distribution,
    )
    save_detailed_balance_run_artifacts(
        result=reward_result,
        config=reward_config,
        diffusion_config=asdict(diffusion_config),
        out_dir=args.out_dir,
    )
    print(f"Saved detailed-balance run to: {args.out_dir}")


if __name__ == "__main__":
    main()
