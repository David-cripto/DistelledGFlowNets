from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from .data import available_datasets
from .diffusion import TrainConfig as DiffusionTrainConfig
from .diffusion import load_diffusion_checkpoint, train_diffusion
from .diffusion.schedules import DDPMSchedule, available_beta_schedules
from .reward import (
    DetailedBalanceTrainConfig,
    available_detailed_balance_models,
    save_detailed_balance_run_artifacts,
    train_detailed_balance_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train f(x, t) with a detailed-balance loss on DDPM image transitions from a UNet denoiser"
    )
    parser.add_argument("--diffusion-checkpoint", type=Path, default=None)
    parser.add_argument("--dataset", choices=available_datasets(), default="mnist")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs") / "denoising_diffusion" / "detailed_balance_run",
    )

    parser.add_argument("--diffusion-train-steps", type=int, default=10000)
    parser.add_argument("--diffusion-batch-size", type=int, default=128)
    parser.add_argument("--diffusion-lr", type=float, default=2e-4)
    parser.add_argument("--diffusion-hidden-channels", type=int, default=64)
    parser.add_argument("--diffusion-depth", type=int, default=4)
    parser.add_argument("--diffusion-time-frequencies", type=int, default=16)
    parser.add_argument("--diffusion-eval-every", type=int, default=500)
    parser.add_argument("--diffusion-eval-samples", type=int, default=64)
    parser.add_argument("--num-sample-steps", type=int, default=None)
    parser.add_argument("--beta-schedule", choices=available_beta_schedules(), default="linear")
    parser.add_argument("--beta-start", type=float, default=1e-4)
    parser.add_argument("--beta-end", type=float, default=2e-2)
    parser.add_argument("--cosine-s", type=float, default=0.008)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--reward-train-steps", type=int, default=4000)
    parser.add_argument("--reward-lr", type=float, default=1e-4)
    parser.add_argument("--reward-model", choices=available_detailed_balance_models(), default="direct")
    parser.add_argument("--reward-hidden-dim", type=int, default=512)
    parser.add_argument("--reward-depth", type=int, default=3)
    parser.add_argument("--reward-time-frequencies", type=int, default=16)
    parser.add_argument("--boundary-penalty-weight", type=float, default=0.1)
    parser.add_argument("--reward-eval-every", type=int, default=250)
    parser.add_argument(
        "--transition-trajectories",
        type=int,
        default=64,
        help="Number of data examples used in each reward minibatch",
    )
    parser.add_argument("--reward-eval-batch-size", type=int, default=256)
    parser.add_argument("--num-preview-samples", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.diffusion_checkpoint is None:
        diffusion_config = DiffusionTrainConfig(
            dataset=args.dataset,
            data_dir=str(args.data_dir),
            download=args.download,
            train_steps=args.diffusion_train_steps,
            batch_size=args.diffusion_batch_size,
            lr=args.diffusion_lr,
            seed=args.seed,
            hidden_channels=args.diffusion_hidden_channels,
            depth=args.diffusion_depth,
            time_frequencies=args.diffusion_time_frequencies,
            eval_every=args.diffusion_eval_every,
            num_eval_samples=args.diffusion_eval_samples,
            num_sample_steps=args.num_sample_steps or 128,
            beta_schedule=args.beta_schedule,
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            cosine_s=args.cosine_s,
            num_workers=args.num_workers,
            device=args.device,
        )
        diffusion_result = train_diffusion(diffusion_config)
        denoiser = diffusion_result.model
        reference_distribution = diffusion_result.reference_distribution
        dataset_info = diffusion_result.dataset_info
    else:
        denoiser, reference_distribution, dataset_info, diffusion_config = load_diffusion_checkpoint(
            args.diffusion_checkpoint,
            device=args.device,
        )
        if args.num_sample_steps is not None and args.num_sample_steps != diffusion_config.num_sample_steps:
            raise ValueError(
                "The reward schedule must match the diffusion checkpoint. "
                f"Checkpoint uses num_sample_steps={diffusion_config.num_sample_steps}, "
                f"received {args.num_sample_steps}."
            )

    schedule = DDPMSchedule(
        num_steps=diffusion_config.num_sample_steps,
        beta_schedule=diffusion_config.beta_schedule,
        beta_start=diffusion_config.beta_start,
        beta_end=diffusion_config.beta_end,
        cosine_s=diffusion_config.cosine_s,
    )
    reward_config = DetailedBalanceTrainConfig(
        dataset=diffusion_config.dataset,
        data_dir=str(args.data_dir),
        download=args.download,
        train_steps=args.reward_train_steps,
        batch_num_trajectories=args.transition_trajectories,
        eval_batch_size=args.reward_eval_batch_size,
        lr=args.reward_lr,
        seed=args.seed,
        model_type=args.reward_model,
        hidden_dim=args.reward_hidden_dim,
        depth=args.reward_depth,
        time_frequencies=args.reward_time_frequencies,
        boundary_penalty_weight=args.boundary_penalty_weight,
        eval_every=args.reward_eval_every,
        num_sample_steps=diffusion_config.num_sample_steps,
        num_preview_samples=args.num_preview_samples,
        num_workers=args.num_workers,
        device=args.device,
    )
    reward_result = train_detailed_balance_model(
        config=reward_config,
        denoiser=denoiser,
        reference_distribution=reference_distribution,
        dataset_info=dataset_info,
        schedule=schedule,
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
