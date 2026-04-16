from __future__ import annotations

import argparse
from pathlib import Path

from .data import available_datasets
from .diffusion import TrainConfig, save_diffusion_artifacts, train_diffusion
from .diffusion.schedules import available_beta_schedules


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DDPM UNet noise predictor on an image dataset")
    parser.add_argument("--dataset", choices=available_datasets(), default="mnist")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train-steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--hidden-channels", type=int, default=64, help="Base channel width of the UNet")
    parser.add_argument("--depth", type=int, default=4, help="Number of UNet resolution levels")
    parser.add_argument("--time-frequencies", type=int, default=16, help="Number of sinusoidal time frequencies")
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--num-eval-samples", type=int, default=64)
    parser.add_argument("--num-sample-steps", type=int, default=128)
    parser.add_argument("--beta-schedule", choices=available_beta_schedules(), default="linear")
    parser.add_argument("--beta-start", type=float, default=1e-4)
    parser.add_argument("--beta-end", type=float, default=2e-2)
    parser.add_argument("--cosine-s", type=float, default=0.008)
    parser.add_argument("--num-workers", type=int, default=0)
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
        dataset=args.dataset,
        data_dir=str(args.data_dir),
        download=args.download,
        train_steps=args.train_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        hidden_channels=args.hidden_channels,
        depth=args.depth,
        time_frequencies=args.time_frequencies,
        eval_every=args.eval_every,
        num_eval_samples=args.num_eval_samples,
        num_sample_steps=args.num_sample_steps,
        beta_schedule=args.beta_schedule,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        cosine_s=args.cosine_s,
        num_workers=args.num_workers,
        device=args.device,
    )
    result = train_diffusion(config)
    save_diffusion_artifacts(result, config, args.out_dir)
    print(f"Saved run to: {args.out_dir}")


if __name__ == "__main__":
    main()
