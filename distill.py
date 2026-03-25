from __future__ import annotations

import argparse
from pathlib import Path

from toy_gfn.distillation import DistillConfig, save_distillation_artifacts, train_inverse_distillation
from ipdb import set_trace as debug

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inverse-distill a pretrained toy 2D GFlowNet into a one-pass Gumbel generator"
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to pretrained checkpoint.pt")
    parser.add_argument(
        "--target-distribution",
        type=Path,
        default=None,
        help="Optional override for target_distribution.npy",
    )
    parser.add_argument(
        "--reward-grid",
        type=Path,
        default=None,
        help="Optional override for reward_grid.npy; used to normalize the pretrained flows",
    )
    parser.add_argument("--steps", type=int, default=300, help="Number of outer alternating steps")
    parser.add_argument("--generator-lr", type=float, default=0.5)
    parser.add_argument("--f-lr", type=float, default=0.1)
    parser.add_argument("--f-updates-per-step", type=int, default=25)
    parser.add_argument("--f-warmup-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--num-terminal-samples", type=int, default=5000)
    parser.add_argument("--init-generator-scale", type=float, default=1e-2)
    parser.add_argument(
        "--init-aux-from-pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Initialize the inner GFlowNet from the normalized pretrained checkpoint",
    )
    parser.add_argument(
        "--clamp-negative-objective",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Clamp negative finite-inner-loop gaps to zero during generator updates",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs") / "distilled",
        help="Directory where inverse-distillation artifacts will be written",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DistillConfig(
        checkpoint_path=args.checkpoint,
        out_dir=args.out_dir,
        steps=args.steps,
        generator_lr=args.generator_lr,
        f_lr=args.f_lr,
        f_updates_per_step=args.f_updates_per_step,
        f_warmup_steps=args.f_warmup_steps,
        eval_every=args.eval_every,
        num_terminal_samples=args.num_terminal_samples,
        seed=args.seed,
        init_generator_scale=args.init_generator_scale,
        init_aux_from_pretrained=args.init_aux_from_pretrained,
        target_distribution_path=args.target_distribution,
        reward_grid_path=args.reward_grid,
        clamp_negative_objective=args.clamp_negative_objective,
    )
    result = train_inverse_distillation(config)
    save_distillation_artifacts(result, config, args.out_dir)
    print(f"Saved inverse-distillation run to: {args.out_dir}")


if __name__ == "__main__":
    main()
