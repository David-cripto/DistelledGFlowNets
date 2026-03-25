from __future__ import annotations

import argparse
from pathlib import Path

from toy_gfn.rewards import available_rewards
from toy_gfn.training import TrainConfig, save_run_artifacts, train_gflownet
from ipdb import set_trace as debug


def parse_args():
    parser = argparse.ArgumentParser(description="Train a tiny tabular GFlowNet on a 2D grid")
    parser.add_argument("--reward", choices=available_rewards(), default="mixture")
    parser.add_argument("--grid-size", type=int, default=20, help="Maximum x/y coordinate")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reward-floor", type=float, default=1e-4)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--num-terminal-samples", type=int, default=5000)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs") / "run")
    return parser.parse_args()


def main():
    args = parse_args()
    config = TrainConfig(
        reward=args.reward,
        grid_size=args.grid_size,
        steps=args.steps,
        lr=args.lr,
        seed=args.seed,
        reward_floor=args.reward_floor,
        eval_every=args.eval_every,
        num_terminal_samples=args.num_terminal_samples,
    )

    result = train_gflownet(config)
    save_run_artifacts(result, config, args.out_dir)
    save_dir = args.out_dir
    print(f"Saved run to: {save_dir}")


if __name__ == "__main__":
    main()
