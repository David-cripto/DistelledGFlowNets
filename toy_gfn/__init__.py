"""Minimal GFlowNet package for toy 2D grid experiments."""

from .model import TabularFlowGFlowNet
from .rewards import available_rewards, build_reward_grid
from .training import TrainConfig, train_gflownet

__all__ = [
    "TabularFlowGFlowNet",
    "available_rewards",
    "build_reward_grid",
    "TrainConfig",
    "train_gflownet",
]
