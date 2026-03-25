"""Minimal GFlowNet package for toy 2D grid experiments."""

from .distillation import DistillConfig, GumbelTerminalGenerator, train_inverse_distillation
from .model import TabularFlowGFlowNet
from .rewards import available_rewards, build_reward_grid
from .training import TrainConfig, train_gflownet

__all__ = [
    "TabularFlowGFlowNet",
    "GumbelTerminalGenerator",
    "available_rewards",
    "build_reward_grid",
    "TrainConfig",
    "DistillConfig",
    "train_gflownet",
    "train_inverse_distillation",
]
