from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn


class Actions:
    RIGHT = 0
    UP = 1
    STOP = 2


@dataclass(frozen=True)
class SampleBatch:
    terminal_x: torch.Tensor
    terminal_y: torch.Tensor


class TabularFlowGFlowNet(nn.Module):
    """A tiny tabular GFlowNet over a 2D grid.

    States are integer coordinates `(x, y)` with `0 <= x, y <= grid_size`.
    From each state, the forward policy can move right, move up, or stop.
    The same terminal state can be reached through multiple right/up orderings,
    which makes the environment a simple DAG for GFlowNet experiments.
    """

    def __init__(self, grid_size: int):
        super().__init__()
        if grid_size < 1:
            raise ValueError("grid_size must be at least 1")

        self.grid_size = int(grid_size)
        n = self.grid_size + 1

        self.log_edge_flows = nn.Parameter(torch.zeros(n, n, 3))

        x_idx = torch.arange(n, dtype=torch.long)[:, None].expand(n, n)
        y_idx = torch.arange(n, dtype=torch.long)[None, :].expand(n, n)
        action_mask = torch.stack(
            [
                x_idx < self.grid_size,
                y_idx < self.grid_size,
                torch.ones_like(x_idx, dtype=torch.bool),
            ],
            dim=-1,
        )

        self.register_buffer("x_idx", x_idx, persistent=False)
        self.register_buffer("y_idx", y_idx, persistent=False)
        self.register_buffer("action_mask", action_mask, persistent=False)

    def masked_log_flows(self):
        """Return log edge flows with invalid actions masked out."""
        return self.log_edge_flows.masked_fill(~self.action_mask, -1e9)

    def forward_policy(self):
        """Return the normalized forward policy `P_F(a | s)` for all states."""
        log_flows = self.masked_log_flows()
        log_policy = log_flows - torch.logsumexp(log_flows, dim=-1, keepdim=True)
        return log_policy.exp()

    @torch.no_grad()
    def sample_terminal_points(self, num_samples: int):
        """Sample terminal grid points from the learned forward policy."""
        if num_samples < 1:
            raise ValueError("num_samples must be positive")

        policy = self.forward_policy()
        device = policy.device
        batch_size = int(num_samples)
        max_steps = 2 * self.grid_size + 1

        x = torch.zeros(batch_size, dtype=torch.long, device=device)
        y = torch.zeros(batch_size, dtype=torch.long, device=device)
        done = torch.zeros(batch_size, dtype=torch.bool, device=device)
        terminal_x = torch.zeros(batch_size, dtype=torch.long, device=device)
        terminal_y = torch.zeros(batch_size, dtype=torch.long, device=device)

        for _ in range(max_steps):
            alive = ~done
            if not bool(alive.any()):
                break

            state_probs = policy[x[alive], y[alive]]
            actions = torch.multinomial(state_probs, num_samples=1).squeeze(1)

            alive_indices = alive.nonzero(as_tuple=False).squeeze(1)
            cur_x = x[alive]
            cur_y = y[alive]

            stop_mask = actions == Actions.STOP
            if bool(stop_mask.any()):
                stop_indices = alive_indices[stop_mask]
                terminal_x[stop_indices] = cur_x[stop_mask]
                terminal_y[stop_indices] = cur_y[stop_mask]
                done[stop_indices] = True

            move_mask = ~stop_mask
            if bool(move_mask.any()):
                move_indices = alive_indices[move_mask]
                move_actions = actions[move_mask]
                x[move_indices] = cur_x[move_mask] + (move_actions == Actions.RIGHT).long()
                y[move_indices] = cur_y[move_mask] + (move_actions == Actions.UP).long()

        if not bool(done.all()):
            raise RuntimeError("Sampling did not terminate for all trajectories")

        return SampleBatch(terminal_x=terminal_x.cpu(), terminal_y=terminal_y.cpu())

    @property
    def shape(self):
        n = self.grid_size + 1
        return (n, n)
