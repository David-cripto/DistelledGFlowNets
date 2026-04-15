from __future__ import annotations

from typing import Optional, Tuple, Union

import torch


class Density2D:
    name: str

    def sample(
        self,
        num_samples: int,
        device: Union[str, torch.device] = "cpu",
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def log_prob(self, points: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def default_limits(self) -> Tuple[float, float, float, float]:
        raise NotImplementedError
