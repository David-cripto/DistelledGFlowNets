from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


AVAILABLE_DATASETS = ("mnist",)
_LOG_2PI = math.log(2.0 * math.pi)


@dataclass(frozen=True)
class DatasetInfo:
    name: str
    image_shape: tuple[int, int, int]
    num_classes: int
    channel_mean: tuple[float, ...]
    channel_std: tuple[float, ...]

    @property
    def flat_dim(self) -> int:
        channels, height, width = self.image_shape
        return channels * height * width

    def denormalize(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim < 3:
            raise ValueError("Expected image tensor with at least 3 dimensions.")
        mean = torch.tensor(self.channel_mean, device=images.device, dtype=images.dtype)
        std = torch.tensor(self.channel_std, device=images.device, dtype=images.dtype)
        shape = [1] * images.ndim
        shape[-3] = len(self.channel_mean)
        return images * std.view(*shape) + mean.view(*shape)


@dataclass(frozen=True)
class StandardNormalReference:
    image_shape: tuple[int, int, int]
    name: str = "gaussian"

    @property
    def flat_dim(self) -> int:
        channels, height, width = self.image_shape
        return channels * height * width

    def sample(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randn((batch_size, *self.image_shape), device=device)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("Expected x to have shape [batch, channels, height, width].")
        flat = x.reshape(x.shape[0], -1)
        return -0.5 * (flat.pow(2).sum(dim=1) + flat.shape[1] * _LOG_2PI)


def available_datasets() -> tuple[str, ...]:
    return AVAILABLE_DATASETS


def get_dataset_info(name: str) -> DatasetInfo:
    if name == "mnist":
        return DatasetInfo(
            name="mnist",
            image_shape=(1, 28, 28),
            num_classes=10,
            channel_mean=(0.5,),
            channel_std=(0.5,),
        )
    raise ValueError(f"Unsupported dataset '{name}'. Expected one of {', '.join(AVAILABLE_DATASETS)}.")


def build_dataset(
    name: str,
    root: str | Path,
    *,
    train: bool,
    download: bool,
) -> Dataset:
    root = Path(root)
    dataset_info = get_dataset_info(name)

    if name == "mnist":
        spatial_size = dataset_info.image_shape[1:]
        transform = transforms.Compose(
            [
                transforms.Resize(spatial_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=dataset_info.channel_mean, std=dataset_info.channel_std),
            ]
        )
        return datasets.MNIST(
            root=str(root),
            train=train,
            download=download,
            transform=transform,
        )

    raise ValueError(f"Unsupported dataset '{name}'. Expected one of {', '.join(AVAILABLE_DATASETS)}.")


def extract_images(batch: object) -> torch.Tensor:
    if isinstance(batch, (tuple, list)) and batch:
        images = batch[0]
    else:
        images = batch

    if not isinstance(images, torch.Tensor):
        raise TypeError("Expected a tensor batch or a dataset batch whose first item is a tensor.")
    if images.ndim != 4:
        raise ValueError("Expected images to have shape [batch, channels, height, width].")
    return images
