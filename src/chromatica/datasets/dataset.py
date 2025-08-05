"""Dataset class for loading images from a directory.

This module provides a Dataset class designed for efficiently loading images
from a specified directory, particularly useful in interactive environments
like Jupyter notebooks.

The Dataset class offers:

- Lazy Loading: Images are loaded on demand, minimizing memory usage,
    especially when dealing with large datasets.
- Parallel Loading:  Image loading can be parallelized to speed up the
    process, significantly reducing load times.
"""

import logging
from collections.abc import Sequence
from functools import cache
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.v2 import (
    RGB,
    Compose,
    PILToTensor,
    Resize,
    ToDtype,
)

from chromatica.datasets.transform import RGB2LAB


class ImageDataset(Dataset):
    """Dataset with CIE-Lab images from directory with images."""

    def __init__(self, path_to_dir: Path) -> None:
        super().__init__()
        self._path_to_dir = path_to_dir
        self._transform = Compose(
            [
                PILToTensor(),
                Resize((256, 256)),
                RGB(),
                ToDtype(torch.float32, scale=True),
                RGB2LAB(),
            ]
        )
        # TODO: Better logging with loguru
        # https://github.com/snailUlitka/chromatica/issues/23
        self._logger = logging.getLogger()

    @cache
    @staticmethod
    def _load_metadata(path_to_dir: Path) -> tuple[int, list[str]]:
        len_ = 0
        labels = set()

        for p in path_to_dir.iterdir():
            if p.is_file():
                len_ += 1

                name = p.name.removesuffix(p.suffix)
                labels.add(name.rsplit("_", 1)[0])

        return len_, sorted(labels)

    def _prepare_images(
        self, images: Sequence[Image.Image]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Use `self._transform` to list with images.

        Parameters
        ----------
        tensor : torch.Tensor
            Tensor shape: [B, 3, H, W] or [3, H, W], where 3 is Lab channels
        """
        tensors = [self._transform(img) for img in images]
        batch = torch.stack(tensors, dim=0)

        l_batch = batch[:, :1, :, :]
        ab_batch = batch[:, 1:, :, :]

        return l_batch, ab_batch

    def _get_item_path(self, idx: int) -> Path:
        class_name = self.label_from_index(idx)
        index_in_class = idx % (len(self) // len(self.labels))

        path = self._path_to_dir / f"{class_name}_{index_in_class}.jpg"

        return path

    def label_number(self, label: str) -> int:
        """Return label's number from label name."""
        return self.labels.index(label)

    def label_from_index(self, idx: int) -> str:
        """Return label name from item's index."""
        return self.labels[idx // (len(self) // len(self.labels))]

    def __len__(self) -> int:
        """Lenght of dataset split."""
        return ImageDataset._load_metadata(self._path_to_dir)[0]

    @property
    def labels(self) -> list[str]:
        """All labels in dataset, sorted with `sorted`."""
        return ImageDataset._load_metadata(self._path_to_dir)[1]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        """
        Get one item from dataset.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Tuple with tensors - X (L channel), Y (ab channels) and label's number (int)
        """
        with Image.open(self._get_item_path(idx)).convert("RGB") as img:
            l_batch, ab_batch = self._prepare_images([img])

            return (
                l_batch[0],
                ab_batch[0],
                self.label_number(self.label_from_index(idx)),
            )
