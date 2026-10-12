"""Deterministic local image discovery and CPU preprocessing."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.v2 import Compose, PILToTensor, Resize, ToDtype

from chromatica.contracts import SPATIAL_MULTIPLE
from chromatica.datasets.transform import RGB2LAB


class DirectoryImageDataset(Dataset):
    """Load real filenames as (normalized L, normalized ab, integer label).

    Samples are CPU tensors of shape (1, H, W) and (2, H, W). Device transfer
    belongs to the training/evaluation loop, allowing multiprocess loaders.
    Recursive folders use the first relative directory as a class label;
    flat folders use the filename prefix before the final underscore.
    """

    _IMAGE_EXTENSIONS: ClassVar[set[str]] = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(
        self,
        path_to_dir: Path,
        device: torch.device | None = None,
        *,
        image_size: int = 256,
        recursive: bool = True,
    ) -> None:
        super().__init__()
        self.root = Path(path_to_dir).expanduser().resolve()
        if not self.root.is_dir():
            msg = f"Image directory does not exist: {self.root}"
            raise FileNotFoundError(msg)
        if image_size < SPATIAL_MULTIPLE or image_size % SPATIAL_MULTIPLE:
            msg = "image_size must be a positive multiple of eight."
            raise ValueError(msg)
        if device is not None and torch.device(device).type != "cpu":
            msg = "Datasets produce CPU tensors; transfer batches in the run loop."
            raise ValueError(msg)
        candidates = self.root.rglob("*") if recursive else self.root.iterdir()
        self.files = sorted(
            (
                path
                for path in candidates
                if path.is_file()
                and path.suffix.lower() in self._IMAGE_EXTENSIONS
                and not any(
                    part.startswith(".") for part in path.relative_to(self.root).parts
                )
            ),
            key=lambda path: path.relative_to(self.root).as_posix(),
        )
        if not self.files:
            msg = f"No image files found in {self.root}."
            raise ValueError(msg)
        self._labels = sorted({self._label(path) for path in self.files})
        self._label_ids = {label: index for index, label in enumerate(self._labels)}
        self._transform = Compose(
            [
                PILToTensor(),
                Resize((image_size, image_size), antialias=True),
                ToDtype(torch.float32, scale=True),
                RGB2LAB(),
            ]
        )

    def _label(self, path: Path) -> str:
        relative = path.relative_to(self.root)
        return (
            relative.parts[0]
            if len(relative.parts) > 1
            else path.stem.rsplit("_", 1)[0]
        )

    @property
    def labels(self) -> list[str]:
        """Return sorted labels discovered from the actual filenames."""
        return self._labels.copy()

    def label_number(self, label: str) -> int:
        """Return a deterministic label index."""
        return self._label_ids[label]

    def label_from_index(self, idx: int) -> str:
        """Return the label associated with an existing sample."""
        return self._label(self.files[idx])

    def label_number_by_path(self, path: Path) -> int:
        """Return the label index for a path under this dataset root."""
        return self.label_number(self._label(path))

    def __len__(self) -> int:
        """Return the number of discovered image files."""
        return len(self.files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Decode one RGB image and return normalized L/ab CPU tensors."""
        if not -len(self) <= idx < len(self):
            raise IndexError(idx)
        path = self.files[idx]
        with Image.open(path) as image:
            lab = self._transform(image.convert("RGB"))
        return lab[:1], lab[1:], self.label_number_by_path(path)


class ImageDataset(DirectoryImageDataset):
    """Compatibility name for flat historical Food-101/COCO directories.

    Arbitrary filenames and unequal class sizes are supported; consecutive
    class indices are no longer synthesized into paths.
    """
