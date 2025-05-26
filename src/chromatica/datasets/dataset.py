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

# TODO: Implement lazy and parallel load logic
# https://github.com/snailUlitka/chromatica/issues/17

from pathlib import Path

import torch
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    """Not implemented yet."""

    def __init__(self, path_to_dir: Path) -> None:
        super().__init__()
        assert path_to_dir
        raise NotImplementedError

    def __len__(self) -> int:
        """Not implemented yet."""
        raise NotImplementedError

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Not implemented yet."""
        assert idx
        raise NotImplementedError

    def __getitems__(
        self,
        idxs: list[int],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Not implemented yet."""
        assert idxs
        raise NotImplementedError
