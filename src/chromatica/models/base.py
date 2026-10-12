"""Tensor contract shared by the two historical colorization architectures."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn

from chromatica.contracts import BATCH_DIMENSIONS, IMAGE_DIMENSIONS, SPATIAL_MULTIPLE


class BaseCNN(nn.Module, ABC):
    """Map normalized L (N, 1, H, W) to normalized ab (N, 2, H, W).

    Single images (1, H, W) are accepted and return (2, H, W). Spatial
    dimensions must be positive multiples of eight. Architecture modules and
    parameter names are preserved for historical state dictionaries.
    """

    @staticmethod
    def prepare_input(x: torch.Tensor) -> tuple[torch.Tensor, bool]:
        """Validate shapes and promote an unbatched image to a batch."""
        single = x.ndim == IMAGE_DIMENSIONS
        batch = x.unsqueeze(0) if single else x
        if batch.ndim != BATCH_DIMENSIONS or batch.shape[1] != 1 or batch.shape[0] == 0:
            msg = "L must have shape (N, 1, H, W) or (1, H, W), with N > 0."
            raise ValueError(msg)
        if any(
            size < SPATIAL_MULTIPLE or size % SPATIAL_MULTIPLE
            for size in batch.shape[-2:]
        ):
            msg = "H and W must be positive multiples of eight."
            raise ValueError(msg)
        if not batch.is_floating_point():
            msg = "L must be floating point in normalized Lab units."
            raise TypeError(msg)
        return batch, single

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return normalized ab with the same batch and spatial dimensions."""
        raise NotImplementedError
