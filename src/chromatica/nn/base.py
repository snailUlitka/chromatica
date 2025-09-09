"""Base class for all CNNs in Chromatica."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from torch.nn import Module

if TYPE_CHECKING:
    from torch import Tensor


class BaseCNN(Module, ABC):
    """Contract for chromatica CNN colorizers (L -> ab)."""

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """
        Colorize tensor in CIE-Lab.

        Input:  (N, 1, H, W)  or (1, H, W) для single-sample
        Output: (N, 2, H, W)  или (2, H, W)
        """
        raise NotImplementedError
