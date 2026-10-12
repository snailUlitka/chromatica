"""Display conversion helpers; metrics are computed before RGB clipping."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from chromatica.contracts import AB_CHANNELS, IMAGE_DIMENSIONS, PAIR_DIMENSIONS
from chromatica.datasets.transform import LAB2RGB

if TYPE_CHECKING:
    import numpy as np


def lab2image(l_channel: torch.Tensor, ab: torch.Tensor) -> np.ndarray:
    """Convert normalized L (1, H, W) and ab (2, H, W) to HWC display RGB."""
    if l_channel.ndim == PAIR_DIMENSIONS:
        l_channel = l_channel.unsqueeze(0)
    if (
        l_channel.ndim != IMAGE_DIMENSIONS
        or ab.ndim != IMAGE_DIMENSIONS
        or l_channel.shape[0] != 1
        or ab.shape[0] != AB_CHANNELS
        or l_channel.shape[-2:] != ab.shape[-2:]
    ):
        msg = "Expected matching single-image L (1, H, W) and ab (2, H, W)."
        raise ValueError(msg)
    lab = torch.cat((l_channel.detach().cpu(), ab.detach().cpu()), dim=0).float()
    return LAB2RGB()(lab).movedim(0, -1).numpy()
