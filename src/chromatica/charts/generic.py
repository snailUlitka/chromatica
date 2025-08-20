"""Module with functions for draw generic charts, like show lab image."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from chromatica.datasets.transform import LAB2RGB

if TYPE_CHECKING:
    import numpy as np


_TWO_DIMENSIONS = 2


def lab2image(l_channel: torch.Tensor, ab: torch.Tensor) -> np.ndarray:
    """Convert L and ab channels to an RGB image.

    Parameters
    ----------
    l_channel : torch.Tensor
        Tensor with the L channel. Shape ``(1, H, W)`` or ``(H, W)``.
    ab : torch.Tensor
        Tensor with the ``a`` and ``b`` channels. Shape ``(2, H, W)``.

    Returns
    -------
    numpy.ndarray
        Image in RGB color space scaled to ``[0, 1]`` and shaped ``(H, W, 3)``.
    """
    if l_channel.ndim == _TWO_DIMENSIONS:
        l_channel = l_channel.unsqueeze(0)

    lab_tensor = torch.cat((l_channel, ab), dim=0)
    rgb_tensor = LAB2RGB()(lab_tensor)

    return rgb_tensor.permute(1, 2, 0).detach().numpy()
