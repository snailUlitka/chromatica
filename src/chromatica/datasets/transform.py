"""CIE-Lab units and non-differentiable RGB conversion using scikit-image.

The checkpoint-compatible encoding is L_n = L* / 100 and
ab_n = (ab* + 128) / 127.5 - 1. Its inverse is ab* = 127.5 * ab_n - 0.5.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
from skimage.color import lab2rgb, rgb2lab
from torch import nn

AB_SCALE = 127.5
AB_OFFSET = -0.5
LAB_ENCODING = "lab_affine_127_5_offset_minus_0_5_v1"
PREPROCESSING_VERSION = "rgb_bilinear_antialias_square_lab_v1"
type ABEncoding = Literal["normalized", "lab"]


def validate_channels(tensor: torch.Tensor, channels: int) -> None:
    """Validate a floating-point (C, H, W) or (N, C, H, W) tensor."""
    if tensor.ndim not in (3, 4) or tensor.shape[-3] != channels:
        msg = f"Expected (C, H, W) or (N, C, H, W), with C={channels}."
        raise ValueError(msg)
    if not tensor.is_floating_point():
        msg = "Color tensors must be floating point."
        raise TypeError(msg)
    if not torch.isfinite(tensor).all():
        msg = "Color tensors must contain finite values."
        raise ValueError(msg)


def encode_ab(ab: torch.Tensor) -> torch.Tensor:
    """Encode physical a*/b* into the historical model coordinates."""
    validate_channels(ab, 2)
    return (ab - AB_OFFSET) / AB_SCALE


def decode_ab(ab: torch.Tensor, *, encoding: ABEncoding = "normalized") -> torch.Tensor:
    """Return physical a*/b*; no range clamping or gamut correction occurs."""
    validate_channels(ab, 2)
    if encoding == "normalized":
        return ab * AB_SCALE + AB_OFFSET
    if encoding == "lab":
        return ab
    msg = "encoding must be normalized or lab."
    raise ValueError(msg)


def ab_to_pairs(ab: torch.Tensor) -> torch.Tensor:
    """Flatten ab into (N*H*W, 2), keeping channels from each pixel paired."""
    validate_channels(ab, 2)
    return ab.movedim(-3, -1).reshape(-1, 2)


def encode_lab(lab: torch.Tensor) -> torch.Tensor:
    """Encode physical Lab with channel axis -3 for images or batches."""
    validate_channels(lab, 3)
    return torch.cat(
        (lab.narrow(-3, 0, 1) / 100.0, encode_ab(lab.narrow(-3, 1, 2))), dim=-3
    )


def decode_lab(lab: torch.Tensor) -> torch.Tensor:
    """Decode model Lab coordinates to physical L*, a*, b*."""
    validate_channels(lab, 3)
    return torch.cat(
        (lab.narrow(-3, 0, 1) * 100.0, decode_ab(lab.narrow(-3, 1, 2))), dim=-3
    )


class RGB2LAB(nn.Module):
    """Convert float32 sRGB [0, 1] to checkpoint-compatible normalized Lab.

    Accepts (3, H, W) or (N, 3, H, W). Conversion detaches gradients and runs
    on CPU, then returns float32 on the input device. Use in preprocessing.
    """

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Return normalized Lab with the input shape and device."""
        validate_channels(tensor, 3)
        if tensor.dtype != torch.float32:
            msg = "RGB2LAB requires torch.float32 sRGB in [0, 1]."
            raise TypeError(msg)
        if tensor.numel() == 0 or (tensor < 0).any() or (tensor > 1).any():
            msg = "RGB images must be nonempty and have values in [0, 1]."
            raise ValueError(msg)
        rgb = tensor.detach().cpu().movedim(-3, -1).numpy()
        physical = torch.from_numpy(np.asarray(rgb2lab(rgb), dtype=np.float32)).movedim(
            -1, -3
        )
        return encode_lab(physical).to(tensor.device)


class LAB2RGB(nn.Module):
    """Convert normalized Lab to display sRGB [0, 1] via CPU scikit-image.

    Display conversion clips out-of-gamut colors; metrics must use physical
    Lab before this operation. Conversion detaches gradients.
    """

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Return float32 sRGB with the input shape and device."""
        validate_channels(tensor, 3)
        if tensor.dtype != torch.float32:
            msg = "LAB2RGB requires torch.float32 normalized Lab."
            raise TypeError(msg)
        if tensor.numel() == 0:
            msg = "Lab images must be nonempty."
            raise ValueError(msg)
        physical = decode_lab(tensor.detach().cpu()).movedim(-3, -1).numpy()
        rgb = torch.from_numpy(np.asarray(lab2rgb(physical), dtype=np.float32)).movedim(
            -1, -3
        )
        return rgb.to(tensor.device)
