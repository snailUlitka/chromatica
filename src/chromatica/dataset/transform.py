"""Module with transform functions like: rgb2lab, lab2rgb and etc."""

import numpy as np
import torch
from skimage.color import lab2rgb, rgb2lab
from torch import nn


class RGB2LAB(nn.Module):
    """
    Transform tensor with image from RGB to LAB colorspace.

    This module uses `skimage.color`.
    """

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Do transformation.

        Parameters
        ----------
        tensor : torch.Tensor
            Tensor with image in RGB colorspace.
            Tensor should be with `torch.float32` dtype and scaled to [0, 1].

        Returns
        -------
        torch.Tensor
            Tensor with image in LAB colorspace.
            Scales:
                L - [0, 1]
                A, B - [-1, 1]
        """
        if tensor.dtype != torch.float32:
            msg = (
                "Try to use `torchvision.transforms.v2.ToDtype(torch.float32, "
                "scale=True)` before this transformation."
            )
            raise TypeError(msg)

        rgb_tensor = tensor.detach().numpy()

        lab_tensor = rgb2lab(rgb_tensor, channel_axis=0)

        l_channel = lab_tensor[:1, :, :] / 100
        ab_channels = (lab_tensor[1:, :, :] + 128) / 127.5 - 1

        lab_scaled_tensor = np.vstack((l_channel, ab_channels))

        return torch.Tensor(lab_scaled_tensor)


class LAB2RGB(nn.Module):
    """
    Transform tensor with image from LAB to RGB colorspace.

    This module uses `skimage.color`.
    """

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Do transformation.

        Parameters
        ----------
        tensor : torch.Tensor
            Tensor with image in LAB colorspace.
            Tensor should be with `torch.float32` dtype and
            scaled [0, 1] for L and [-1, 1] for a, b channels.

        Returns
        -------
        torch.Tensor
            Tensor with image in RGB colorspace scaled to [0, 1].
        """
        if tensor.dtype != torch.float32:
            msg = (
                "Try to use `torchvision.transforms.v2.ToDtype(torch.float32, "
                "scale=True)` before this transformation."
            )
            raise TypeError(msg)

        lab_tensor = tensor.detach().numpy()

        l_channel = lab_tensor[:1, :, :] * 100
        ab_channels = (lab_tensor[1:, :, :] + 1) * 127.5 - 128

        rgb_tensor = lab2rgb(
            np.vstack((l_channel, ab_channels)),
            channel_axis=0,
        )

        return torch.Tensor(rgb_tensor)
