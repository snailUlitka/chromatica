"""Module with layers for CNN, each include transform and activation."""

import torch
from torch import nn


class ConvLayer(nn.Module):
    """
    Layer with convolution without reduce resolution for use in CNN.

    ConvLayer use convolution fucntion from `nn.Conv2d`,
    batch normalization from `nn.BatchNorm2d` and provided activation function.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: nn.Module,
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding="same",
        )

        self.bn = nn.BatchNorm2d(out_channels)
        self.act = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transforms tensor with convolution."""
        return self.act(self.bn(self.conv(x)))
