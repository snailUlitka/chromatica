"""Module with layers for CNN, each include transform and activation."""

import torch
from torch import nn


class ConvLayer(nn.Module):
    """
    Layer with convolution without reduce resolution for use in CNN.

    `ConvLayer` use convolution fucntion from `nn.Conv2d`,
    batch normalization from `nn.BatchNorm2d` and provided activation function.

    The activation can be ReLU or Tanh. Tanh is used for the last layer in CNN.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: nn.ReLU | nn.Tanh,
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


class ReduceResolutionLayer(nn.Module):
    """
    Layer with convolution with reduce resolution for use in CNN.

    `ReduceResolutionLayer` use convolution fucntion from `nn.Conv2d`,
    batch normalization from `nn.BatchNorm2d` and ReLU as activation function.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
        )

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transforms and reduces the resolution tensor with convolution."""
        return nn.ReLU()(self.bn(self.conv(x)))
