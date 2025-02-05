"""Module with CNN for colorization task."""

import torch
from torch import nn

from chromatica.cnn.layers import (
    ConvLayer,
    IncreaseResolutionLayer,
    ReduceResolutionLayer,
)


class CNN(nn.Module):
    """Implementation of Convolution Neural Network for colorization task."""

    def __init__(self):
        super().__init__()
        # TODO: Requires refactoring
        # I think it can be written better, for example,
        # using a generator or a loop, since there are repetitive patterns
        # here, but that's not necessary at the moment.
        # https://github.com/snailUlitka/chromatica/issues/8

        self.nn = nn.Sequential(
            ConvLayer(1, 64, activation=nn.ReLU()),
            ReduceResolutionLayer(64, 64),
            ConvLayer(64, 128, activation=nn.ReLU()),
            ConvLayer(128, 128, activation=nn.ReLU()),
            ReduceResolutionLayer(128, 128),
            ConvLayer(128, 256, activation=nn.ReLU()),
            ConvLayer(256, 256, activation=nn.ReLU()),
            nn.Dropout2d(0.3),
            ReduceResolutionLayer(256, 256),
            ConvLayer(256, 512, activation=nn.ReLU()),
            ConvLayer(512, 512, activation=nn.ReLU()),
            ConvLayer(512, 512, activation=nn.ReLU()),
            ConvLayer(512, 256, activation=nn.ReLU()),
            nn.Dropout2d(0.3),
            IncreaseResolutionLayer(256, 256),
            ConvLayer(256, 128, activation=nn.ReLU()),
            ConvLayer(128, 128, activation=nn.ReLU()),
            IncreaseResolutionLayer(128, 128),
            ConvLayer(128, 64, activation=nn.ReLU()),
            ConvLayer(64, 64, activation=nn.ReLU()),
            IncreaseResolutionLayer(64, 64),
            ConvLayer(64, 2, activation=nn.Tanh()),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Colorize tensor with image in LAB colorspace.

        Transform tensor with (1, h, w) shape to (3, h, w).
        Input and output tensors are images in LAB colorspace.
        """
        return self.nn(x)
