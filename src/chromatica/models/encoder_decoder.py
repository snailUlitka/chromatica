"""Class with U-Net like CNN without skip-connections."""

from typing import override

import torch
from torch import nn

from chromatica.models.base import BaseCNN
from chromatica.models.encoder_decoder_layers import (
    ConvLayer,
    IncreaseResolutionLayer,
    ReduceResolutionLayer,
)


class EncoderDecoder(BaseCNN):
    """U-Net like CNN without skip-connections."""

    def __init__(self) -> None:
        super().__init__()
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

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Colorize tensor with image in LAB colorspace.

        Transform tensor with (1, h, w) shape to (2, h, w).
        Input and output tensors are images in LAB colorspace.
        """
        x, single = self.prepare_input(x)
        output = self.nn(x)
        return output[0] if single else output
