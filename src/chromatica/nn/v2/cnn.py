"""Class with U-Net like CNN with skip-connections."""

from typing import override

import torch
from torch import nn

from chromatica.nn.base import BaseCNN
from chromatica.nn.v2.layers import (
    ConvLayer,
    IncreaseResolutionLayer,
    ReduceResolutionLayer,
)


class CNN(BaseCNN):
    """U-Net like CNN with skip-connections."""

    def __init__(self):
        # TODO: Requires refactoring
        # I think it can be written better, for example,
        # using a generator or a loop, since there are repetitive patterns
        # here, but that's not necessary at the moment.
        # https://github.com/snailUlitka/chromatica/issues/8

        super().__init__()

        self.c1 = ConvLayer(1, 64)
        self.r1 = ReduceResolutionLayer(64, 64)
        self.c2 = ConvLayer(64, 128)
        self.c3 = ConvLayer(128, 128)
        self.r2 = ReduceResolutionLayer(128, 128)
        self.c4 = ConvLayer(128, 256)
        self.c5 = ConvLayer(256, 256)
        self.drop1 = nn.Dropout2d(0.3)
        self.r3 = ReduceResolutionLayer(256, 256)
        self.c6 = ConvLayer(256, 512)
        self.c7 = ConvLayer(512, 512)
        self.c8 = ConvLayer(512, 512)
        self.c9 = ConvLayer(512, 256)
        self.drop2 = nn.Dropout2d(0.3)

        self.i1 = IncreaseResolutionLayer(256, 256)
        self.c10 = ConvLayer(256 + 256, 128)
        self.c11 = ConvLayer(128, 128)
        self.i2 = IncreaseResolutionLayer(128, 128)
        self.c12 = ConvLayer(128 + 128, 64)
        self.c13 = ConvLayer(64, 64)
        self.i3 = IncreaseResolutionLayer(64, 64)
        self.c14 = ConvLayer(64 + 64, 2)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Colorize tensor with image in LAB colorspace.

        Transform tensor with (1, h, w) shape to (2, h, w).
        Input and output tensors are images in LAB colorspace.
        """
        x1 = self.c1(x)
        x2 = self.r1(x1)

        x3 = self.c2(x2)
        x4 = self.c3(x3)
        x5 = self.r2(x4)

        x6 = self.c4(x5)
        x7 = self.c5(x6)
        x7 = self.drop1(x7)
        x8 = self.r3(x7)

        x9 = self.c6(x8)
        x10 = self.c7(x9)
        x11 = self.c8(x10)
        x12 = self.c9(x11)
        x12 = self.drop2(x12)

        x13 = self.i1(x12)
        x13 = torch.cat([x13, x7], dim=1)  # skip from c5
        x14 = self.c10(x13)
        x15 = self.c11(x14)

        x16 = self.i2(x15)
        x16 = torch.cat([x16, x4], dim=1)  # skip from c3
        x17 = self.c12(x16)
        x18 = self.c13(x17)

        x19 = self.i3(x18)
        x19 = torch.cat([x19, x1], dim=1)  # skip from c1
        x20 = self.c14(x19)

        return x20
