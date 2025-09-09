"""Different versions submodules for NN staff."""

from __future__ import annotations

import os

from .base import BaseCNN
from .loader import load_cnn_class as _load_cnn_class

__all__ = [
    "DEFAULT_VERSION",
    "BaseCNN",
    "load_cnn",
]

DEFAULT_VERSION = os.getenv("CHROMATICA_MODEL_VERSION", "v2")


def load_cnn(version: str) -> type[BaseCNN]:
    """Load model by version from `chromatica.nn.{version}.cnn.CNN`."""
    return _load_cnn_class(version)
