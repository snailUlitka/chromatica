"""Compatibility imports; new experiments use chromatica.models."""

from chromatica.models import BaseCNN
from chromatica.models.loader import load_cnn_class as load_cnn

DEFAULT_VERSION = "v2"

__all__ = ["DEFAULT_VERSION", "BaseCNN", "load_cnn"]
