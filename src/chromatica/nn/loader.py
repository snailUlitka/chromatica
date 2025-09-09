"""Util functions for load CNNs classes."""

from __future__ import annotations

import pkgutil
from functools import cache
from importlib import import_module
from pathlib import Path

from chromatica.nn.base import BaseCNN


@cache
def load_cnn_class(version: str) -> type[BaseCNN]:
    """Load model by version from `chromatica.nn.{version}.cnn.CNN`."""
    mod = import_module(f"chromatica.nn.{version}.cnn")
    cls = getattr(mod, "CNN", None)

    if cls is None:
        msg = f"`CNN` not found in chromatica.nn.{version}.cnn"
        raise ImportError(msg)
    if not issubclass(cls, BaseCNN):
        msg = f"{cls} must subclass BaseCNN"
        raise TypeError(msg)
    return cls


def available_versions() -> list[str]:
    """Search and show all available CNNs versions (starts with `v`)."""
    pkg_path = Path(__file__).parent

    return sorted(
        name
        for _, name, ispkg in pkgutil.iter_modules([str(pkg_path)])
        if ispkg and name.startswith("v")
    )
