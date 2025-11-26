"""An API for using Chromatica."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = ["app", "build_app"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        module = importlib.import_module("chromatica.api.app")
        return getattr(module, name)
    raise AttributeError(name)
