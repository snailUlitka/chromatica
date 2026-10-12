"""Compatibility registry for historical model version names."""

from chromatica.models.loader import load_cnn_class

__all__ = ["available_versions", "load_cnn_class"]


def available_versions() -> list[str]:
    """Return historical aliases retained for checkpoint compatibility."""
    return ["v1", "v2"]
