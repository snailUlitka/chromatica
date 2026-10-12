"""Explicit device selection, seeding, and run provenance."""

from __future__ import annotations

import hashlib
import importlib.metadata
import platform
import random
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


def select_device(requested: str | torch.device = "cpu") -> torch.device:
    """Select a requested available device; automatic selection is opt-in."""
    name = str(requested)
    if name == "auto":
        name = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        msg = "CUDA was requested but is unavailable."
        raise RuntimeError(msg)
    if device.type == "mps" and not torch.backends.mps.is_available():
        msg = "MPS was requested but is unavailable."
        raise RuntimeError(msg)
    if device.type not in {"cpu", "cuda", "mps"}:
        msg = "Supported devices are CPU, CUDA, and MPS."
        raise ValueError(msg)
    return device


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch when an experiment is explicitly started."""
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - Seed legacy third-party transforms.
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic


def file_sha256(path: Path) -> str:
    """Hash a local checkpoint or manifest without importing its content."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def environment_metadata(device: torch.device) -> dict[str, Any]:
    """Capture the actual environment at the time of an explicit run."""
    versions = {}
    for name in (
        "chromatica",
        "torch",
        "torchvision",
        "numpy",
        "scikit-image",
        "clearml",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": str(device),
        "packages": versions,
        "cuda_runtime": torch.version.cuda,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "source_sha256": source_sha256(),
    }


def source_sha256() -> str:
    """Fingerprint installed package sources without depending on a Git checkout."""
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()
