"""Local state-dictionary checkpoints with explicit architecture provenance."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from chromatica.datasets.transform import LAB_ENCODING
from chromatica.models.loader import architecture_name, create_model
from chromatica.runtime import file_sha256

if TYPE_CHECKING:
    from chromatica.models.base import BaseCNN


def save_checkpoint(
    path: Path, model: BaseCNN, architecture: str, metadata: dict[str, Any]
) -> None:
    """Store CPU tensors and primitive metadata, preserving parameter keys."""
    torch.save(
        {
            "format_version": 1,
            "architecture": architecture_name(architecture),
            "lab_encoding": LAB_ENCODING,
            "state_dict": {
                name: value.detach().cpu() for name, value in model.state_dict().items()
            },
            "metadata": metadata,
        },
        Path(path),
    )


def load_checkpoint(
    path: Path, *, architecture: str | None = None, device: str | torch.device = "cpu"
) -> tuple[BaseCNN, dict[str, Any]]:
    """Load tensor checkpoints with weights_only=True; legacy dictionaries need a name.

    Full pickled model objects are intentionally unsupported. Old v1/v2
    state dictionaries retain their parameter keys and use the same Lab
    encoding. File provenance alone cannot establish a training protocol.
    """
    path = Path(path).expanduser().resolve()
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        msg = "Checkpoint must contain a state dictionary or tensor checkpoint."
        raise TypeError(msg)
    if "state_dict" in payload:
        if (
            payload.get("format_version") != 1
            or payload.get("lab_encoding") != LAB_ENCODING
        ):
            msg = "Unsupported checkpoint format or Lab encoding."
            raise ValueError(msg)
        canonical = architecture_name(payload["architecture"])
        if architecture is not None and architecture_name(architecture) != canonical:
            msg = "Requested architecture does not match checkpoint metadata."
            raise ValueError(msg)
        state = payload["state_dict"]
        metadata = dict(payload.get("metadata", {}))
    else:
        if architecture is None:
            msg = "Legacy state dictionaries require an explicit architecture."
            raise ValueError(msg)
        canonical = architecture_name(architecture)
        state = payload
        metadata = {"legacy_state_dict": True, "training_provenance": "unknown"}
    if (
        not isinstance(state, Mapping)
        or not state
        or not all(isinstance(value, torch.Tensor) for value in state.values())
    ):
        msg = "State dictionary must be a nonempty mapping of tensors."
        raise TypeError(msg)
    model = create_model(canonical)
    model.load_state_dict(state, strict=True)
    model.to(device)
    metadata.update(
        {
            "architecture": canonical,
            "checkpoint_sha256": file_sha256(path),
            "checkpoint_filename": path.name,
            "lab_encoding": LAB_ENCODING,
        }
    )
    return model, metadata
