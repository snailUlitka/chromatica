"""Opt-in ClearML datasets, tracking, and checkpoint artifact retrieval."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _sdk() -> Any:
    """Import the optional SDK only for an explicitly requested integration."""
    try:
        import clearml
    except ImportError as error:
        msg = "Add --extra clearml to uv sync with your selected PyTorch extra."
        raise ImportError(msg) from error
    return clearml


def dataset_directory(dataset_id: str) -> Path:
    """Fetch an immutable ClearML dataset ID; callers opt in through config."""
    return Path(_sdk().Dataset.get(dataset_id=dataset_id).get_local_copy())


def checkpoint_artifact(task_id: str, artifact_name: str) -> Path:
    """Read a named artifact from an explicitly selected training task."""
    task = _sdk().Task.get_task(task_id=task_id)
    if artifact_name not in task.artifacts:
        msg = f"Task {task_id} has no artifact {artifact_name!r}."
        raise KeyError(msg)
    path = task.artifacts[artifact_name].get_local_copy()
    if path is None:
        msg = f"Artifact {artifact_name!r} is unavailable."
        raise FileNotFoundError(msg)
    return Path(path)


def start_tracking(
    project: str, name: str, config: dict[str, Any], *, task_type: str = "training"
) -> Any:
    """Create a new tracking task for a deliberately started experiment."""
    task = _sdk().Task.init(
        project_name=project,
        task_name=name,
        task_type=task_type,
        reuse_last_task_id=False,
        auto_connect_frameworks=False,
    )
    task.connect_configuration(config, name="experiment")
    return task
