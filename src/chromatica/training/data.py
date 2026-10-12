"""Deterministic local splits and loaders with persisted input manifests."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from chromatica.datasets.dataset import DirectoryImageDataset
from chromatica.runtime import file_sha256

if TYPE_CHECKING:
    from chromatica.config import DataConfig


MIN_SPLIT_SAMPLES = 2


def resolve_data_directory(config: DataConfig, *, split: str) -> Path:
    """Resolve a local directory, fetching ClearML only when its ID is selected."""
    if split not in {"train", "test"}:
        msg = "split must be train or test."
        raise ValueError(msg)
    local = config.train_dir if split == "train" else config.test_dir
    if local:
        return Path(local).expanduser().resolve()
    if config.clearml_dataset_id:
        from chromatica.integrations.clearml import dataset_directory

        root = dataset_directory(config.clearml_dataset_id)
        subdir = config.train_subdir if split == "train" else config.test_subdir
        return (root / subdir).resolve()
    msg = f"Set data.{split}_dir or an explicit data.clearml_dataset_id."
    raise ValueError(msg)


def seed_worker(worker_id: int) -> None:
    """Seed non-PyTorch transforms in each DataLoader worker."""
    del worker_id
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)  # noqa: NPY002 - Seed legacy third-party transforms.
    random.seed(seed)


def make_training_loaders(
    config: DataConfig, *, seed: int, device: torch.device
) -> tuple[DataLoader, DataLoader | None, dict[str, Any]]:
    """Split only the training directory; never use test images for validation."""
    root = resolve_data_directory(config, split="train")
    if config.test_dir:
        test_root = Path(config.test_dir).expanduser().resolve()
        if root == test_root or root in test_root.parents or test_root in root.parents:
            msg = "Train and test directories must be separate and non-nested."
            raise ValueError(msg)
    dataset = DirectoryImageDataset(
        root, image_size=config.image_size, recursive=config.recursive
    )
    if len(dataset) < MIN_SPLIT_SAMPLES and config.val_fraction > 0:
        msg = "At least two training images are required for a validation split."
        raise ValueError(msg)
    val_count = (
        max(1, int(len(dataset) * config.val_fraction))
        if config.val_fraction > 0
        else 0
    )
    order = torch.randperm(
        len(dataset), generator=torch.Generator().manual_seed(seed)
    ).tolist()
    val_indices = order[:val_count]
    train_indices = order[val_count:]
    train = Subset(dataset, train_indices)
    validation = Subset(dataset, val_indices) if val_indices else None
    common = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker,
    }
    train_loader = DataLoader(
        train, shuffle=True, generator=torch.Generator().manual_seed(seed), **common
    )
    val_loader = (
        DataLoader(validation, shuffle=False, **common)
        if validation is not None
        else None
    )
    manifest = dataset_manifest(dataset)
    manifest.update(
        {
            "seed": seed,
            "train_indices": train_indices,
            "val_indices": val_indices,
            "val_fraction": config.val_fraction,
        }
    )
    return train_loader, val_loader, manifest


def dataset_manifest(dataset: DirectoryImageDataset) -> dict[str, Any]:
    """Record relative filenames, byte sizes, and file-content SHA256 hashes."""
    files = [
        {
            "path": path.relative_to(dataset.root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in dataset.files
    ]
    return {"root": str(dataset.root), "files": files, "samples": len(dataset)}


def write_json(path: Path, data: Any) -> None:
    """Write primitive run metadata, mapping nonfinite metric values to null."""
    import math

    def clean(value: Any) -> Any:
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(item) for item in value]
        return value

    Path(path).write_text(
        json.dumps(clean(data), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
