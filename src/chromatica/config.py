"""Serializable experiment settings; reading a config never starts a run."""

from __future__ import annotations

import math
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from chromatica.contracts import SPATIAL_MULTIPLE
from chromatica.models.loader import architecture_name


@dataclass(frozen=True)
class DataConfig:
    """Local split directories or an explicitly selected ClearML dataset ID."""

    name: str = "local"
    train_dir: str | None = None
    test_dir: str | None = None
    clearml_dataset_id: str | None = None
    train_subdir: str = "train"
    test_subdir: str = "test"
    image_size: int = 256
    batch_size: int = 8
    num_workers: int = 0
    val_fraction: float = 0.2
    recursive: bool = True

    def __post_init__(self) -> None:
        """Reject invalid shapes, loader sizes, and split fractions."""
        if self.image_size < SPATIAL_MULTIPLE or self.image_size % SPATIAL_MULTIPLE:
            msg = "image_size must be a positive multiple of eight."
            raise ValueError(msg)
        if self.batch_size < 1 or self.num_workers < 0:
            msg = "batch_size must be positive and num_workers nonnegative."
            raise ValueError(msg)
        if not 0 <= self.val_fraction < 1:
            msg = "val_fraction must be in [0, 1)."
            raise ValueError(msg)


@dataclass(frozen=True)
class TrainingConfig:
    """Adam/MSE training parameters and optional explicit batch limits."""

    epochs: int = 3
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    use_amp: bool = False
    max_train_batches: int | None = None
    max_val_batches: int | None = None

    def __post_init__(self) -> None:
        """Reject invalid optimization parameters."""
        if (
            self.epochs < 1
            or not math.isfinite(self.learning_rate)
            or self.learning_rate <= 0
        ):
            msg = "epochs and learning_rate must be positive."
            raise ValueError(msg)
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            msg = "weight_decay must be finite and nonnegative."
            raise ValueError(msg)
        for limit in (self.max_train_batches, self.max_val_batches):
            if limit is not None and limit < 1:
                msg = "Batch limits must be positive or omitted."
                raise ValueError(msg)


@dataclass(frozen=True)
class EvaluationConfig:
    """Physical Lab metrics, histogram precision, and optional evaluation cap."""

    hue_bins: int = 360
    min_chroma: float = 8.0
    median_bins: int = 4096
    delta_a_min: float = -255.0
    delta_a_max: float = 255.0
    max_batches: int | None = None

    def __post_init__(self) -> None:
        """Validate histogram support before any predictions are produced."""
        if self.hue_bins < 1 or self.median_bins < 1:
            msg = "Histogram bin counts must be positive."
            raise ValueError(msg)
        if not math.isfinite(self.min_chroma) or self.min_chroma < 0:
            msg = "min_chroma must be finite and nonnegative."
            raise ValueError(msg)
        if not (
            math.isfinite(self.delta_a_min)
            and math.isfinite(self.delta_a_max)
            and self.delta_a_min < self.delta_a_max
        ):
            msg = "The finite delta-a histogram minimum must be below its maximum."
            raise ValueError(msg)
        if self.max_batches is not None and self.max_batches < 1:
            msg = "max_batches must be positive or omitted."
            raise ValueError(msg)


@dataclass(frozen=True)
class TrackingConfig:
    """Optional ClearML tracking; local runs are the default."""

    enabled: bool = False
    project: str = "Chromatica"
    task_name: str | None = None


@dataclass(frozen=True)
class ExperimentConfig:
    """Independent model/dataset run with explicit seed, device, and outputs."""

    architecture: str = "unet"
    seed: int = 42
    device: str = "cpu"
    deterministic: bool = True
    output_dir: str = ".runs/experiment"
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)

    def __post_init__(self) -> None:
        """Validate names and reject seeds unsupported by NumPy."""
        architecture_name(self.architecture)
        if not 0 <= self.seed < 2**32:
            msg = "seed must be in [0, 2**32)."
            raise ValueError(msg)
        if self.device not in {"auto", "cpu", "cuda", "mps"}:
            msg = "device must be auto, cpu, cuda, or mps."
            raise ValueError(msg)

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-compatible configuration data."""
        return asdict(self)


def load_config(path: Path) -> ExperimentConfig:
    """Read TOML; relative data/output paths are relative to the invocation cwd."""
    with Path(path).open("rb") as stream:
        raw = tomllib.load(stream)
    sections = {
        "data": DataConfig,
        "training": TrainingConfig,
        "evaluation": EvaluationConfig,
        "tracking": TrackingConfig,
    }
    values = dict(raw)
    for name, cls in sections.items():
        values[name] = cls(**values.pop(name, {}))
    return ExperimentConfig(**values)
