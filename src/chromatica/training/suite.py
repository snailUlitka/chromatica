"""Explicit sequential execution of the configured research experiment matrix."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from chromatica.config import ExperimentConfig, load_config
from chromatica.datasets.dataset import DirectoryImageDataset
from chromatica.evaluation.experiment import evaluate_experiment
from chromatica.evaluation.html import write_html_report
from chromatica.runtime import select_device
from chromatica.training.data import MIN_SPLIT_SAMPLES, resolve_data_directory
from chromatica.training.experiment import train_experiment

DEFAULT_CONFIGS = (
    "configs/food101_encoder_decoder.toml",
    "configs/food101_unet.toml",
    "configs/coco_encoder_decoder.toml",
    "configs/coco_unet.toml",
)


def _check_inputs(config: ExperimentConfig) -> None:
    """Check split discovery and requested device before starting any training."""
    device = select_device(config.device)
    if config.training.use_amp and device.type != "cuda":
        msg = "Mixed precision training is supported only on CUDA."
        raise ValueError(msg)
    roots = [
        resolve_data_directory(config.data, split=key) for key in ("train", "test")
    ]
    train, test = roots
    if train == test or train in test.parents or test in train.parents:
        msg = "Train and test directories must be separate and non-nested."
        raise ValueError(msg)
    for root in roots:
        dataset = DirectoryImageDataset(
            root,
            image_size=config.data.image_size,
            recursive=config.data.recursive,
        )
        if (
            root == train
            and config.data.val_fraction > 0
            and len(dataset) < MIN_SPLIT_SAMPLES
        ):
            msg = f"At least two training images are required: {root}"
            raise ValueError(msg)


def _check_outputs(outputs: list[Path], reports_dir: Path) -> None:
    """Reject existing outputs, duplicate HTML names, and nested run directories."""
    names = [path.name for path in outputs]
    if len(set(names)) != len(names):
        msg = "Each experiment must have a unique output directory basename."
        raise ValueError(msg)
    index = reports_dir / "index.html"
    targets = [index, *(reports_dir / f"{name}.html" for name in names)]
    for path in [*outputs, *targets]:
        if path.exists():
            msg = f"Refusing to overwrite existing experiment output: {path}"
            raise FileExistsError(msg)
    for position, path in enumerate(outputs):
        others = [*outputs[position + 1 :], reports_dir]
        if any(path == other or path in other.parents for other in others):
            msg = "Run directories must not contain other runs or HTML reports."
            raise ValueError(msg)
        if any(other in path.parents for other in outputs[position + 1 :]):
            msg = "Run directories must not be nested."
            raise ValueError(msg)


def run_suite(
    config_paths: list[Path], reports_dir: Path, *, device: str | None = None
) -> Path:
    """Train/evaluate each config and save per-run HTML plus a final index.

    Paths are relative to the invocation directory. Existing run directories
    and reports cause failure before training. A later runtime failure stops
    the suite; completed checkpoints and individual reports remain available.
    """
    if not config_paths:
        msg = "At least one experiment config is required."
        raise ValueError(msg)
    configs = [load_config(path) for path in config_paths]
    if device is not None:
        configs = [replace(config, device=device) for config in configs]
    outputs = [Path(config.output_dir).expanduser().resolve() for config in configs]
    names = [path.name for path in outputs]
    reports_dir = reports_dir.expanduser().resolve()
    index = reports_dir / "index.html"
    _check_outputs(outputs, reports_dir)
    for config in configs:
        _check_inputs(config)
    reports_dir.mkdir(parents=True, exist_ok=True)
    evaluations = []
    for config, output, name in zip(configs, outputs, names, strict=True):
        print(f"Training {config.data.name} / {config.architecture}", flush=True)
        checkpoint = train_experiment(config)
        evaluation = output / "evaluation.json"
        print(f"Evaluating {checkpoint}", flush=True)
        evaluate_experiment(config, checkpoint, output=evaluation)
        write_html_report([evaluation], reports_dir / f"{name}.html")
        evaluations.append(evaluation)
    write_html_report(evaluations, index)
    return index


def main() -> None:
    """Run the four default experiments only on deliberate CLI invocation."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate experiments, saving standalone HTML reports."
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        type=Path,
        default=[Path(path) for path in DEFAULT_CONFIGS],
    )
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--device", choices=("cpu", "cuda", "mps", "auto"))
    args = parser.parse_args()
    print(run_suite(args.configs, args.reports_dir, device=args.device))
