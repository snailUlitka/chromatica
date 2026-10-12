"""Local independent training runs with optional ClearML tracking."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from chromatica.config import ExperimentConfig, load_config
from chromatica.datasets.transform import LAB_ENCODING, PREPROCESSING_VERSION
from chromatica.models import architecture_name, create_model
from chromatica.models.checkpoints import save_checkpoint
from chromatica.runtime import (
    environment_metadata,
    file_sha256,
    seed_everything,
    select_device,
)
from chromatica.training.data import make_training_loaders, write_json
from chromatica.training.loop import EpochMetrics, fit


def train_experiment(config: ExperimentConfig) -> Path:
    """Start one fresh model/dataset run and return its local checkpoint path.

    Existing run directories are never overwritten. COCO training cannot
    silently continue the Food-101 model: every call constructs a new model.
    """
    device = select_device(config.device)
    seed_everything(config.seed, deterministic=config.deterministic)
    train_loader, val_loader, manifest = make_training_loaders(
        config.data, seed=config.seed, device=device
    )
    model = create_model(config.architecture)
    output = Path(config.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    metadata = {
        "config": config.as_dict(),
        "environment": environment_metadata(device),
        "lab_encoding": LAB_ENCODING,
        "preprocessing_version": PREPROCESSING_VERSION,
        "architecture": architecture_name(config.architecture),
        "dataset": config.data.name,
    }
    write_json(output / "split_manifest.json", manifest)
    metadata["split_manifest_sha256"] = file_sha256(output / "split_manifest.json")
    write_json(output / "run.json", metadata)
    task = None
    if config.tracking.enabled:
        from chromatica.integrations.clearml import start_tracking

        task = start_tracking(
            config.tracking.project,
            config.tracking.task_name
            or f"{config.data.name}/{architecture_name(config.architecture)}",
            config.as_dict(),
        )

    def on_epoch(metrics: EpochMetrics) -> None:
        if task is not None:
            logger = task.get_logger()
            logger.report_scalar(
                "normalized_ab_mse", "train", metrics.train_mse, metrics.epoch
            )
            if metrics.val_mse is not None:
                logger.report_scalar(
                    "normalized_ab_mse", "validation", metrics.val_mse, metrics.epoch
                )

    try:
        history = fit(
            model,
            train_loader,
            val_loader,
            config.training,
            device=device,
            on_epoch=on_epoch,
        )
        write_json(output / "history.json", [epoch.as_dict() for epoch in history])
        checkpoint = output / "checkpoint.pth"
        save_checkpoint(checkpoint, model, config.architecture, metadata)
        if task is not None:
            for name in (
                "checkpoint.pth",
                "run.json",
                "split_manifest.json",
                "history.json",
            ):
                task.upload_artifact(name, output / name)
            task.mark_completed()
    except Exception:
        if task is not None:
            task.mark_failed()
        raise
    finally:
        if task is not None:
            task.close()
    return checkpoint


def main() -> None:
    """CLI entry point; training requires a deliberate command invocation."""
    parser = argparse.ArgumentParser(
        description="Train one independent colorization experiment."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=str)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.output:
        config = replace(config, output_dir=args.output)
    print(train_experiment(config))
