"""One-pass offline metrics, input provenance, and comparable report tables."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from chromatica.config import EvaluationConfig, ExperimentConfig, load_config
from chromatica.datasets.dataset import DirectoryImageDataset
from chromatica.datasets.transform import (
    LAB_ENCODING,
    PREPROCESSING_VERSION,
    ab_to_pairs,
    decode_ab,
)
from chromatica.evaluation.predictions import iter_predictions
from chromatica.metrics.bias import BiasAccumulator
from chromatica.metrics.color_diversity import compare_red_fraction, compute_rai
from chromatica.metrics.hue import hue_counts
from chromatica.models.checkpoints import load_checkpoint
from chromatica.runtime import environment_metadata, seed_everything, select_device
from chromatica.training.data import (
    dataset_manifest,
    resolve_data_directory,
    write_json,
)

METRIC_VERSION = "physical-lab-v1"


def evaluate_model(
    loader: DataLoader,
    model: torch.nn.Module,
    config: EvaluationConfig,
    *,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Compute physical Lab errors and hue counts in one bounded prediction pass."""
    stats = BiasAccumulator(
        bins=config.median_bins, a_range=(config.delta_a_min, config.delta_a_max)
    )
    hist_gt = torch.zeros(config.hue_bins, dtype=torch.int64)
    hist_pred = torch.zeros_like(hist_gt)
    numerator, denominator = 0.0, 0.0
    for _, target, prediction, _ in iter_predictions(
        loader, model, device=device, max_batches=config.max_batches
    ):
        stats.update(prediction, target)
        gt, pred = (
            ab_to_pairs(decode_ab(target.double())),
            ab_to_pairs(decode_ab(prediction.double())),
        )
        hist_gt += hue_counts(gt, nbins=config.hue_bins, min_chroma=config.min_chroma)
        hist_pred += hue_counts(
            pred, nbins=config.hue_bins, min_chroma=config.min_chroma
        )
        weights = torch.linalg.vector_norm(gt, dim=1)
        numerator += ((pred[:, 0] - gt[:, 0]) * weights).sum().item()
        denominator += weights.sum().item()
    metrics = stats.summary()
    metrics.update(
        {
            "chroma_weighted_delta_a": numerator / denominator
            if denominator
            else float("nan"),
            "rai_gt": compute_rai(hist_gt),
            "rai_pred": compute_rai(hist_pred),
        }
    )
    red = compare_red_fraction(hist_pred, hist_gt)
    return {
        "metric_version": METRIC_VERSION,
        "lab_encoding": LAB_ENCODING,
        "preprocessing_version": PREPROCESSING_VERSION,
        "metrics": metrics,
        "hue_gt": hist_gt.tolist(),
        "hue_pred": hist_pred.tolist(),
        "red_sector": red,
        "notes": [
            "Hue excludes low-chroma pixels independently in originals/predictions.",
            "Signed delta-a describes directional bias, not diversity gain.",
            "Median delta-a is estimated; tail counts and bin width are reported.",
        ],
    }


def evaluate_experiment(
    config: ExperimentConfig, checkpoint: Path, *, output: Path
) -> dict[str, Any]:
    """Evaluate an explicitly selected local checkpoint and persist provenance."""
    if output.exists():
        msg = f"Refusing to overwrite an existing evaluation: {output}"
        raise FileExistsError(msg)
    device = select_device(config.device)
    seed_everything(config.seed, deterministic=config.deterministic)
    dataset = DirectoryImageDataset(
        resolve_data_directory(config.data, split="test"),
        image_size=config.data.image_size,
        recursive=config.data.recursive,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=device.type == "cuda",
    )
    model, checkpoint_metadata = load_checkpoint(
        checkpoint, architecture=config.architecture, device=device
    )
    manifest = dataset_manifest(dataset)
    fingerprint = hashlib.sha256(
        json.dumps(manifest["files"], sort_keys=True).encode()
    ).hexdigest()
    result = evaluate_model(loader, model, config.evaluation, device=device)
    result.update(
        {
            "config": config.as_dict(),
            "environment": environment_metadata(device),
            "checkpoint": checkpoint_metadata,
            "test_manifest_sha256": fingerprint,
            "test_manifest": manifest,
            "status": "evaluated",
            "samples_available": len(dataset),
            "evaluation_limited": config.evaluation.max_batches is not None,
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    return result


def _training_signature(checkpoint: dict[str, Any]) -> str | None:
    """Fingerprint comparable training conditions, keeping unknown history explicit."""
    config = checkpoint.get("config")
    split = checkpoint.get("split_manifest_sha256")
    if not config or not split:
        return None
    data = config["data"]
    protocol = {
        "seed": config["seed"],
        "device": config["device"],
        "deterministic": config["deterministic"],
        "data": {
            key: data[key]
            for key in ("name", "image_size", "batch_size", "val_fraction", "recursive")
        },
        "training": config["training"],
        "split_manifest_sha256": split,
        "preprocessing_version": checkpoint.get("preprocessing_version"),
        "source_sha256": checkpoint.get("environment", {}).get("source_sha256"),
    }
    if not protocol["preprocessing_version"] or not protocol["source_sha256"]:
        return None
    return hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()


def comparison_rows(paths: list[Path]) -> list[dict[str, Any]]:
    """Build table rows only for matching test inputs and evaluation settings.

    Training provenance is reported per row. These preserved architectures
    differ in output activation as well as skips, so the rows never establish
    an isolated skip-connection effect.
    """
    if not paths:
        msg = "At least one evaluation report is required."
        raise ValueError(msg)
    reports = [json.loads(Path(path).read_text(encoding="utf-8")) for path in paths]
    signatures = []
    for report in reports:
        if (
            report.get("status") != "evaluated"
            or report.get("metric_version") != METRIC_VERSION
            or report.get("lab_encoding") != LAB_ENCODING
            or report.get("preprocessing_version") != PREPROCESSING_VERSION
        ):
            msg = "Comparison requires completed reports with the current metrics."
            raise ValueError(msg)
        config = report["config"]
        evaluation = dict(config["evaluation"])
        if evaluation["max_batches"] is not None:
            evaluation["batch_size"] = config["data"]["batch_size"]
        signatures.append(
            (
                report["test_manifest_sha256"],
                config["data"]["image_size"],
                config["data"]["recursive"],
                json.dumps(evaluation, sort_keys=True),
            )
        )
    if len(set(signatures)) != 1:
        msg = "Reports require matching test inputs, image sizes, and metric settings."
        raise ValueError(msg)
    rows = []
    training_signatures = [_training_signature(item["checkpoint"]) for item in reports]
    matched_training = (
        None not in training_signatures and len(set(training_signatures)) == 1
    )
    for report, training_signature in zip(reports, training_signatures, strict=True):
        checkpoint = report["checkpoint"]
        training_config = checkpoint.get("config")
        rows.append(
            {
                "architecture": checkpoint["architecture"],
                "trained_on": checkpoint.get("dataset", "unknown"),
                "evaluated_on": report["config"]["data"]["name"],
                "checkpoint_sha256": checkpoint["checkpoint_sha256"],
                "training_provenance": "recorded" if training_config else "unknown",
                "training_config": training_config,
                "split_manifest_sha256": checkpoint.get("split_manifest_sha256"),
                "training_protocol_sha256": training_signature,
                "matched_training_protocol": matched_training,
                "isolated_skip_effect": False,
                **report["metrics"],
            }
        )
    return rows


def main() -> None:
    """CLI entry point for an explicitly requested offline evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate one local checkpoint offline."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps", "auto"))
    args = parser.parse_args()
    config = load_config(args.config)
    if args.device:
        config = replace(config, device=args.device)
    evaluate_experiment(config, args.checkpoint, output=args.output)
