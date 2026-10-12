"""Matched evaluation and honest training-provenance comparison contracts."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from chromatica.config import ExperimentConfig
from chromatica.datasets.transform import LAB_ENCODING, PREPROCESSING_VERSION
from chromatica.evaluation.experiment import METRIC_VERSION, comparison_rows


def fixture_report() -> dict:
    """Build a synthetic report fixture, without claiming a measured model result."""
    config = ExperimentConfig().as_dict()
    return {
        "status": "evaluated",
        "metric_version": METRIC_VERSION,
        "lab_encoding": LAB_ENCODING,
        "preprocessing_version": PREPROCESSING_VERSION,
        "test_manifest_sha256": "synthetic_test_fixture",
        "config": config,
        "metrics": {"mean_delta_a": 0.0},
        "checkpoint": {
            "architecture": "unet",
            "checkpoint_sha256": "synthetic_checkpoint_fixture",
            "dataset": "synthetic_fixture",
            "config": config,
            "split_manifest_sha256": "synthetic_split_fixture",
            "preprocessing_version": PREPROCESSING_VERSION,
            "environment": {"source_sha256": "synthetic_source_fixture"},
        },
    }


def write_reports(tmp_path: Path, reports: list[dict]) -> list[Path]:
    """Persist only synthetic JSON fixtures in a temporary test directory."""
    paths = []
    for index, report in enumerate(reports):
        path = tmp_path / f"report_{index}.json"
        path.write_text(json.dumps(report), encoding="utf-8")
        paths.append(path)
    return paths


@pytest.mark.parametrize("difference", ["files", "resolution", "limit", "encoding"])
def test_incompatible_evaluations_are_rejected(tmp_path: Path, difference: str) -> None:
    """Different populations, preprocessing, or metrics cannot form a matched table."""
    first = fixture_report()
    second = deepcopy(first)
    if difference == "files":
        second["test_manifest_sha256"] = "another_test_fixture"
    elif difference == "resolution":
        second["config"]["data"]["image_size"] = 128
    elif difference == "limit":
        second["config"]["evaluation"]["max_batches"] = 1
    else:
        second["lab_encoding"] = "unknown_encoding"
    with pytest.raises(ValueError, match=r"matching|current metrics"):
        comparison_rows(write_reports(tmp_path, [first, second]))


def test_matched_training_does_not_establish_a_skip_effect(tmp_path: Path) -> None:
    """Matching budgets still cannot remove the preserved activation confound."""
    first = fixture_report()
    second = deepcopy(first)
    second["checkpoint"]["architecture"] = "encoder_decoder"
    second["checkpoint"]["config"]["architecture"] = "encoder_decoder"
    rows = comparison_rows(write_reports(tmp_path, [first, second]))
    assert all(row["matched_training_protocol"] for row in rows)
    assert not any(row["isolated_skip_effect"] for row in rows)


@pytest.mark.parametrize("difference", ["seed", "unknown"])
def test_unmatched_or_unknown_training_is_explicit(
    tmp_path: Path, difference: str
) -> None:
    """Unknown history and unequal seeds cannot be reported as matched training."""
    first = fixture_report()
    second = deepcopy(first)
    if difference == "seed":
        second["checkpoint"]["config"]["seed"] = 17
    else:
        second["checkpoint"].pop("config")
    rows = comparison_rows(write_reports(tmp_path, [first, second]))
    assert not any(row["matched_training_protocol"] for row in rows)
    if difference == "unknown":
        assert rows[1]["training_provenance"] == "unknown"
        assert rows[1]["training_protocol_sha256"] is None
