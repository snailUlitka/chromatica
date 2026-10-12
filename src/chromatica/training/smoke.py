"""An opt-in tiny CPU contract experiment, separate from research runs."""

from __future__ import annotations

import json

import torch
from torch.utils.data import DataLoader, TensorDataset

from chromatica.config import EvaluationConfig, TrainingConfig
from chromatica.datasets.transform import RGB2LAB
from chromatica.evaluation.experiment import evaluate_model
from chromatica.models import create_model
from chromatica.runtime import seed_everything
from chromatica.training.loop import fit


def cpu_smoke() -> dict[str, object]:
    """Perform one CPU optimizer step per architecture on two 16x16 images.

    This checks plumbing only. Synthetic losses and predictions are not
    evidence about Food-101, COCO, or colorization quality.
    """
    torch.set_num_threads(1)
    seed_everything(42)
    lab = RGB2LAB()(torch.rand(2, 3, 16, 16))
    loader = DataLoader(
        TensorDataset(lab[:, :1], lab[:, 1:], torch.zeros(2, dtype=torch.long)),
        batch_size=2,
    )
    results = {}
    for name in ("encoder_decoder", "unet"):
        seed_everything(42)
        model = create_model(name)
        history = fit(
            model,
            loader,
            None,
            TrainingConfig(epochs=1, max_train_batches=1),
            device=torch.device("cpu"),
        )
        evaluation = evaluate_model(loader, model, EvaluationConfig(max_batches=1))
        results[name] = {
            "train_mse": history[0].train_mse,
            "metrics": evaluation["metrics"],
        }
    return {
        "kind": "synthetic_contract_check",
        "research_result": False,
        "architectures": results,
    }


def main() -> None:
    """Run only after a deliberate chromatica-smoke command invocation."""
    print(json.dumps(cpu_smoke(), indent=2))
