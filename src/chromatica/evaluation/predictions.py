"""Bounded prediction streams shared by metrics and visualization."""

from __future__ import annotations

from contextlib import nullcontext
from itertools import islice
from typing import TYPE_CHECKING, Any

import torch

from chromatica.contracts import BATCH_DIMENSIONS
from chromatica.datasets.transform import validate_channels
from chromatica.runtime import select_device

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

type Batch = tuple[torch.Tensor, torch.Tensor, Any]
type PredictionBatch = tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any]


def iter_predictions(
    loader: Iterable[Batch],
    model: torch.nn.Module,
    *,
    device: str | torch.device = "cpu",
    use_amp: bool = False,
    max_batches: int | None = None,
) -> Iterator[PredictionBatch]:
    """Yield CPU (L, target ab, predicted ab, labels) without caching outputs.

    Input and output ab use the model's normalized Lab encoding. Predictions
    must match the target shape exactly; silent resizing is not performed.
    The previous model training mode is restored when the stream closes.
    """
    selected = select_device(device)
    if use_amp and selected.type != "cuda":
        msg = (
            "Mixed precision is supported only for an explicitly selected CUDA device."
        )
        raise ValueError(msg)
    if max_batches is not None and max_batches < 1:
        msg = "max_batches must be positive or omitted."
        raise ValueError(msg)
    batches = loader if max_batches is None else islice(loader, max_batches)
    was_training = model.training
    model.to(selected).eval()
    try:
        for l_channel, ab, labels in batches:
            validate_channels(l_channel, 1)
            validate_channels(ab, 2)
            if (
                l_channel.ndim != BATCH_DIMENSIONS
                or ab.ndim != BATCH_DIMENSIONS
                or l_channel.shape[0] != ab.shape[0]
                or l_channel.shape[-2:] != ab.shape[-2:]
            ):
                msg = "Loader must yield matching batched L and ab tensors."
                raise ValueError(msg)
            with torch.inference_mode():
                with torch.amp.autocast("cuda") if use_amp else nullcontext():
                    pred = model(l_channel.to(selected, dtype=torch.float32))
                if pred.shape != ab.shape:
                    msg = (
                        f"Prediction shape {tuple(pred.shape)} differs from "
                        f"target {tuple(ab.shape)}."
                    )
                    raise ValueError(msg)
                validate_channels(pred, 2)
                cpu_pred = pred.detach().to("cpu", dtype=torch.float32)
            yield (
                l_channel.detach().cpu().float(),
                ab.detach().cpu().float(),
                cpu_pred,
                labels,
            )
    finally:
        model.train(was_training)
