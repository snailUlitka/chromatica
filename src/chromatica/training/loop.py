"""Shared Adam/MSE training with element-weighted loss accounting."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
from itertools import islice
from typing import TYPE_CHECKING

import torch
from torch import nn

from chromatica.evaluation.predictions import Batch, iter_predictions

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from chromatica.config import TrainingConfig


@dataclass(frozen=True)
class EpochMetrics:
    """One epoch of normalized-ab MSE; absent validation is explicit."""

    epoch: int
    train_mse: float
    val_mse: float | None

    def as_dict(self) -> dict[str, float | int | None]:
        """Return primitive epoch values for local or ClearML logging."""
        return asdict(self)


def mean_loss(
    loader: Iterable[Batch],
    model: nn.Module,
    *,
    device: torch.device,
    use_amp: bool = False,
    max_batches: int | None = None,
) -> float:
    """Measure normalized-ab MSE over actual elements, including short batches."""
    squared_error, elements = 0.0, 0
    for _, target, prediction, _ in iter_predictions(
        loader, model, device=device, use_amp=use_amp, max_batches=max_batches
    ):
        squared_error += (prediction.double() - target.double()).square().sum().item()
        elements += target.numel()
    if elements == 0:
        msg = "No validation pixels were supplied."
        raise ValueError(msg)
    return squared_error / elements


def fit(  # noqa: PLR0913
    model: nn.Module,
    train_loader: Iterable[Batch],
    val_loader: Iterable[Batch] | None,
    config: TrainingConfig,
    *,
    device: torch.device,
    on_epoch: Callable[[EpochMetrics], None] | None = None,
) -> list[EpochMetrics]:
    """Train only when explicitly called; restore no checkpoints automatically."""
    if config.use_amp and device.type != "cuda":
        msg = "Mixed precision training is supported only on CUDA."
        raise ValueError(msg)
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    loss_fn = nn.MSELoss()
    scaler = torch.amp.GradScaler("cuda") if config.use_amp else None
    history = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        squared_error, elements = 0.0, 0
        batches = (
            train_loader
            if config.max_train_batches is None
            else islice(train_loader, config.max_train_batches)
        )
        for l_batch, ab_batch, _ in batches:
            l_channel, ab = l_batch.to(device), ab_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda") if config.use_amp else nullcontext():
                prediction = model(l_channel)
                if prediction.shape != ab.shape:
                    msg = "Training prediction and target shapes differ."
                    raise ValueError(msg)
                loss = loss_fn(prediction, ab)
            if not torch.isfinite(loss):
                msg = "Training loss is nonfinite."
                raise FloatingPointError(msg)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            squared_error += float(loss.detach().item()) * ab.numel()
            elements += ab.numel()
        if elements == 0:
            msg = "No training pixels were supplied."
            raise ValueError(msg)
        validation_mse = (
            mean_loss(
                val_loader,
                model,
                device=device,
                use_amp=config.use_amp,
                max_batches=config.max_val_batches,
            )
            if val_loader is not None
            else None
        )
        metrics = EpochMetrics(epoch, squared_error / elements, validation_mse)
        history.append(metrics)
        if on_epoch is not None:
            on_epoch(metrics)
    return history
