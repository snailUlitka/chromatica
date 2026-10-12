"""Streaming physical Lab bias and error with explicit median tail accounting."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from chromatica.contracts import AB_CHANNELS, PAIR_DIMENSIONS
from chromatica.datasets.transform import AB_SCALE, ABEncoding, ab_to_pairs, decode_ab


@dataclass
class BiasAccumulator:
    """Accumulate pixel-weighted statistics on CPU in physical a*/b* units.

    The histogram estimates median signed delta-a. Underflow and overflow are
    counted rather than dropped. If a central order statistic lies outside
    histogram support, the reported median is NaN. No prediction is clamped.
    """

    bins: int = 4096
    a_range: tuple[float, float] = (-255.0, 255.0)
    count: int = 0
    sum_delta_a: float = 0.0
    sum_abs_a: float = 0.0
    sum_sq_a: float = 0.0
    sum_abs_ab: float = 0.0
    sum_sq_ab: float = 0.0
    sum_chroma_gt: float = 0.0
    sum_chroma_pred: float = 0.0
    underflow: int = 0
    overflow: int = 0
    hist: torch.Tensor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize a CPU histogram after validating its range."""
        lo, hi = self.a_range
        if self.bins < 1 or not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
            msg = "Histogram bins must be positive; its finite range must increase."
            raise ValueError(msg)
        self.hist = torch.zeros(self.bins, dtype=torch.int64)

    def update(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        *,
        encoding: ABEncoding = "normalized",
    ) -> None:
        """Add matching channel-first images or batches to all accumulators."""
        if prediction.shape != target.shape:
            msg = "Prediction and target shapes must match."
            raise ValueError(msg)
        pred = ab_to_pairs(
            decode_ab(
                prediction.detach().to("cpu", dtype=torch.float64), encoding=encoding
            )
        )
        gt = ab_to_pairs(
            decode_ab(target.detach().to("cpu", dtype=torch.float64), encoding=encoding)
        )
        self.update_pairs(pred, gt)

    def update_pairs(self, pred: torch.Tensor, gt: torch.Tensor) -> None:
        """Add finite CPU physical Lab pairs of shape (pixels, 2)."""
        if (
            pred.shape != gt.shape
            or pred.ndim != PAIR_DIMENSIONS
            or pred.shape[1] != AB_CHANNELS
        ):
            msg = "Physical ab pairs must have matching (pixels, 2) shapes."
            raise ValueError(msg)
        pred, gt = (
            pred.detach().to("cpu", dtype=torch.float64),
            gt.detach().to("cpu", dtype=torch.float64),
        )
        if not torch.isfinite(pred).all() or not torch.isfinite(gt).all():
            msg = "Physical ab pairs must be finite."
            raise ValueError(msg)
        if pred.shape[0] == 0:
            return
        delta = pred - gt
        a = delta[:, 0]
        self.count += a.numel()
        self.sum_delta_a += a.sum().item()
        self.sum_abs_a += a.abs().sum().item()
        self.sum_sq_a += a.square().sum().item()
        self.sum_abs_ab += delta.abs().sum().item()
        self.sum_sq_ab += delta.square().sum().item()
        self.sum_chroma_gt += torch.linalg.vector_norm(gt, dim=1).sum().item()
        self.sum_chroma_pred += torch.linalg.vector_norm(pred, dim=1).sum().item()
        lo, hi = self.a_range
        self.underflow += int((a < lo).sum().item())
        self.overflow += int((a > hi).sum().item())
        indices = (
            ((a - lo) * self.bins / (hi - lo)).floor().long().clamp(0, self.bins - 1)
        )
        self.hist += torch.bincount(indices, minlength=self.bins)

    def summary(self) -> dict[str, float | int]:
        """Return errors, signed bias, approximate median, and tail counts."""
        if self.count == 0:
            msg = "No eligible pixels were supplied for evaluation."
            raise ValueError(msg)
        ranks = ((self.count - 1) // 2, self.count // 2)
        if ranks[0] < self.underflow or ranks[1] >= self.count - self.overflow:
            median = float("nan")
        else:
            cdf = self.hist.cumsum(0)
            indices = torch.searchsorted(cdf, torch.tensor(ranks) + 1).double()
            lo, hi = self.a_range
            median = lo + (indices.mean().item() + 0.5) * (hi - lo) / self.bins
        return {
            "pixels": self.count,
            "mean_delta_a": self.sum_delta_a / self.count,
            "median_delta_a": median,
            "mae_a": self.sum_abs_a / self.count,
            "rmse_a": math.sqrt(self.sum_sq_a / self.count),
            "mae_ab": self.sum_abs_ab / (2 * self.count),
            "rmse_ab": math.sqrt(self.sum_sq_ab / (2 * self.count)),
            "normalized_mse": self.sum_sq_ab / (2 * self.count * AB_SCALE**2),
            "mean_chroma_gt": self.sum_chroma_gt / self.count,
            "mean_chroma_pred": self.sum_chroma_pred / self.count,
            "delta_a_underflow": self.underflow,
            "delta_a_overflow": self.overflow,
            "median_bin_width": (self.a_range[1] - self.a_range[0]) / self.bins,
        }
