"""Signed color bias and descriptive hue-sector statistics.

Positive delta-a means a shift toward positive a* (the red/magenta direction),
not improved color diversity. Report magnitude errors separately.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal

import torch

from chromatica.datasets.transform import ABEncoding, ab_to_pairs, decode_ab
from chromatica.evaluation.predictions import Batch, iter_predictions
from chromatica.metrics.bias import BiasAccumulator

if TYPE_CHECKING:
    from collections.abc import Iterable

TWO_PI = 2.0 * math.pi
MAX_HALF_WIDTH_DEG = 180.0


def compute_delta_a_stats(  # noqa: PLR0913
    dataloader: Iterable[Batch],
    model: torch.nn.Module,
    *,
    device: str | torch.device = "cpu",
    use_amp: bool = False,
    encoding: ABEncoding = "normalized",
    bins: int = 4096,
    a_range: tuple[float, float] = (-255.0, 255.0),
    min_chroma: float = 0.0,
    chroma_source: Literal["gt", "pred"] = "gt",
    max_batches: int | None = None,
) -> dict[str, float | int]:
    """Return signed delta-a mean/median and magnitude errors in physical Lab.

    No multiplicative 110/128 scale is accepted: normalized inputs must be
    decoded with both scale and offset. Histogram outliers are reported.
    """
    if (
        not math.isfinite(min_chroma)
        or min_chroma < 0
        or chroma_source not in {"gt", "pred"}
    ):
        msg = "Invalid chroma threshold or source."
        raise ValueError(msg)
    stats = BiasAccumulator(bins=bins, a_range=a_range)
    for _, target, prediction, _ in iter_predictions(
        dataloader, model, device=device, use_amp=use_amp, max_batches=max_batches
    ):
        gt = ab_to_pairs(decode_ab(target.double(), encoding=encoding))
        pred = ab_to_pairs(decode_ab(prediction.double(), encoding=encoding))
        chroma = torch.linalg.vector_norm(gt if chroma_source == "gt" else pred, dim=1)
        keep = chroma >= min_chroma
        stats.update_pairs(pred[keep], gt[keep])
    summary = stats.summary()
    return {
        "mean": summary["mean_delta_a"],
        "median": summary["median_delta_a"],
        "mae": summary["mae_a"],
        "rmse": summary["rmse_a"],
        "pixels": summary["pixels"],
        "underflow": summary["delta_a_underflow"],
        "overflow": summary["delta_a_overflow"],
    }


def compute_delta_a_weighted(  # noqa: PLR0913
    dataloader: Iterable[Batch],
    model: torch.nn.Module,
    *,
    device: str | torch.device = "cpu",
    use_amp: bool = False,
    encoding: ABEncoding = "normalized",
    weight_source: Literal["gt", "pred", "avg"] = "gt",
    min_chroma: float = 0.0,
    max_batches: int | None = None,
) -> float:
    """Return chroma-weighted signed delta-a; absent weight mass yields NaN."""
    if (
        weight_source not in {"gt", "pred", "avg"}
        or not math.isfinite(min_chroma)
        or min_chroma < 0
    ):
        msg = "Invalid chroma threshold or weight source."
        raise ValueError(msg)
    numerator, denominator = 0.0, 0.0
    for _, target, prediction, _ in iter_predictions(
        dataloader, model, device=device, use_amp=use_amp, max_batches=max_batches
    ):
        gt = ab_to_pairs(decode_ab(target.double(), encoding=encoding))
        pred = ab_to_pairs(decode_ab(prediction.double(), encoding=encoding))
        c_gt = torch.linalg.vector_norm(gt, dim=1)
        c_pred = torch.linalg.vector_norm(pred, dim=1)
        weights = (
            c_gt
            if weight_source == "gt"
            else c_pred
            if weight_source == "pred"
            else (c_gt + c_pred) / 2
        )
        weights = torch.where(weights >= min_chroma, weights, 0.0)
        numerator += ((pred[:, 0] - gt[:, 0]) * weights).sum().item()
        denominator += weights.sum().item()
    return numerator / denominator if denominator > 0 else float("nan")


def _validate_histogram(hist: torch.Tensor, half_width: float) -> None:
    """Reject nonfinite/negative counts and invalid overlapping sectors."""
    if hist.ndim != 1 or not torch.isfinite(hist).all() or (hist < 0).any():
        msg = "Hue counts must be a finite nonnegative one-dimensional tensor."
        raise ValueError(msg)
    if not math.isfinite(half_width) or not 0 < half_width <= MAX_HALF_WIDTH_DEG:
        msg = "Sector half-width must be in (0, 180] degrees."
        raise ValueError(msg)


def _sector_indices(nbins: int, start_rad: float, extent_rad: float) -> torch.Tensor:
    """Return bin indices for the circular segment [start, start + extent)."""
    start = start_rad % TWO_PI
    extent = max(0.0, float(extent_rad))
    if extent == 0.0:
        return torch.empty(0, dtype=torch.long)
    width_bins = min(nbins, max(1, round(extent / TWO_PI * nbins)))
    start_bin = math.floor(start / TWO_PI * nbins)  # 0..nbins-1
    return (torch.arange(width_bins, dtype=torch.long) + start_bin) % nbins


def _threshold_mask(
    values: torch.Tensor,
    *,
    threshold: float = 0.0,
    kind: Literal["absolute", "relative_max", "relative_total"] = "absolute",
) -> torch.Tensor:
    """Build a boolean mask of bins considered significant.

    - absolute: count > threshold
    - relative_max: count > threshold * max(count)
    - relative_total: count > threshold * sum(count)
    """
    if values.numel() == 0:
        return torch.zeros(0, dtype=torch.bool)
    if kind == "absolute":
        thr = threshold
    elif kind == "relative_max":
        thr = float(values.max().item()) * threshold
    elif kind == "relative_total":
        thr = float(values.sum().item()) * threshold
    else:
        message = "kind must be 'absolute' | 'relative_max' | 'relative_total'"
        raise ValueError(message)
    return values > thr


def red_sector_fraction(
    hist: torch.Tensor,
    *,
    red_half_width_deg: float = 30.0,
) -> float:
    """Compute the mass inside the red wedge around zero hue.

    Expects a 1-D histogram tensor. Returns a fraction in [0, 1].
    """
    if hist.ndim != 1:
        message = "`hist` must be 1-D"
        raise ValueError(message)
    _validate_histogram(hist, red_half_width_deg)
    nbins = hist.numel()
    total = float(hist.sum().item())
    if total == 0.0:
        return float("nan")

    w = math.radians(red_half_width_deg)
    # single sector crossing zero: start at (2π - w), span 2w
    idx = _sector_indices(nbins, TWO_PI - w, 2.0 * w)
    red_sum = float(hist[idx.to(hist.device)].sum().item())
    return red_sum / total


def red_left_right_stats(
    hist: torch.Tensor,
    *,
    red_half_width_deg: float = 30.0,
    bin_threshold: float = 0.0,
    threshold_kind: Literal["absolute", "relative_max", "relative_total"] = "absolute",
) -> dict[str, float]:
    """Compare the left and right halves of the red wedge around zero."""
    if hist.ndim != 1:
        message = "`hist` must be 1-D"
        raise ValueError(message)
    _validate_histogram(hist, red_half_width_deg)
    nbins = hist.numel()
    total = float(hist.sum().item())
    if total == 0.0:
        return {
            k: float("nan")
            for k in [
                "left_sum",
                "right_sum",
                "left_frac",
                "right_frac",
                "left_bins",
                "right_bins",
                "bins_ratio",
                "frac_diff",
                "frac_ratio",
            ]
        }

    w = math.radians(red_half_width_deg)

    # left half (-w, 0) => start at (2π - w), width w
    idx_left = _sector_indices(nbins, TWO_PI - w, w)
    # right half (0, +w)
    idx_right = _sector_indices(nbins, 0.0, w)

    left_vals = hist[idx_left.to(hist.device)]
    right_vals = hist[idx_right.to(hist.device)]

    left_sum = float(left_vals.sum().item())
    right_sum = float(right_vals.sum().item())

    left_frac = left_sum / total
    right_frac = right_sum / total

    # Count bins above the configured threshold
    mask_left = _threshold_mask(left_vals, threshold=bin_threshold, kind=threshold_kind)
    mask_right = _threshold_mask(
        right_vals, threshold=bin_threshold, kind=threshold_kind
    )
    left_bins = int(mask_left.sum().item())
    right_bins = int(mask_right.sum().item())

    eps = 1e-12
    return {
        "left_sum": left_sum,
        "right_sum": right_sum,
        "left_frac": left_frac,
        "right_frac": right_frac,
        "left_bins": float(left_bins),
        "right_bins": float(right_bins),
        "bins_ratio": float(left_bins / max(right_bins, 1)),
        "frac_diff": float(right_frac - left_frac),  # >0 means the right half dominates
        "frac_ratio": float(right_frac / max(left_frac, eps)),
    }


def compare_red_fraction(
    hist_pred: torch.Tensor,
    hist_gt: torch.Tensor,
    *,
    red_half_width_deg: float = 30.0,
) -> dict[str, float]:
    """Compare red mass fraction between prediction and ground truth."""
    p_pred = red_sector_fraction(hist_pred, red_half_width_deg=red_half_width_deg)
    p_gt = red_sector_fraction(hist_gt, red_half_width_deg=red_half_width_deg)
    eps = 1e-12
    return {
        "p_red_pred": float(p_pred),
        "p_red_gt": float(p_gt),
        "delta": float(p_pred - p_gt),
        "ratio": float(p_pred / max(p_gt, eps)),
    }


def compute_rai(
    hist: torch.Tensor, red_half_width_deg: float = 30.0, eps: float = 1e-12
) -> float:
    """Compute red asymmetry; insufficient bins or absent red mass yields NaN."""
    if hist.ndim != 1:
        message = "`hist` must be 1-D"
        raise ValueError(message)
    _validate_histogram(hist, red_half_width_deg)
    nbins = hist.numel()
    if nbins < 2 or hist.sum().item() == 0:  # noqa: PLR2004
        return float("nan")

    two_pi = 2.0 * math.pi
    w = math.radians(red_half_width_deg)
    half_bins = min(nbins // 2, max(1, round(nbins * w / two_pi)))

    # wrap-around: left half (-θ..0) — last half_bins bins,
    # right half (0..+θ) — first half_bins bins
    left = float(hist[nbins - half_bins :].sum().item())
    right = float(hist[:half_bins].sum().item())
    if right + left == 0:
        return float("nan")
    return (right - left) / (right + left + eps)
