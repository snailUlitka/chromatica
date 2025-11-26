"""Utilities for measuring color diversity metrics."""

from __future__ import annotations

import logging
import math
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, Literal

import torch
import torch.nn.functional as fn

if TYPE_CHECKING:
    from collections.abc import Iterable

TWO_PI = 2.0 * math.pi
logger = logging.getLogger(__name__)


@torch.inference_mode()
def compute_delta_a_stats(  # noqa: PLR0913, PLR0912, PLR0915, C901
    dataloader: Iterable[tuple[Any, torch.Tensor, Any]],
    model: torch.nn.Module,
    *,
    device: torch.device | str | None = None,
    use_amp: bool | None = None,
    resize_mode: Literal["bilinear", "nearest", "bicubic", "area"] = "bilinear",
    clamp_ab: bool = False,
    ab_range: tuple[float, float] = (-110.0, 110.0),
    # if your a,b channels are normalized to [-1, 1], set gt_scale=pred_scale=110.0
    gt_scale: float = 1.0,
    pred_scale: float = 1.0,
    # histogram median bins
    bins: int = 1024,
    a_range: tuple[float, float] = (-110.0, 110.0),
    # filter out low-chroma pixels if needed
    min_chroma: float = 0.0,
    chroma_source: Literal["gt", "pred"] = "gt",
    max_batches: int | None = None,
    progress: bool = False,
) -> dict[str, float]:
    """Compute Δa* mean and median across a dataset.

    Returns a dictionary with the arithmetic mean and a histogram-based median.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if use_amp is None:
        use_amp = torch.cuda.is_available()

    model = model.to(device).eval()

    total_sum = 0.0
    total_count = 0

    a_min, a_max = a_range
    # MPS backend does not support float64; use float32 on MPS
    use_float32 = isinstance(device, torch.device) and device.type == "mps"
    hist_dtype = torch.float32 if use_float32 else torch.float64
    hist_edges = torch.linspace(a_min, a_max, bins + 1, device=device, dtype=hist_dtype)
    hist_counts = torch.zeros(bins, dtype=hist_dtype, device=device)

    for batch_idx, batch in enumerate(dataloader, start=1):
        if max_batches is not None and batch_idx > max_batches:
            break

        l_channel, ab_gt, _ = batch  # l_channel: (B,1,H,W), ab_gt: (B,2,H,W)
        l_channel = l_channel.to(device, non_blocking=True).float()
        ab_gt = ab_gt.to(device, non_blocking=True).float() * gt_scale

        # Use the new torch.amp.autocast API on CUDA; otherwise no autocast
        amp_ctx = (
            torch.amp.autocast("cuda", enabled=bool(use_amp))
            if isinstance(device, torch.device) and device.type == "cuda"
            else nullcontext()
        )
        with amp_ctx:
            ab_pred = model(l_channel)
        ab_pred = ab_pred.float() * pred_scale

        # Resize predictions to match GT if needed
        if ab_pred.shape[-2:] != ab_gt.shape[-2:]:
            ab_pred = fn.interpolate(
                ab_pred,
                size=ab_gt.shape[-2:],
                mode=resize_mode,
                align_corners=False if resize_mode in ("bilinear", "bicubic") else None,
            )

        if clamp_ab:
            lo, hi = ab_range
            ab_pred = torch.clamp(ab_pred, lo, hi)

        a_gt = ab_gt[:, 0, ...]
        a_pred = ab_pred[:, 0, ...]
        delta_a = a_pred - a_gt  # (B,H,W)

        if min_chroma > 0.0:
            c_src = ab_gt if chroma_source == "gt" else ab_pred
            chroma = torch.linalg.norm(c_src, dim=1)  # (B,H,W)
            mask = chroma >= min_chroma
            if not mask.any():
                if progress:
                    logger.info(
                        "[%s] all pixels masked (min_chroma=%s)", batch_idx, min_chroma
                    )
                continue
            delta_a = delta_a[mask]

        # Exact mean
        # Avoid float64 on devices like MPS; accumulate on host as Python float
        total_sum += delta_a.sum().item()
        total_count += delta_a.numel()

        # histogram-based median
        if delta_a.numel():
            idx = torch.bucketize(delta_a.reshape(-1), hist_edges, right=False) - 1
            valid = (idx >= 0) & (idx < bins)
            if valid.any():
                bincount = torch.bincount(idx[valid], minlength=bins).to(
                    hist_counts.dtype
                )
                hist_counts[: bincount.numel()] += bincount

        if progress and (batch_idx % 10 == 0):
            logger.info(
                "[%s] mean Δa*: %.4f", batch_idx, total_sum / max(total_count, 1)
            )

    mean_delta_a = total_sum / max(total_count, 1)

    # Median derived from the histogram
    cdf = torch.cumsum(hist_counts, dim=0)
    if cdf[-1] > 0:
        half = cdf[-1] * 0.5
        median_bin = torch.searchsorted(cdf, half).clamp(max=bins - 1)
        bin_centers = 0.5 * (hist_edges[:-1] + hist_edges[1:])
        median_delta_a = bin_centers[median_bin].item()
    else:
        median_delta_a = float("nan")

    return {"mean": float(mean_delta_a), "median": float(median_delta_a)}


@torch.inference_mode()
def compute_delta_a_weighted(  # noqa: PLR0913, C901
    dataloader: Iterable[tuple[Any, torch.Tensor, Any]],
    model: torch.nn.Module,
    *,
    device: torch.device | str | None = None,
    use_amp: bool | None = None,
    resize_mode: Literal["bilinear", "nearest", "bicubic", "area"] = "bilinear",
    clamp_ab: bool = False,
    ab_range: tuple[float, float] = (-110.0, 110.0),
    gt_scale: float = 1.0,
    pred_scale: float = 1.0,
    weight_source: Literal["gt", "pred", "avg"] = "gt",
    min_chroma: float = 0.0,
    eps: float = 1e-8,
    max_batches: int | None = None,
    progress: bool = False,
) -> float:
    """Compute weighted Δa* using chroma as weights."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if use_amp is None:
        use_amp = torch.cuda.is_available()

    model = model.to(device).eval()

    num = 0.0  # Σ Δa* * w (accumulated as Python float)
    den = 0.0  # Σ w       (accumulated as Python float)

    for batch_idx, batch in enumerate(dataloader, start=1):
        if max_batches is not None and batch_idx > max_batches:
            break

        l_channel, ab_gt, _ = batch
        l_channel = l_channel.to(device, non_blocking=True).float()
        ab_gt = ab_gt.to(device, non_blocking=True).float() * gt_scale

        amp_ctx = (
            torch.amp.autocast("cuda", enabled=bool(use_amp))
            if isinstance(device, torch.device) and device.type == "cuda"
            else nullcontext()
        )
        with amp_ctx:
            ab_pred = model(l_channel)
        ab_pred = ab_pred.float() * pred_scale

        if ab_pred.shape[-2:] != ab_gt.shape[-2:]:
            ab_pred = fn.interpolate(
                ab_pred,
                size=ab_gt.shape[-2:],
                mode=resize_mode,
                align_corners=False if resize_mode in ("bilinear", "bicubic") else None,
            )

        if clamp_ab:
            lo, hi = ab_range
            ab_pred = torch.clamp(ab_pred, lo, hi)

        a_gt = ab_gt[:, 0, ...]
        a_pred = ab_pred[:, 0, ...]
        delta_a = a_pred - a_gt  # (B,H,W)

        c_gt = torch.linalg.norm(ab_gt, dim=1)  # (B,H,W)
        c_pred = torch.linalg.norm(ab_pred, dim=1)

        if weight_source == "gt":
            w = c_gt
        elif weight_source == "pred":
            w = c_pred
        elif weight_source == "avg":
            w = 0.5 * (c_gt + c_pred)
        else:
            message = "weight_source must be 'gt' | 'pred' | 'avg'"
            raise ValueError(message)

        if min_chroma > 0.0:
            w = torch.where(w >= min_chroma, w, torch.zeros_like(w))

        # Keep computations in float32 on device (MPS doesn't support float64)
        num += (delta_a * w).sum().item()
        den += w.sum().item()

        if progress and (batch_idx % 10 == 0):
            logger.info("[%s] Δa*_w: %.4f", batch_idx, num / max(den, eps))

    return float(num / max(den, eps))


def _sector_indices(nbins: int, start_rad: float, extent_rad: float) -> torch.Tensor:
    """Return bin indices for the circular segment [start, start + extent)."""
    start = start_rad % TWO_PI
    extent = max(0.0, float(extent_rad))
    if extent == 0.0:
        return torch.empty(0, dtype=torch.long)
    width_bins = max(1, round(extent / TWO_PI * nbins))
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
    nbins = hist.numel()
    total = float(hist.sum().item())
    if total == 0.0:
        return float("nan")

    w = math.radians(red_half_width_deg)
    # single sector crossing zero: start at (2π - w), span 2w
    idx = _sector_indices(nbins, TWO_PI - w, 2.0 * w)
    red_sum = float(hist[idx].sum().item())
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

    left_vals = hist[idx_left]
    right_vals = hist[idx_right]

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
    """Compute Red Asymmetry Index (RAI) on a hue histogram."""
    if hist.ndim != 1:
        message = "`hist` must be 1-D"
        raise ValueError(message)
    nbins = hist.numel()
    if nbins == 0 or hist.sum().item() == 0:
        return float("nan")

    two_pi = 2.0 * math.pi
    w = math.radians(red_half_width_deg)
    half_bins = max(1, round(nbins * w / two_pi))

    # wrap-around: left half (-θ..0) — last half_bins bins,
    # right half (0..+θ) — first half_bins bins
    left = float(hist[nbins - half_bins :].sum().item())
    right = float(hist[:half_bins].sum().item())

    return (right - left) / (right + left + eps)
