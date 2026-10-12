"""Hue distributions in physical CIE-Lab with correct channel pairing."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from chromatica.contracts import AB_CHANNELS, PAIR_DIMENSIONS
from chromatica.datasets.transform import ABEncoding, ab_to_pairs, decode_ab

if TYPE_CHECKING:
    from collections.abc import Iterable

    from chromatica.evaluation.predictions import Batch

TWO_PI = 2.0 * math.pi


def sampled_pairs(
    ab: torch.Tensor,
    *,
    encoding: ABEncoding,
    sample_fraction: float,
    rng: torch.Generator | None,
) -> torch.Tensor:
    """Decode and optionally Bernoulli-sample CPU ab pairs without replacement."""
    if not 0 < sample_fraction <= 1:
        msg = "sample_fraction must be in (0, 1]."
        raise ValueError(msg)
    if rng is not None and rng.device.type != "cpu":
        msg = "Sampling uses a CPU generator for portable CPU/CUDA/MPS inputs."
        raise ValueError(msg)
    pairs = ab_to_pairs(
        decode_ab(ab.detach().to("cpu", dtype=torch.float64), encoding=encoding)
    )
    if sample_fraction < 1:
        mask = torch.rand(pairs.shape[0], generator=rng) < sample_fraction
        pairs = pairs[mask]
    return pairs


def hue_counts(
    pairs: torch.Tensor, *, nbins: int, min_chroma: float = 0.0
) -> torch.Tensor:
    """Count physical Lab pairs; neutral pixels have undefined hue and are excluded."""
    if nbins < 1 or not math.isfinite(min_chroma) or min_chroma < 0:
        msg = "nbins must be positive and min_chroma finite and nonnegative."
        raise ValueError(msg)
    if (
        pairs.ndim != PAIR_DIMENSIONS
        or pairs.shape[1] != AB_CHANNELS
        or not torch.isfinite(pairs).all()
    ):
        msg = "Expected finite physical Lab pairs of shape (pixels, 2)."
        raise ValueError(msg)
    pairs = pairs.detach().to("cpu", dtype=torch.float64)
    chroma = torch.linalg.vector_norm(pairs, dim=1)
    keep = (chroma > 0) & (chroma >= min_chroma)
    hue = torch.atan2(pairs[keep, 1], pairs[keep, 0]).remainder(TWO_PI)
    indices = (hue * nbins / TWO_PI).floor().long().clamp(0, nbins - 1)
    return torch.bincount(indices, minlength=nbins)


def compute_histogram(  # noqa: PLR0913
    loader: Iterable[Batch],
    /,
    nbins: int = 360,
    *,
    sample_fraction: float = 1.0,
    rng: torch.Generator | None = None,
    encoding: ABEncoding = "normalized",
    min_chroma: float = 0.0,
) -> torch.Tensor:
    """Stream a hue histogram as CPU int64 counts without batch-size weighting.

    With sample_fraction=1, every eligible pixel is counted and partitioning
    into batches does not change the result. Empty loaders raise ValueError;
    a nonempty dataset with no chromatic pixels returns zero counts.
    """
    if nbins < 1 or not 0 < sample_fraction <= 1:
        msg = "nbins must be positive and sample_fraction in (0, 1]."
        raise ValueError(msg)
    histogram = torch.zeros(nbins, dtype=torch.int64)
    pixels = 0
    for _, ab, _ in loader:
        pixels += ab.numel() // 2
        pairs = sampled_pairs(
            ab, encoding=encoding, sample_fraction=sample_fraction, rng=rng
        )
        histogram += hue_counts(pairs, nbins=nbins, min_chroma=min_chroma)
    if pixels == 0:
        msg = "No pixels found in loader."
        raise ValueError(msg)
    return histogram
