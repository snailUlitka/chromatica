"""Descriptive ab palettes with portable CPU sampling and bounded memory."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from chromatica.contracts import AB_CHANNELS, PAIR_DIMENSIONS
from chromatica.metrics.hue import sampled_pairs

if TYPE_CHECKING:
    from collections.abc import Iterable

    from chromatica.datasets.transform import ABEncoding
    from chromatica.evaluation.predictions import Batch


@dataclass(frozen=True)
class TopColors:
    """CPU palette centers in physical Lab and counts in its sampled population."""

    centers_ab: torch.Tensor
    counts: torch.Tensor
    percents: torch.Tensor
    total_count: int
    eligible_count: int
    method: str

    @property
    def remainder_percent(self) -> float:
        """Return mass outside the selected palette with rounding protection."""
        return max(0.0, 100.0 - self.percents.sum().item())


def _validate_options(topk: int, sample_fraction: float, chroma_floor: float) -> None:
    if (
        topk < 1
        or not 0 < sample_fraction <= 1
        or not math.isfinite(chroma_floor)
        or chroma_floor < 0
    ):
        msg = "Require topk > 0, sample_fraction in (0, 1], and chroma_floor >= 0."
        raise ValueError(msg)


def _palette(  # noqa: PLR0913
    centers: torch.Tensor,
    counts: torch.Tensor,
    *,
    total: int,
    eligible: int,
    topk: int,
    method: str,
) -> TopColors:
    order = torch.argsort(counts, descending=True, stable=True)[:topk]
    centers, counts = centers[order].float().cpu(), counts[order].long().cpu()
    return TopColors(
        centers, counts, counts.float() * 100.0 / total, total, eligible, method
    )


def compute_top_colors_modes(  # noqa: PLR0913
    loader: Iterable[Batch],
    /,
    topk: int = 16,
    *,
    sample_fraction: float = 1.0,
    rng: torch.Generator | None = None,
    ab_minmax: tuple[float, float] = (-128.0, 128.0),
    bin_size: float = 5.0,
    merge_radius: float | None = 7.5,
    candidate_factor: int = 8,
    encoding: ABEncoding = "normalized",
    chroma_floor: float = 8.0,
) -> TopColors:
    """Aggregate physical ab grid modes, optionally merging nearby candidates.

    Out-of-range colors are rejected rather than clamped into boundary bins.
    Percentages describe eligible sampled pixels, not unsampled images. With
    full sampling the grid result is independent of loader batch size.
    """
    _validate_options(topk, sample_fraction, chroma_floor)
    lo, hi = ab_minmax
    if not (
        math.isfinite(lo)
        and math.isfinite(hi)
        and lo < hi
        and math.isfinite(bin_size)
        and bin_size > 0
    ):
        msg = "Use a finite increasing ab range and positive bin_size."
        raise ValueError(msg)
    if candidate_factor < 1 or (
        merge_radius is not None
        and (not math.isfinite(merge_radius) or merge_radius < 0)
    ):
        msg = "candidate_factor must be positive and merge_radius nonnegative."
        raise ValueError(msg)
    rng = rng or torch.Generator().manual_seed(0)
    width = math.ceil((hi - lo) / bin_size)
    counts = torch.zeros(width * width, dtype=torch.int64)
    sums = torch.zeros(width * width, 2, dtype=torch.float64)
    total = 0
    for _, ab, _ in loader:
        pairs = sampled_pairs(
            ab, encoding=encoding, sample_fraction=sample_fraction, rng=rng
        )
        keep = (torch.linalg.vector_norm(pairs, dim=1) >= chroma_floor) & (
            (pairs >= lo) & (pairs <= hi)
        ).all(1)
        pairs = pairs[keep]
        indices = ((pairs - lo) / bin_size).floor().long().clamp(0, width - 1)
        linear = indices[:, 0] * width + indices[:, 1]
        counts += torch.bincount(linear, minlength=width * width)
        sums.index_add_(0, linear, pairs)
        total += pairs.shape[0]
    if total == 0:
        msg = "No eligible palette pixels found after chroma and range filtering."
        raise ValueError(msg)
    occupied = counts > 0
    centers, counts = sums[occupied] / counts[occupied, None], counts[occupied]
    order = torch.argsort(counts, descending=True, stable=True)[
        : topk * candidate_factor
    ]
    centers, counts = centers[order], counts[order]
    if merge_radius is not None:
        alive = torch.ones(counts.shape[0], dtype=torch.bool)
        merged_centers, merged_counts = [], []
        for index in range(counts.shape[0]):
            if not alive[index]:
                continue
            near = (
                torch.linalg.vector_norm(centers - centers[index], dim=1)
                <= merge_radius
            ) & alive
            weights = counts[near]
            merged_centers.append(
                (centers[near] * weights[:, None]).sum(0) / weights.sum()
            )
            merged_counts.append(weights.sum())
            alive[near] = False
        centers, counts = torch.stack(merged_centers), torch.stack(merged_counts)
    return _palette(
        centers, counts, total=total, eligible=total, topk=topk, method="binned_modes"
    )


def compute_top_colors_kmeans(  # noqa: C901, PLR0912, PLR0913, PLR0915
    loader: Iterable[Batch],
    /,
    topk: int = 16,
    *,
    sample_fraction: float = 1.0,
    rng: torch.Generator | None = None,
    encoding: ABEncoding = "normalized",
    chroma_floor: float = 8.0,
    max_points: int = 100_000,
    ab_clip: tuple[float, float] | None = None,
    n_iter: int = 25,
    tol: float = 1e-3,
    init_centers_ab: torch.Tensor | None = None,
) -> TopColors:
    """Fit CPU k-means to a bounded uniform sample of physical ab pairs.

    A random-priority reservoir bounds memory without sampling duplicate
    pixels. Percentages describe this fitting sample. Constant colors and
    fewer distinct colors than topk return only occupied clusters.
    """
    _validate_options(topk, sample_fraction, chroma_floor)
    if max_points < 1 or n_iter < 1 or not math.isfinite(tol) or tol < 0:
        msg = "max_points/n_iter must be positive and tol finite and nonnegative."
        raise ValueError(msg)
    if ab_clip is not None and not (
        math.isfinite(ab_clip[0])
        and math.isfinite(ab_clip[1])
        and ab_clip[0] < ab_clip[1]
    ):
        msg = "ab_clip must be a finite increasing range."
        raise ValueError(msg)
    rng = rng or torch.Generator().manual_seed(0)
    reservoir = torch.empty(0, 2, dtype=torch.float64)
    priorities = torch.empty(0)
    eligible = 0
    for _, ab, _ in loader:
        points = sampled_pairs(
            ab, encoding=encoding, sample_fraction=sample_fraction, rng=rng
        )
        keep = torch.linalg.vector_norm(points, dim=1) >= chroma_floor
        if ab_clip is not None:
            keep &= ((points >= ab_clip[0]) & (points <= ab_clip[1])).all(1)
        points = points[keep]
        eligible += points.shape[0]
        reservoir = torch.cat((reservoir, points))
        priorities = torch.cat((priorities, torch.rand(points.shape[0], generator=rng)))
        if reservoir.shape[0] > max_points:
            selected = torch.topk(
                priorities, max_points, largest=False, sorted=True
            ).indices
            reservoir, priorities = reservoir[selected], priorities[selected]
    if eligible == 0:
        msg = "No eligible pixels found for k-means."
        raise ValueError(msg)
    unique = torch.unique(reservoir, dim=0)
    k = min(topk, unique.shape[0])
    if init_centers_ab is not None:
        initial = init_centers_ab.detach().to("cpu", dtype=torch.float64)
        if (
            initial.ndim != PAIR_DIMENSIONS
            or initial.shape[1] != AB_CHANNELS
            or not torch.isfinite(initial).all()
        ):
            msg = "Initial centers must be finite physical Lab pairs."
            raise ValueError(msg)
        centers_list = list(torch.unique(initial, dim=0)[:k].unbind())
    else:
        centers_list = []
    if not centers_list:
        centers_list.append(
            reservoir[
                int(torch.randint(reservoir.shape[0], (1,), generator=rng).item())
            ]
        )
    while len(centers_list) < k:
        distances = torch.cdist(reservoir, torch.stack(centers_list)).square().amin(1)
        if distances.sum().item() <= 0:
            break
        index = int(
            torch.multinomial(distances / distances.sum(), 1, generator=rng).item()
        )
        centers_list.append(reservoir[index])
    centers = torch.stack(centers_list)
    for _ in range(n_iter):
        labels = torch.cdist(reservoir, centers).argmin(1)
        new_centers = []
        for index in range(centers.shape[0]):
            cluster = reservoir[labels == index]
            if cluster.shape[0]:
                new_centers.append(cluster.mean(0))
        new = torch.stack(new_centers)
        converged = (
            new.shape == centers.shape
            and torch.linalg.vector_norm(new - centers).item() <= tol
        )
        centers = new
        if converged:
            break
    labels = torch.cdist(reservoir, centers).argmin(1)
    counts = torch.bincount(labels, minlength=centers.shape[0])
    occupied = counts > 0
    return _palette(
        centers[occupied],
        counts[occupied],
        total=reservoir.shape[0],
        eligible=eligible,
        topk=topk,
        method="kmeans_sample",
    )
