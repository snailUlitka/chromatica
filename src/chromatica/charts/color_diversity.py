"""Module with functions for color diversity draw charts."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Final

import matplotlib.pyplot as plt
import numpy as np
import torch
from pydantic import BaseModel, ConfigDict
from skimage.color import lab2rgb

if TYPE_CHECKING:
    from collections.abc import Iterable

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

TWO_PI: Final[float] = 2.0 * math.pi


def compute_histogram(
    loader: Iterable[tuple[Any, torch.Tensor, Any]],
    /,
    nbins: int = 100,
    *,
    sample_fraction: float = 0.1,
    rng: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Compute a 1-D hue histogram from a stream of (..., ab, ...) batches.

    This function estimates the distribution of hue angles in the ab plane of
    CIE-Lab while ignoring lightness L. For each pixel with chromatic components
    (a, b), it computes hue = atan2(b, a) in [0, 2pi) and adds a unit count to
    the corresponding bin. Optional sub-sampling is applied for speed. Counts
    are not normalized.

    Why it may look warm-heavy
    --------------------------
    Chroma ||ab|| is not used as a weight, so near-neutral pixels contribute
    the same as saturated ones. Datasets with many weakly saturated browns or
    grays can therefore produce tall bars around those hues.

    Scale invariance
    ----------------
    If both a and b are scaled by the same factor k, the histogram is unchanged
    because atan2(k*b, k*a) = atan2(b, a). Non-uniform or non-linear scalings
    across a and b will distort angles.

    Parameters
    ----------
    loader : iterable of tuples
        Yields (..., ab, ...) where ab has shape (B, 2, H, W).
    nbins : int
        Number of equal-width bins spanning [0, 2pi).
    sample_fraction : float
        Fraction of pixels per batch to sample, in (0, 1].
    rng : torch.Generator or None
        Generator for reproducible sampling.

    Returns
    -------
    torch.Tensor
        Shape (nbins,), raw counts on the same device as the input `ab`.

    Notes
    -----
    The computation streams over batches and uses vectorized torch operations,
    which is suitable for large datasets.
    """
    if not (0.0 < sample_fraction <= 1.0):
        msg = "`sample_fraction` must be in (0, 1]"
        raise ValueError(msg)

    hist: torch.Tensor | None = None

    with torch.no_grad():
        for *_, ab, _ in loader:
            if hist is None:
                hist = torch.zeros(nbins, dtype=torch.int64, device=ab.device)

            ab_flat = ab.reshape(2, -1).T

            if sample_fraction < 1.0:
                total = ab_flat.shape[0]
                keep = max(1, int(total * sample_fraction))
                indices = torch.randint(
                    total, (keep,), generator=rng, device=ab_flat.device
                )
                ab_flat = ab_flat[indices]

            a_vals, b_vals = ab_flat.T
            hue = torch.atan2(b_vals, a_vals)
            hue = torch.remainder(hue + TWO_PI, TWO_PI)

            bin_idx = torch.div(hue, TWO_PI / nbins, rounding_mode="floor").to(
                torch.int64
            )
            batch_hist = torch.bincount(bin_idx, minlength=nbins)

            hist += batch_hist

    if hist is None:
        msg = "No data found in loader."
        raise RuntimeError(msg)

    return hist


def plot_histogram(
    hist: torch.Tensor,
    /,
    *,
    l_value: float = 50.0,
    title: str | None = "Hue-based color histogram",
) -> tuple[Figure, Axes]:
    """
    Render a hue histogram as a colored bar plot and show it.

    Parameters
    ----------
    hist : Tensor
        1-D tensor of bin counts.
    l_value : float
        Lightness value for Lab -> RGB mapping.
    title : str
        Figure title, or None for no title.

    Returns
    -------
    tuple[Figure, Axes]
    """
    if hist.ndim != 1:
        msg = "`hist` must be a 1-D tensor."
        raise ValueError(msg)

    nbins = hist.numel()
    data = hist.to(dtype=torch.float32, device="cpu").numpy()
    if data.max() > 0:
        data /= data.max()

    centers = (np.arange(nbins) + 0.5) * TWO_PI / nbins
    lab = np.empty((nbins, 3), dtype=np.float32)
    lab[:, 0] = l_value
    lab[:, 1] = np.cos(centers) * 50
    lab[:, 2] = np.sin(centers) * 50
    rgb = lab2rgb(lab[None])[0]

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.bar(
        np.arange(nbins),
        data,
        color=rgb,
        width=1.0,
        label="Hue distribution",
        align="edge",
    )

    _ = ax.set_xticks([]), ax.set_yticks([])
    _ = ax.set_xlim(0, nbins), ax.set_ylim(0, 1.05)

    if title:
        ax.set_title(title)

    plt.tight_layout()
    return fig, ax


class _TopColors(BaseModel):
    centers_ab: torch.Tensor
    counts: torch.Tensor
    percents: torch.Tensor
    total_count: int
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    @property
    def remainder_percent(self) -> float:
        return float(100.0 - self.percents.sum().item())


def compute_top_colors_modes(
    loader: Iterable[tuple[Any, torch.Tensor, Any]],
    /,
    topk: int = 16,
    *,
    sample_fraction: float = 0.05,
    rng: torch.Generator | None = None,
    ab_minmax: tuple[float, float] = (-110.0, 110.0),
    bin_size: float = 5.0,
    merge_radius: float | None = 7.5,
    candidate_factor: int = 8,
    input_scale: float = 1.0,
    chroma_floor: float = 8.0,
) -> _TopColors:
    """
    Estimate the top-K chromatic colors by aggregating pixels in the ab plane.

    The algorithm discards lightness L and operates on (a, b) only. Inputs can
    be rescaled to Lab units via `input_scale`. Pixels with chroma below
    `chroma_floor` (||ab|| < chroma_floor) are dropped. The remaining pixels
    are quantized onto a uniform grid over [ab_minmax] x [ab_minmax] with step
    `bin_size`. For each occupied bin, counts and mean (a, b) are accumulated.
    The most populated bins are kept as candidates and then greedily merged:
    bins whose centers lie within `merge_radius` in ab are fused using
    count-weighted centroids. The final K clusters define representative colors.
    Percentages are computed relative to the number of pixels that survived the
    chroma filter.

    Why results can differ from the hue histogram
    --------------------------------------------
    The hue histogram counts every pixel equally and spreads broad color
    families across many angles. This method keeps only sufficiently saturated
    pixels and consolidates dense regions in (a, b). Rare but concentrated
    colors (e.g., saturated blues) can enter the top-K even if they occupy a
    small share of all pixels, while broad warm tones split across many bins.

    Parameters
    ----------
    loader : iterable of tuples
        Yields (..., ab, ...) where ab has shape (B, 2, H, W).
    topk : int
        Number of colors to return after merging.
    sample_fraction : float
        Fraction of pixels per batch to sample, in (0, 1].
    rng : torch.Generator or None
        Generator for reproducible sampling.
    ab_minmax : tuple of float
        Inclusive min and max Lab values along a and b before binning.
    bin_size : float
        Grid step in Lab units for a and b.
    merge_radius : float or None
        Euclidean radius in ab for greedy merging. None disables merging.
    candidate_factor : int
        Multiplier controlling how many top bins are considered before merging.
    input_scale : float
        Factor applied to input ab prior to processing. Use 110.0 if ab is
        normalized to [-1, 1].
    chroma_floor : float
        Minimum ||ab|| in Lab units required to keep a pixel.

    Returns
    -------
    _TopColors
        centers_ab : (K, 2) float32, representative ab centers in Lab units (CPU)
        counts     : (K,) int64, pixel counts per cluster
        percents   : (K,) float32, 100 * counts / total_kept
        total_count: int, number of pixels that passed the chroma filter

    Notes
    -----
    Single-pass streaming accumulation. Memory is O(n_bins). Adjust `bin_size`,
    `merge_radius`, and `sample_fraction` to trade accuracy for speed on large
    datasets.
    """
    if not (0.0 < sample_fraction <= 1.0):
        raise ValueError("`sample_fraction` must be in (0, 1].")
    if topk < 1:
        raise ValueError("`topk` must be >= 1.")
    a_min, a_max = ab_minmax
    b_min, b_max = ab_minmax
    nb_a = int(math.ceil((a_max - a_min) / bin_size))
    nb_b = int(math.ceil((b_max - b_min) / bin_size))
    n_bins = nb_a * nb_b

    counts: torch.Tensor | None = None
    sum_a: torch.Tensor | None = None
    sum_b: torch.Tensor | None = None
    total_count = 0

    with torch.no_grad():
        for *_, ab, _ in loader:
            ab_flat = ab.reshape(2, -1).T
            if sample_fraction < 1.0:
                N = ab_flat.shape[0]
                keep = max(1, int(N * sample_fraction))
                idx = torch.randint(N, (keep,), generator=rng, device=ab_flat.device)
                ab_flat = ab_flat[idx]
            ab_flat = ab_flat * input_scale

            if chroma_floor > 0.0:
                chroma = torch.linalg.norm(ab_flat, dim=1)
                mask = chroma >= chroma_floor
                if mask.any():
                    ab_flat = ab_flat[mask]
                else:
                    continue

            if ab_flat.numel() == 0:
                continue

            a = ab_flat[:, 0]
            b = ab_flat[:, 1]
            ai = torch.clamp(
                ((a - a_min) / bin_size).floor().to(torch.int64), 0, nb_a - 1
            )
            bi = torch.clamp(
                ((b - b_min) / bin_size).floor().to(torch.int64), 0, nb_b - 1
            )
            lin = ai * nb_b + bi

            if counts is None:
                device = ab_flat.device
                counts = torch.zeros(n_bins, dtype=torch.int64, device=device)
                sum_a = torch.zeros(n_bins, dtype=torch.float32, device=device)
                sum_b = torch.zeros(n_bins, dtype=torch.float32, device=device)

            counts += torch.bincount(lin, minlength=n_bins)
            sum_a.index_add_(0, lin, a.to(torch.float32))
            sum_b.index_add_(0, lin, b.to(torch.float32))
            total_count += lin.numel()

    if counts is None or sum_a is None or sum_b is None or total_count == 0:
        raise RuntimeError("No data found in loader after chroma filtering.")

    nz = (counts > 0).nonzero(as_tuple=False).squeeze(1)
    if nz.numel() == 0:
        raise RuntimeError("All bins are empty after processing.")

    cnt = counts[nz]
    mean_a = sum_a[nz] / cnt.to(torch.float32)
    mean_b = sum_b[nz] / cnt.to(torch.float32)
    centers = torch.stack([mean_a, mean_b], dim=1)

    k_cand = min(centers.shape[0], max(topk, topk * candidate_factor))
    order = torch.argsort(cnt, descending=True)
    centers = centers[order][:k_cand]
    cnt = cnt[order][:k_cand]

    if merge_radius is not None and k_cand > 1:
        centers_cpu = centers.to(dtype=torch.float32)
        cnt_cpu = cnt.to(dtype=torch.float32)
        alive = torch.ones(k_cand, dtype=torch.bool)
        merged_centers: list[torch.Tensor] = []
        merged_counts: list[float] = []
        for i in range(k_cand):
            if not alive[i]:
                continue
            c_i = centers_cpu[i]
            d = torch.norm(centers_cpu - c_i, dim=1)
            near = (d <= merge_radius) & alive
            if near.any():
                weights = cnt_cpu[near]
                c_group = (centers_cpu[near] * weights[:, None]).sum(0) / weights.sum()
                merged_centers.append(c_group)
                merged_counts.append(float(weights.sum().item()))
                alive[near] = False
        centers = torch.stack(merged_centers, dim=0)
        cnt = torch.tensor(merged_counts, dtype=torch.float32)
    else:
        centers = centers.to(dtype=torch.float32)
        cnt = cnt.to(dtype=torch.float32)

    order = torch.argsort(cnt, descending=True)
    centers = centers[order][:topk].cpu()
    cnt = cnt[order][:topk].cpu()
    perc = (cnt / float(total_count)) * 100.0

    return _TopColors(
        centers_ab=centers.to(torch.float32),
        counts=cnt.to(torch.int64),
        percents=perc.to(torch.float32),
        total_count=total_count,
    )


def plot_top_colors_pie(
    top: _TopColors,
    /,
    *,
    l_value: float = 50.0,
    title: str | None = "Top colors",
    include_other: bool = True,
    other_label: str = "Other colors",
    min_show_other: float = 0.05,
) -> tuple[Figure, Axes]:
    K = top.centers_ab.shape[0]
    lab = np.empty((K, 3), dtype=np.float32)
    lab[:, 0] = l_value
    lab[:, 1:] = top.centers_ab.numpy()
    rgb = lab2rgb(lab[None])[0]
    sizes = top.percents.numpy().tolist()
    colors = rgb.tolist()
    labels = [""] * K
    remainder = top.remainder_percent
    if include_other and remainder >= min_show_other:
        sizes.append(remainder)
        colors.append((0.8, 0.8, 0.8))
        labels.append(other_label)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(
        sizes,
        labels=labels,
        colors=colors,
        startangle=90,
        autopct=lambda p: f"{p:.1f}%" if p >= 1.0 else "",
        pctdistance=0.7,
        wedgeprops={"linewidth": 0},
        textprops={"fontsize": 10},
    )
    if title:
        ax.set_title(title)
    ax.axis("equal")
    plt.tight_layout()
    return fig, ax


def plot_top_colors_palette(
    top: _TopColors,
    /,
    *,
    l_value: float = 50.0,
    title: str | None = "Top colors palette",
    cols: int | None = None,
    tile_px: int = 80,
    spacing_px: int = 4,
    fmt: str = "{:.1f}%",
) -> tuple[Figure, Axes]:
    K = top.centers_ab.shape[0]
    lab = np.empty((K, 3), dtype=np.float32)
    lab[:, 0] = l_value
    lab[:, 1:] = top.centers_ab.numpy()
    rgb = lab2rgb(lab[None])[0]
    if cols is None or cols < 1:
        cols = min(K, 8)
    rows = int(math.ceil(K / cols))
    H = rows * tile_px + (rows - 1) * spacing_px if rows > 0 else tile_px
    W = cols * tile_px + (cols - 1) * spacing_px if cols > 0 else tile_px
    canvas = np.ones((H, W, 3), dtype=np.float32)
    for idx in range(K):
        r = idx // cols
        c = idx % cols
        y0 = r * (tile_px + spacing_px)
        x0 = c * (tile_px + spacing_px)
        canvas[y0 : y0 + tile_px, x0 : x0 + tile_px, :] = rgb[idx]
    fig, ax = plt.subplots(figsize=(W / 80.0, H / 80.0), dpi=160)
    ax.imshow(canvas, interpolation="nearest")
    ax.set_xticks([]), ax.set_yticks([])
    ax.set_xlim(0, W), ax.set_ylim(H, 0)
    perc = top.percents.numpy()
    for idx in range(K):
        r = idx // cols
        c = idx % cols
        y0 = r * (tile_px + spacing_px)
        x0 = c * (tile_px + spacing_px)
        cx = x0 + tile_px / 2.0
        cy = y0 + tile_px / 2.0
        col = rgb[idx]
        Y = 0.2126 * col[0] + 0.7152 * col[1] + 0.0722 * col[2]
        text_color = (0.0, 0.0, 0.0) if Y > 0.5 else (1.0, 1.0, 1.0)
        ax.text(
            cx,
            cy,
            fmt.format(float(perc[idx])),
            ha="center",
            va="center",
            fontsize=10,
            weight="bold",
            color=text_color,
        )
    if title:
        ax.set_title(title)
    plt.tight_layout()
    return fig, ax


def compute_top_colors_kmeans(
    loader: Iterable[tuple[Any, torch.Tensor, Any]],
    /,
    topk: int = 16,
    *,
    sample_fraction: float = 0.05,
    rng: torch.Generator | None = None,
    input_scale: float = 1.0,
    chroma_floor: float = 8.0,
    max_points: int = 1_000_000,
    ab_clip: tuple[float, float] | None = (-110.0, 110.0),
    n_iter: int = 25,
    tol: float = 1e-3,
    init_centers_ab: torch.Tensor | None = None,
) -> _TopColors:
    if not (0.0 < sample_fraction <= 1.0):
        raise ValueError("`sample_fraction` must be in (0, 1].")
    if topk < 1:
        raise ValueError("`topk` must be >= 1.")

    def _collect_points() -> tuple[torch.Tensor, int]:
        buf: list[torch.Tensor] = []
        kept_total = 0
        with torch.no_grad():
            for *_, ab, _ in loader:
                x = ab.reshape(2, -1).T
                if sample_fraction < 1.0:
                    N = x.shape[0]
                    keep = max(1, int(N * sample_fraction))
                    idx = torch.randint(N, (keep,), generator=rng, device=x.device)
                    x = x[idx]
                x = x * input_scale
                if chroma_floor > 0.0:
                    C = torch.linalg.norm(x, dim=1)
                    m = chroma_floor <= C
                    if not m.any():
                        continue
                    x = x[m]
                if ab_clip is not None:
                    lo, hi = ab_clip
                    m = (
                        (x[:, 0] >= lo)
                        & (x[:, 0] <= hi)
                        & (x[:, 1] >= lo)
                        & (x[:, 1] <= hi)
                    )
                    if not m.any():
                        continue
                    x = x[m]
                kept_total += x.shape[0]
                if x.numel():
                    buf.append(x.detach().to("cpu", dtype=torch.float32))

                    if sum(t.shape[0] for t in buf) > int(max_points * 1.5):
                        tmp = torch.cat(buf, 0)
                        sel = torch.randint(tmp.shape[0], (max_points,), generator=rng)
                        buf = [tmp[sel]]
        if not buf:
            raise RuntimeError("No data found in loader after chroma filtering.")
        pts = torch.cat(buf, 0)
        if pts.shape[0] > max_points:
            sel = torch.randint(pts.shape[0], (max_points,), generator=rng)
            pts = pts[sel]
        return pts, kept_total

    def _kmeans_pp(x: torch.Tensor, k: int) -> torch.Tensor:
        N = x.shape[0]
        centers = torch.empty(k, 2, dtype=x.dtype)
        i0 = torch.randint(N, (1,), generator=rng).item()
        centers[0] = x[i0]
        d2 = ((x - centers[0]) ** 2).sum(1)
        for i in range(1, k):
            probs = d2 / d2.sum().clamp_min(1e-12)
            j = torch.multinomial(probs, 1, generator=rng).item()
            centers[i] = x[j]
            d2 = torch.minimum(d2, ((x - centers[i]) ** 2).sum(1))
        return centers

    def _lloyd(x: torch.Tensor, c0: torch.Tensor, iters: int, tol: float):
        c = c0.clone()
        N, K = x.shape[0], c.shape[0]
        last_shift = float("inf")
        for _ in range(iters):
            dist2 = torch.cdist(x, c, p=2) ** 2
            labels = dist2.argmin(dim=1)
            counts = torch.bincount(labels, minlength=K).to(torch.int64)

            if (counts == 0).any():
                near = dist2.min(dim=1).values
                empties = (counts == 0).nonzero(as_tuple=False).squeeze(1)
                far_idx = torch.topk(near, k=empties.numel()).indices
                c[empties] = x[far_idx]

                dist2 = torch.cdist(x, c, p=2) ** 2
                labels = dist2.argmin(dim=1)
                counts = torch.bincount(labels, minlength=K).to(torch.int64)

            new_c = torch.zeros_like(c)
            for k in range(K):
                m = labels == k
                new_c[k] = x[m].mean(dim=0)
            shift = (new_c - c).pow(2).sum().sqrt().item()
            c = new_c
            last_shift = shift
            if shift <= tol:
                break

        dist2 = torch.cdist(x, c, p=2) ** 2
        labels = dist2.argmin(dim=1)
        counts = torch.bincount(labels, minlength=c.shape[0]).to(torch.int64)
        return c, counts, last_shift

    pts, kept_total = _collect_points()
    if pts.shape[0] < topk:
        uniq = torch.unique(pts, dim=0)
        centers = uniq[:topk]
        counts = torch.ones(centers.shape[0], dtype=torch.int64)
    else:
        if init_centers_ab is not None:
            centers0 = init_centers_ab.detach().to("cpu", dtype=torch.float32)
            centers0 = centers0[:topk]
            if centers0.shape[0] < topk:
                extra = _kmeans_pp(pts, topk - centers0.shape[0])
                centers0 = torch.cat([centers0, extra], dim=0)
        else:
            centers0 = _kmeans_pp(pts, topk)
        centers, counts, _ = _lloyd(pts, centers0, n_iter, tol)

    order = torch.argsort(counts, descending=True)
    centers = centers[order][:topk].contiguous()
    counts = counts[order][:topk].contiguous()
    perc = (counts.to(torch.float32) / float(pts.shape[0])) * 100.0

    return _TopColors(
        centers_ab=centers.to(torch.float32),
        counts=counts.to(torch.int64),
        percents=perc.to(torch.float32),
        total_count=int(pts.shape[0]),
    )
