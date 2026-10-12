"""Hue charts and physical Lab palettes (install the visualization extra)."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.color import lab2rgb

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from chromatica.metrics.palettes import TopColors

TWO_PI = 2.0 * math.pi


def plot_histogram(
    hist: torch.Tensor,
    /,
    *,
    l_value: float = 50.0,
    title: str | None = "Hue-based color histogram",
) -> tuple[Figure, Axes]:
    """
    Render hue fractions of eligible pixels as a colored bar plot.

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

    if hist.numel() == 0 or not torch.isfinite(hist).all() or (hist < 0).any():
        msg = "Histogram must contain finite nonnegative counts."
        raise ValueError(msg)
    nbins = hist.numel()
    data = hist.to(dtype=torch.float32, device="cpu").numpy()
    total = data.sum()
    if total > 0:
        data /= total

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

    ax.set_xticks(np.linspace(0, nbins, 5), ["0", "90", "180", "270", "360"])
    ax.set_xlim(0, nbins)
    ax.set_xlabel("Hue (degrees, from positive a*)")
    ax.set_ylabel("Fraction of eligible pixels")

    if title:
        ax.set_title(title)

    plt.tight_layout()
    return fig, ax


def plot_top_colors_pie(  # noqa: D103, PLR0913
    top: TopColors,
    /,
    *,
    l_value: float = 50.0,
    title: str | None = "Top colors",
    include_other: bool = True,
    other_label: str = "Other colors",
    min_show_other: float = 0.05,
) -> tuple[Figure, Axes]:
    if top.centers_ab.shape[0] == 0:
        msg = "Palette must contain at least one color."
        raise ValueError(msg)
    k = top.centers_ab.shape[0]
    lab = np.empty((k, 3), dtype=np.float32)
    lab[:, 0] = l_value
    lab[:, 1:] = top.centers_ab.numpy()
    rgb = lab2rgb(lab[None])[0]
    sizes = top.percents.numpy().tolist()
    colors = rgb.tolist()
    labels = [""] * k
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


def plot_top_colors_palette(  # noqa: D103, PLR0913
    top: TopColors,
    /,
    *,
    l_value: float = 50.0,
    title: str | None = "Top colors palette",
    cols: int | None = None,
    tile_px: int = 80,
    spacing_px: int = 4,
    fmt: str = "{:.1f}%",
) -> tuple[Figure, Axes]:
    if top.centers_ab.shape[0] == 0:
        msg = "Palette must contain at least one color."
        raise ValueError(msg)
    k = top.centers_ab.shape[0]
    lab = np.empty((k, 3), dtype=np.float32)
    lab[:, 0] = l_value
    lab[:, 1:] = top.centers_ab.numpy()
    rgb = lab2rgb(lab[None])[0]
    if cols is None or cols < 1:
        cols = min(k, 8)
    rows = math.ceil(k / cols)
    h = rows * tile_px + (rows - 1) * spacing_px if rows > 0 else tile_px
    w = cols * tile_px + (cols - 1) * spacing_px if cols > 0 else tile_px
    canvas = np.ones((h, w, 3), dtype=np.float32)
    for idx in range(k):
        r = idx // cols
        c = idx % cols
        y0 = r * (tile_px + spacing_px)
        x0 = c * (tile_px + spacing_px)
        canvas[y0 : y0 + tile_px, x0 : x0 + tile_px, :] = rgb[idx]
    fig, ax = plt.subplots(figsize=(w / 80.0, h / 80.0), dpi=160)
    ax.imshow(canvas, interpolation="nearest")
    _ = ax.set_xticks([]), ax.set_yticks([])
    _ = ax.set_xlim(0, w), ax.set_ylim(h, 0)
    perc = top.percents.numpy()
    for idx in range(k):
        r = idx // cols
        c = idx % cols
        y0 = r * (tile_px + spacing_px)
        x0 = c * (tile_px + spacing_px)
        cx = x0 + tile_px / 2.0
        cy = y0 + tile_px / 2.0
        col = rgb[idx]
        y = 0.2126 * col[0] + 0.7152 * col[1] + 0.0722 * col[2]
        text_color = (0.0, 0.0, 0.0) if y > 0.5 else (1.0, 1.0, 1.0)  # noqa: PLR2004
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
