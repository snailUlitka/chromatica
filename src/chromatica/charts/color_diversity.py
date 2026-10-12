"""Compatibility imports; computations live in metrics and plots in visualization."""

from chromatica.metrics.hue import compute_histogram
from chromatica.metrics.palettes import (
    compute_top_colors_kmeans,
    compute_top_colors_modes,
)
from chromatica.visualization.color_diversity import (
    plot_histogram,
    plot_top_colors_palette,
    plot_top_colors_pie,
)

__all__ = [
    "compute_histogram",
    "compute_top_colors_kmeans",
    "compute_top_colors_modes",
    "plot_histogram",
    "plot_top_colors_palette",
    "plot_top_colors_pie",
]
