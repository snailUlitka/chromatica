"""Analytical bias, histogram, palette, and batch-partition invariants."""

import math

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from chromatica.datasets.transform import encode_ab
from chromatica.metrics.bias import BiasAccumulator
from chromatica.metrics.color_diversity import (
    compute_delta_a_stats,
    compute_delta_a_weighted,
    compute_rai,
)
from chromatica.metrics.hue import compute_histogram
from chromatica.metrics.palettes import (
    compute_top_colors_kmeans,
    compute_top_colors_modes,
)
from chromatica.training.loop import mean_loss


def physical_loader(ab: torch.Tensor, batch_size: int) -> DataLoader:
    """Return normalized ab from physical coordinate fixtures."""
    l_channel = torch.zeros(ab.shape[0], 1, *ab.shape[-2:])
    dataset = TensorDataset(
        l_channel, encode_ab(ab), torch.zeros(ab.shape[0], dtype=torch.long)
    )
    return DataLoader(dataset, batch_size=batch_size)


class ConstantAB(nn.Module):
    """Return a known constant physical color through normalized model coordinates."""

    def __init__(self, a: float, b: float) -> None:
        super().__init__()
        self.register_buffer("ab", encode_ab(torch.tensor([a, b])[:, None, None]))

    def forward(self, l_channel: torch.Tensor) -> torch.Tensor:
        """Broadcast the known color to the supplied batch shape."""
        return self.ab[None].expand(l_channel.shape[0], -1, *l_channel.shape[-2:])


def test_histogram_is_independent_of_batch_size() -> None:
    """Quadrant-center colors occupy distinct bins without boundary rounding."""
    ab = torch.tensor([[10.0, 10.0], [-10.0, 10.0], [-10.0, -10.0], [10.0, -10.0]])[
        :, :, None, None
    ]
    hist_one = compute_histogram(physical_loader(ab, 1), 4, encoding="normalized")
    hist_many = compute_histogram(physical_loader(ab, 3), 4, encoding="normalized")
    assert torch.equal(hist_one, hist_many)
    assert torch.equal(hist_many, torch.ones(4, dtype=torch.int64))


def test_known_bias_and_absolute_error() -> None:
    """Signed bias and magnitude agree with known physical coordinate shifts."""
    target = torch.zeros(5, 2, 2, 2)
    model = ConstantAB(12.0, -7.0)
    for batch_size in (1, 2, 5):
        summary = compute_delta_a_stats(
            physical_loader(target, batch_size), model, device="cpu"
        )
        assert summary["mean"] == pytest.approx(12.0, abs=1e-5)
        assert summary["mae"] == pytest.approx(12.0, abs=1e-5)
        assert summary["rmse"] == pytest.approx(12.0, abs=1e-5)
        assert summary["median"] == pytest.approx(12.0, abs=510.0 / 4096)
        assert summary["pixels"] == 20
    assert model.training


def test_bias_tails_are_reported_instead_of_discarded() -> None:
    """A delta outside support preserves its mean/error and invalidates the median."""
    stats = BiasAccumulator(bins=20, a_range=(-255.0, 255.0))
    stats.update(torch.full((2, 1, 1), 1000.0), torch.zeros(2, 1, 1), encoding="lab")
    summary = stats.summary()
    assert summary["mean_delta_a"] == 1000.0
    assert summary["delta_a_overflow"] == 1
    assert math.isnan(summary["median_delta_a"])


def test_zero_chroma_has_no_weighted_bias() -> None:
    """A neutral reference supplies no chroma weight, so the metric is undefined."""
    target = torch.zeros(2, 2, 1, 1)
    loader = DataLoader(
        TensorDataset(torch.zeros(2, 1, 1, 1), target, torch.zeros(2)), batch_size=2
    )
    model = ConstantAB(0.0, 0.0)
    model.ab.fill_(1.0)
    value = compute_delta_a_weighted(loader, model, encoding="lab")
    assert math.isnan(value)


def test_constant_palette_retains_true_pixel_counts() -> None:
    """Constant-color clusters retain counts and produce finite percentages."""
    ab = (
        torch.tensor([20.0, 10.0])[None, :, None, None].expand(5, -1, 2, 2).contiguous()
    )
    top = compute_top_colors_kmeans(physical_loader(ab, 3), topk=16, chroma_floor=0.0)
    assert top.centers_ab.shape == (1, 2)
    assert torch.allclose(top.centers_ab, torch.tensor([[20.0, 10.0]]), atol=1e-5)
    assert top.counts.tolist() == [20]
    assert top.percents.tolist() == [100.0]


def test_grid_modes_keep_batch_pairing() -> None:
    """Mode centers and counts remain consistent across different batch partitions."""
    ab = torch.tensor([[25.0, 15.0], [-25.0, -15.0], [25.0, 15.0]])[:, :, None, None]
    one = compute_top_colors_modes(physical_loader(ab, 1), topk=2, merge_radius=None)
    many = compute_top_colors_modes(physical_loader(ab, 3), topk=2, merge_radius=None)
    assert torch.equal(one.counts, many.counts)
    assert torch.allclose(one.centers_ab, many.centers_ab)


def test_empty_data_and_no_eligible_pixels_fail() -> None:
    """An absent population cannot produce an apparently valid metric."""
    with pytest.raises(ValueError, match="No pixels"):
        compute_histogram([])
    with pytest.raises(ValueError, match="No eligible"):
        BiasAccumulator().summary()


def test_mse_counts_short_final_batches_by_elements() -> None:
    """Five unequal targets produce the same MSE for batches of one, two, or five."""
    ab = (
        torch.tensor([1.0, 1.0, 1.0, 1.0, 5.0])[:, None, None, None]
        .expand(-1, 2, 1, 1)
        .contiguous()
    )
    expected = 29.0 / (5 * 127.5**2)
    model = ConstantAB(0.0, 0.0)
    for batch_size in (1, 2, 5):
        loss = mean_loss(
            physical_loader(ab, batch_size), model, device=torch.device("cpu")
        )
        assert loss == pytest.approx(expected, rel=1e-5)


def test_red_asymmetry_requires_red_mass() -> None:
    """Absent red pixels is undefined, rather than evidence of balanced red hues."""
    green_only = torch.zeros(360, dtype=torch.int64)
    green_only[180] = 5
    assert math.isnan(compute_rai(green_only))
    assert math.isnan(compute_rai(torch.ones(1)))
    balanced = torch.zeros(360)
    balanced[0], balanced[-1] = 3, 3
    assert compute_rai(balanced) == 0.0
