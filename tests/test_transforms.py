"""Known colors, physical units, and channel-pairing regression tests."""

import pytest
import torch

from chromatica.datasets.transform import (
    LAB2RGB,
    RGB2LAB,
    ab_to_pairs,
    decode_ab,
    decode_lab,
    encode_ab,
)


def test_rgb_lab_round_trip_for_batched_primary_colors() -> None:
    """White/black/primary colors survive a non-differentiable Lab round trip."""
    colors = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    images = colors[:, :, None, None].expand(-1, -1, 2, 2).contiguous()
    lab = RGB2LAB()(images)
    assert lab.shape == images.shape
    assert torch.allclose(LAB2RGB()(lab), images, atol=2e-5)
    physical = decode_lab(lab)
    assert physical[0, 0, 0, 0].item() == pytest.approx(0.0, abs=1e-5)
    assert physical[1, 0, 0, 0].item() == pytest.approx(100.0, abs=1e-4)


def test_affine_offset_is_preserved() -> None:
    """Normalized zero decodes to -0.5, while physical neutral encodes above zero."""
    zero = torch.zeros(2, 1, 1)
    assert torch.equal(decode_ab(zero), torch.full_like(zero, -0.5))
    assert torch.allclose(encode_ab(zero), torch.full_like(zero, 0.5 / 127.5))
    assert torch.allclose(decode_ab(encode_ab(zero)), zero, atol=1e-7)


def test_ab_pairs_do_not_mix_images() -> None:
    """The B>1 regression keeps a and b from each image and pixel together."""
    ab = torch.tensor([[[[1.0, 2.0]], [[10.0, 20.0]]], [[[3.0, 4.0]], [[30.0, 40.0]]]])
    expected = torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])
    assert torch.equal(ab_to_pairs(ab), expected)
    assert torch.equal(ab_to_pairs(ab.transpose(-1, -2)), expected)


@pytest.mark.parametrize(
    "tensor",
    [
        torch.zeros(3, 2, 2, dtype=torch.uint8),
        torch.full((3, 2, 2), float("nan")),
        torch.full((3, 2, 2), 2.0),
    ],
)
def test_invalid_rgb_is_rejected(tensor: torch.Tensor) -> None:
    """Invalid dtypes, nonfinite values, and non-sRGB ranges fail explicitly."""
    with pytest.raises((TypeError, ValueError)):
        RGB2LAB()(tensor)
