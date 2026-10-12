"""Historical architecture contracts and checkpoint parameter compatibility."""

from pathlib import Path

import pytest
import torch
from torch import nn

from chromatica.models import EncoderDecoder, UNet, create_model
from chromatica.models.checkpoints import load_checkpoint, save_checkpoint
from chromatica.nn.v1.cnn import CNN as HISTORICAL_V1
from chromatica.nn.v2.cnn import CNN as HISTORICAL_V2


@pytest.mark.parametrize("name", ["encoder_decoder", "unet"])
def test_model_shapes_for_batches_and_single_images(name: str) -> None:
    """The public tensor contract supports batched and unbatched evaluation."""
    torch.manual_seed(42)
    model = create_model(name).eval()
    with torch.inference_mode():
        assert model(torch.zeros(2, 1, 16, 24)).shape == (2, 2, 16, 24)
        assert model(torch.zeros(1, 16, 24)).shape == (2, 16, 24)
    with pytest.raises(ValueError, match="multiples of eight"):
        model(torch.zeros(2, 1, 17, 24))


def test_historical_aliases_and_output_activations() -> None:
    """Renaming retains the historical models and their different activations."""
    assert HISTORICAL_V1 is EncoderDecoder
    assert HISTORICAL_V2 is UNet
    no_skips = create_model("v1")
    skips = create_model("v2")
    assert isinstance(no_skips.nn[-1].act, nn.Tanh)
    assert isinstance(skips.c14.act, nn.ReLU)
    assert "nn.0.conv.weight" in no_skips.state_dict()
    assert "c14.bn.running_mean" in skips.state_dict()


def test_checkpoint_metadata_and_legacy_dictionary(tmp_path: Path) -> None:
    """Primitive metadata checkpoints and legacy state dictionaries load strictly."""
    model = create_model("encoder_decoder")
    path = tmp_path / "new.pth"
    save_checkpoint(path, model, "v1", {"dataset": "synthetic_fixture"})
    restored, metadata = load_checkpoint(path)
    assert metadata["architecture"] == "encoder_decoder"
    assert torch.equal(
        restored.state_dict()["nn.0.conv.weight"],
        model.state_dict()["nn.0.conv.weight"],
    )
    with pytest.raises(ValueError, match="does not match"):
        load_checkpoint(path, architecture="unet")
    legacy = tmp_path / "legacy.pth"
    torch.save(model.state_dict(), legacy)
    with pytest.raises(ValueError, match="explicit architecture"):
        load_checkpoint(legacy)
    _, legacy_metadata = load_checkpoint(legacy, architecture="v1")
    assert legacy_metadata["training_provenance"] == "unknown"
