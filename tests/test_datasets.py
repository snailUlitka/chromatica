"""Real filenames, uneven classes, and explicit CPU preprocessing."""

from pathlib import Path

import pytest
import torch
from PIL import Image

from chromatica.datasets.dataset import DirectoryImageDataset, ImageDataset


def test_sparse_names_and_unequal_class_sizes(tmp_path: Path) -> None:
    """Discovery must not synthesize class_0...class_N filenames."""
    for name in ("apple_7.png", "apple_21.png", "banana_3.png"):
        Image.new("RGB", (12, 9), (200, 50, 10)).save(tmp_path / name)
    (tmp_path / "notes.txt").write_text("not an image", encoding="utf-8")
    dataset = ImageDataset(tmp_path, image_size=16)
    assert len(dataset) == 3
    assert dataset.labels == ["apple", "banana"]
    assert [dataset.label_from_index(index) for index in range(3)] == [
        "apple",
        "apple",
        "banana",
    ]
    l_channel, ab, label = dataset[1]
    assert l_channel.shape == (1, 16, 16)
    assert ab.shape == (2, 16, 16)
    assert ab.device.type == "cpu"
    assert label == 0


def test_recursive_class_folders(tmp_path: Path) -> None:
    """Standard Food-101 class folders produce stable labels."""
    for label in ("pizza", "cake"):
        (tmp_path / label).mkdir()
        Image.new("RGB", (8, 8)).save(tmp_path / label / "sample.jpg")
    dataset = DirectoryImageDataset(tmp_path, image_size=16)
    assert dataset.labels == ["cake", "pizza"]


def test_empty_directories_and_gpu_dataset_request(tmp_path: Path) -> None:
    """Empty datasets and transfers inside dataset workers fail explicitly."""
    with pytest.raises(ValueError, match="No image"):
        DirectoryImageDataset(tmp_path)
    with pytest.raises(ValueError, match="CPU tensors"):
        DirectoryImageDataset(tmp_path, device=torch.device("cuda"))
