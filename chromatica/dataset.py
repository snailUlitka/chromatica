"""Function(s) for load and prepare dataset."""

import shutil
from collections import defaultdict

import numpy as np
import torch
from clearml import Dataset
from pathlib2 import Path
from skimage.color import rgb2lab
from torch.utils.data import DataLoader
from torchvision.datasets.food101 import Food101
from torchvision.transforms.v2 import RGB, Compose, Resize, ToDtype, ToImage
from tqdm import tqdm

DATASET_FOLDER = ".data/"
NEW_DATASET_FOLDER = DATASET_FOLDER + "food101-colorization/"


def prepare() -> None:
    """
    Prepare the dataset for use in training or testing.

    Dataset can be loaded if necessary from ClearML or torchvision.
    """
    transform = Compose(
        [
            ToImage(),
            Resize((256, 256)),
            RGB(),
            ToDtype(torch.float32, scale=True),
        ],
    )

    train = Food101(
        root=DATASET_FOLDER,
        split="train",
        transform=transform,
        download=True,
    )
    test = Food101(
        root=DATASET_FOLDER,
        split="test",
        transform=transform,
        download=True,
    )

    Path(NEW_DATASET_FOLDER).mkdir(exist_ok=True)

    _preprocess_dataset(train, test)

    shutil.rmtree(DATASET_FOLDER + "food-101/")

    # WARN: WIP, doesn't execute this
    dataset = Dataset.create(
        dataset_name="food101-colorization",
        dataset_project="Chromatica",
        dataset_version="1.1.0",
        description=(
            "Food101 loaded from torchvision datasets. "
            "X is grayscale img, Y is original."
        ),
    )

    dataset.add_files(path=DATASET_FOLDER)

    dataset.upload()
    dataset.finalize()


def _preprocess_dataset(train: Food101, test: Food101) -> None:
    for (
        dataset,
        split,
    ) in (
        (train, "train"),
        (test, "test"),
    ):
        loader = DataLoader(dataset)
        path = NEW_DATASET_FOLDER + f"{split}/"

        Path(path).mkdir()

        exists_paths = set()
        idx_by_class = defaultdict(int)

        for x, y in tqdm(loader, desc=split):
            class_name = dataset.classes[y.item()]
            path_with_cls = f"{path}{class_name}/"

            if path_with_cls not in exists_paths:
                Path(path_with_cls).mkdir()
                exists_paths.add(path_with_cls)

            item = torch.stack((torch.Tensor(_rgb2lab(x[0])), x[0]))

            torch.save(
                item,
                path_with_cls + f"{y.item()}_{idx_by_class[y.item()]}.pt",
            )

            idx_by_class[y.item()] += 1


def _rgb2lab(tensor: torch.Tensor) -> np.ndarray:
    rgb_tensor = tensor.detach().numpy()

    lab_tensor = rgb2lab(rgb_tensor, channel_axis=0)
    l_channel = lab_tensor[:1, :, :] / 100
    ab_channels = (lab_tensor[1:, :, :] + 128) / 127.5 - 1

    return np.vstack((l_channel, ab_channels))
