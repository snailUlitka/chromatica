"""Function(s) for load and prepare dataset."""

import argparse
import json
import shutil

import torch
from clearml import Dataset
from pathlib2 import Path
from torchvision.datasets.food101 import Food101
from torchvision.transforms.v2 import RGB, Compose, Resize, ToDtype, ToImage
from tqdm import tqdm

from chromatica.models.dataset import DatasetMetadata

DATASET_FOLDER = ".data/"
OLD_DATASET_FOLDER = DATASET_FOLDER + "food-101/"
NEW_DATASET_FOLDER = DATASET_FOLDER + "food101-colorization/"


class Food101Colorization(torch.utils.data.Dataset):
    """Dataset for colorization task based on Food101."""

    def __init__(self, *, train: bool = True):
        self._folder_path = None
        self._train = train

        self._transform = Compose(
            [
                ToImage(),
                Resize((256, 256)),
                RGB(),
                ToDtype(torch.float32, scale=True),
            ],
        )

    def _load(self) -> None:
        dataset = Dataset.get(
            dataset_name="food101-colorization",
            dataset_project="Chromatica",
        )

        self._folder_path = Path(dataset.get_local_copy())

        metadata_file_path = (
            self._folder_path / Path("food101-colorization") / Path("METADATA")
        )

        with metadata_file_path.open(mode="r") as metadata_file:
            metadata = json.load(metadata_file)

        self._metadata = DatasetMetadata(
            dataset_len=metadata["dataset_len"],
            train_len=metadata["train_len"],
            test_len=metadata["test_len"],
        )

    def __len__(self):
        """Length of dataset slice.

        If 'train' is True, returns length of train slice, otherwise of test
        slice.

        Returns
        -------
        int
            Length of slice
        """
        if self._folder_path is None:
            self._load()

        if self._train:
            return self._metadata.train_len
        return self._metadata.test_len

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return item of dataset slice.

        Parameters
        ----------
        idx : int
            Index of the element to be returned.
        """


def create() -> None:
    """Create dataset from Food101 and upload it in ClearML cloud."""
    parser = argparse.ArgumentParser(
        description="CLI tool for create dataset",
    )

    parser.add_argument(
        "-o",
        "--overwrite",
        action="store_true",
        help="If set, the data in `.data` will be overwritten",
    )

    args = parser.parse_args()

    if args.overwrite:
        shutil.rmtree(DATASET_FOLDER)

    paths = [
        DATASET_FOLDER,
        NEW_DATASET_FOLDER,
    ]

    for path in paths:
        Path(path).mkdir(exist_ok=True)

    train = Food101(
        root=DATASET_FOLDER,
        split="train",
        download=True,
    )
    test = Food101(
        root=DATASET_FOLDER,
        split="test",
        download=True,
    )

    with Path(NEW_DATASET_FOLDER + "METADATA").open(mode="w") as meta:
        metadata = json.dumps(
            {
                "dataset_len": len(train) + len(test),
                "train_len": len(train),
                "test_len": len(test),
            },
        )
        meta.write(metadata)

    source_path = Path(OLD_DATASET_FOLDER + "images/")
    dest_path = Path(NEW_DATASET_FOLDER)
    dataset_folders = [
        subfolder for subfolder in source_path.iterdir() if subfolder.is_dir()
    ]

    for images in tqdm(dataset_folders, desc="Copying"):
        (dest_path / "train" / images.name).mkdir(parents=True, exist_ok=True)
        (dest_path / "test" / images.name).mkdir(parents=True, exist_ok=True)

        for idx, image in enumerate(images.iterdir()):
            images_count = len(list(images.iterdir()))
            split = "train" if idx < int(images_count * 0.7) else "test"

            new_filename = f"{images.name}_{idx}{image.suffix}"

            shutil.copy2(image, dest_path / split / images.name / new_filename)

    for item in Path(DATASET_FOLDER).iterdir():
        if item.name != dest_path.name:
            if item.is_dir():
                shutil.rmtree(item.absolute())
            else:
                item.unlink()

    dataset = Dataset.create(
        dataset_name="food101-colorization",
        dataset_project="Chromatica",
        dataset_version="1.1.0",
        description="Food101 loaded from torchvision datasets.",
    )

    dataset.add_files(path=DATASET_FOLDER)

    dataset.upload()
    dataset.finalize()
