"""
The adapted version of the Food101 dataset for the purpose of colorization.

Original dataset: https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/
"""

import json

import torch
from clearml import Dataset
from pathlib2 import Path
from PIL import Image
from PIL.ImageFile import ImageFile
from torchvision.transforms.v2 import (
    RGB,
    Compose,
    PILToTensor,
    Resize,
    ToDtype,
)

from chromatica.dataset.transform import RGB2LAB
from chromatica.models.dataset import DatasetMetadata, DatasetSplitType

OLD_DATASET_NAME = "food-101"
NEW_DATASET_NAME = "food101-colorization"


class Food101Colorization(torch.utils.data.Dataset):
    """Dataset for colorization task based on Food101."""

    def __init__(self, *, train: DatasetSplitType):
        self._folder_path = None
        self._split_type = train

    def _load(self) -> None:
        dataset = Dataset.get(
            dataset_name="food101-colorization",
            dataset_project="Chromatica",
        )

        self._folder_path = Path(dataset.get_local_copy())

        metadata_file_path = (
            self._folder_path / Path(NEW_DATASET_NAME) / Path("METADATA")
        )

        with metadata_file_path.open(mode="r") as metadata_file:
            metadata = json.load(metadata_file)

        self._metadata = DatasetMetadata(
            dataset_len=metadata["dataset_len"],
            train_len=metadata["train_len"],
            test_len=metadata["test_len"],
            classes_names=metadata["classes_names"],
            classes_count=metadata["classes_count"],
        )

    def __len__(self) -> int:
        """Length of current dataset split (train or test).

        Returns
        -------
        int
            Length of split
        """
        if self._folder_path is None:
            self._load()

        match self._split_type:
            case DatasetSplitType.TRAIN:
                return self._metadata.train_len
            case DatasetSplitType.TEST:
                return self._metadata.test_len

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return pair of dataset split, X and Y.

        Parameters
        ----------
        idx : int
            Index of the element to be returned.

        Returns
        -------
        tuple[Tensor, Tensor]
            - First tensor is X
            - Second tensor is Y
        """
        if self._folder_path is None:
            self._load()

        original_image = Image.open(self._get_item_path(index))

        return self._prepare_image(original_image)

    def __getitems__(
        self,
        indexes: list[int],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Return list of pairs of dataset split, X and Y.

        Parameters
        ----------
        idx : int
            Index of the element to be returned.

        Returns
        -------
        list[tuple[Tensor, Tensor]]
            - First tensor is X
            - Second tensor is Y
        """
        if self._folder_path is None:
            self._load()

        paths = [self._get_item_path(idx) for idx in indexes]
        images = [Image.open(path) for path in paths]

        return [self._prepare_image(img) for img in images]

    def _prepare_image(
        self,
        image: ImageFile,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        transform = Compose(
            [
                PILToTensor(),
                Resize((256, 256)),
                RGB(),
                ToDtype(torch.float32, scale=True),
                RGB2LAB(),
            ],
        )

        tensor_with_image = transform(image)

        return tensor_with_image[:1, :, :], tensor_with_image

    def _get_item_path(self, index: int) -> Path:
        train_len = self._metadata.train_len
        test_len = self._metadata.test_len
        classes_count = self._metadata.classes_count

        class_name = self._metadata.classes_names[
            index // (train_len // classes_count)
        ]

        match self._split_type:
            case DatasetSplitType.TRAIN:
                index_in_class = index % (train_len // classes_count)
            case DatasetSplitType.TEST:
                index_in_class = index % (test_len // classes_count)

                offset = train_len // classes_count
                index_in_class += offset

        return Path(
            f"{self._folder_path}/{NEW_DATASET_NAME}/{self._split_type.value}/"
            f"{class_name}/{class_name}_{index_in_class}.jpg",
        )
