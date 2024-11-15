"""Function(s) for load and prepare dataset."""

from clearml import Dataset
from torchvision.datasets.food101 import Food101


def prepare() -> None:
    """
    Prepare the dataset for use in training or testing.

    Dataset can be loaded if necessary from ClearML or torchvision.
    """
    Food101(".data", "train", download=True)
    Food101(".data", "test", download=True)

    dataset = Dataset.create(
        dataset_name="food101-colorization",
        dataset_project="Chromatica",
        dataset_version="1.0.0",
        description=(
            "Food101 loaded from torchvision datasets. "
            "X is grayscale img, Y is original."
        ),
    )

    dataset.add_files(path=".data/")

    dataset.upload(show_progress=True)
    dataset.finalize()
