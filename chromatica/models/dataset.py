"""Models used for dataset."""

from enum import Enum

from pydantic import BaseModel, Field


class DatasetMetadata(BaseModel):
    """Metadata of dataset, e.g. dataset length."""

    dataset_len: int = Field(..., description="Full length of dataset.")
    train_len: int = Field(..., description="Train length of dataset.")
    test_len: int = Field(..., description="Test length of dataset.")
    classes_names: list[str] = Field(
        ...,
        description="List of unique classes names in dataset.",
    )
    classes_count: int = Field(
        ...,
        description="Count of unique classes in dataset",
    )


class DatasetSplitType(str, Enum):
    """Parameter for selecting the data to use from the dataset."""

    TRAIN = "train"
    TEST = "test"
