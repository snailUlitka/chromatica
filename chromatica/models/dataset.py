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


class CreateDatasetConfig(BaseModel):
    """Config for dataset create method."""

    dataset_name: str = Field(..., description="Name for ClearML")
    dataset_version: str = Field(
        ...,
        description="Version for ClearML",
        examples=["1.0.1", "0.0.0", "101.0.0"],
    )
    dataset_description: str | None = Field(
        None,
        description="Description for ClearML",
    )


class DatasetSplitType(str, Enum):
    """Parameter for selecting the data to use from the dataset."""

    TRAIN = "train"
    TEST = "test"
