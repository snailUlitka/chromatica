"""Models used for dataset."""

from pydantic import BaseModel, Field


class DatasetMetadata(BaseModel):
    """Metadata of dataset, e.g. dataset length."""

    dataset_len: int = Field(..., description="Full length of dataset.")
    train_len: int = Field(..., description="Train length of dataset.")
    test_len: int = Field(..., description="Test length of dataset.")
