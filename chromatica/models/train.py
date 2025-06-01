"""Models used for train task."""

from pydantic import BaseModel, Field


class ModelTrainHyperparameters(BaseModel):
    """Hyperparameters used during training."""

    train_batch_size: int = Field(
        ...,
        description="Batch size in train data loader.",
    )
    test_batch_size: int = Field(
        ...,
        description="Batch size in test data loader.",
    )
    learning_rate: float = Field(
        ...,
        description="Learning rate used in Adam optimizer.",
    )
    num_epochs: int = Field(
        ...,
        description="Number of epochs (iteration) during training.",
    )
    train_shuffle: bool = Field(
        default=True,
        description="Apply shuffle on dataset during training.",
    )
    test_shuffle: bool = Field(
        default=False,
        description="Apply shuffle on dataset during testing.",
    )
