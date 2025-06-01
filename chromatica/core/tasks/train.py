"""An abstraction for train CNN."""

import tempfile
import time

from clearml.model import Model
import torch
from clearml import Task
from pathlib2 import Path
from torch import nn
from torch.utils.data import DataLoader, Subset

from chromatica.core.cnn import CNN
from chromatica.core.dataset.food101_colorization import Food101Colorization
from chromatica.models.dataset import DatasetSplitType
from chromatica.models.train import ModelTrainHyperparameters

# TODO: Enhance Hyperparameter Selection
# - Add an option to choose whether to use the full dataset or a subset.
# - Implement the ability to select a loss function.
# - Implement the ability to choose an optimizer.
# https://github.com/snailUlitka/chromatica/issues/13


def train(hyperparameters: ModelTrainHyperparameters) -> None:
    """Train CNN with given hyperparameters."""
    task = Task.init(
        project_name="Chromatica",
        task_name="Chromatica CNN train",
        task_type=Task.TaskTypes.training,
    )

    model = CNN()

    _train(task, model, hyperparameters)
    _test(task, model, hyperparameters)

    path = Path(tempfile.gettempdir(), "model.pt")

    torch.save(
        obj=model.state_dict(),
        f=path,
    )

    task.upload_artifact(name="Model", artifact_object=path)

    task.close()


def _train(
    task: Task,
    model: CNN,
    hyperparameters: ModelTrainHyperparameters,
) -> None:
    logger = task.get_logger()

    train_loader = DataLoader(
        Subset(
            Food101Colorization(split=DatasetSplitType.TRAIN),
            list(range(1000)),
        ),
        batch_size=hyperparameters.train_batch_size,
        shuffle=hyperparameters.train_shuffle,
    )

    criterion = nn.MSELoss()
    optim = torch.optim.Adam(
        params=model.parameters(),
        lr=hyperparameters.learning_rate,
    )
    total_iterations = hyperparameters.num_epochs * len(train_loader)

    start_time = time.time()
    iteration = 0

    model.train()
    for epoch in range(hyperparameters.num_epochs):
        train_loss = 0
        for i, (x, y) in enumerate(train_loader):
            optim.zero_grad()

            pred = model(x)
            loss = criterion(pred, y[:, 1:, :, :])

            loss.backward()
            optim.step()
            train_loss += loss.item()

            iteration += 1
            elapsed = time.time() - start_time
            avg_iter_time = elapsed / iteration
            remaining_iterations = total_iterations - iteration
            eta = avg_iter_time * remaining_iterations / 60

            logger.report_scalar(
                title="Training ETA",
                series="ETA",
                value=eta,
                iteration=iteration,
            )
            logger.report_scalar(
                title="Train loss",
                series="loss",
                value=loss.item(),
                iteration=i,
            )

        logger.report_scalar(
            title="Train loss (By Epochs)",
            series="loss (sum)",
            value=train_loss,
            iteration=epoch,
        )
        logger.report_scalar(
            title="Train loss (By Epochs)",
            series="loss (avg)",
            value=train_loss / len(train_loader),
            iteration=epoch,
        )

    total_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    logger.report_single_value(
        name="Total parameters",
        value=total_params,
    )


def _test(
    task: Task,
    model: CNN,
    hyperparameters: ModelTrainHyperparameters,
) -> None:
    logger = task.get_logger()

    test_loader = DataLoader(
        Subset(
            Food101Colorization(split=DatasetSplitType.TEST),
            list(range(500)),
        ),
        batch_size=hyperparameters.test_batch_size,
        shuffle=hyperparameters.test_shuffle,
    )

    criterion = nn.MSELoss()

    total_loss = 0
    model.eval()
    for i, (x, y) in enumerate(test_loader):
        pred = model(x)
        loss = criterion(pred, y[:, 1:, :, :])

        total_loss += loss.item()

        logger.report_scalar(
            title="Test loss",
            series="loss",
            value=loss.item(),
            iteration=i,
        )

    logger.report_single_value(
        name="Avg test loss",
        value=total_loss / len(test_loader) * 1000,
    )


if __name__ == "__main__":
    p = ModelTrainHyperparameters(
        train_batch_size=10,
        test_batch_size=10,
        learning_rate=1e-4,
        num_epochs=1,
    )

    train(p)

    # FIX: Единственный вариант это грузить ссаный артефакт, а не модель. Почему так сделано, я хз
    # m = Model.query_models(project_name="Chromatica")[0]
    #
    # print(m.get_local_copy(force_download=True))
