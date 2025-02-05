"""CLI for load and prepare dataset."""

import argparse

from chromatica.core.dataset.food101_colorization import (
    NEW_DATASET_NAME,
    OLD_DATASET_NAME,
)
from chromatica.core.tasks import dataset
from chromatica.models.dataset import CreateDatasetConfig

DATASET_FOLDER = ".data/"
OLD_DATASET_FOLDER = DATASET_FOLDER + OLD_DATASET_NAME
NEW_DATASET_FOLDER = DATASET_FOLDER + NEW_DATASET_NAME


# TODO: Rewrite CLI with Typer (https://typer.tiangolo.com/)
# https://github.com/snailUlitka/chromatica/issues/5
def create() -> None:
    """Create dataset, uses `create` from `core`."""
    parser = argparse.ArgumentParser(
        description="CLI tool for create dataset",
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Name for ClearML",
    )
    parser.add_argument(
        "--dataset_version",
        type=str,
        required=True,
        help="Version for ClearML",
    )
    parser.add_argument(
        "--dataset_description",
        type=str,
        required=False,
        help="Description for ClearML",
    )

    args = parser.parse_args()

    dataset.create(
        CreateDatasetConfig(
            dataset_name=args.dataset_name,
            dataset_version=args.dataset_version,
            dataset_description=args.dataset_description,
        ),
    )
