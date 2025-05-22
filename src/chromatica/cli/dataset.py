"""Function(s) for load and prepare dataset."""

import argparse
import json
import shutil

from clearml import Dataset
from pathlib2 import Path
from torchvision.datasets.food101 import Food101
from tqdm import tqdm

from chromatica.dataset.food101_colorization import (
    NEW_DATASET_NAME,
    OLD_DATASET_NAME,
)

DATASET_FOLDER = ".data/"
OLD_DATASET_FOLDER = DATASET_FOLDER + OLD_DATASET_NAME
NEW_DATASET_FOLDER = DATASET_FOLDER + NEW_DATASET_NAME


# TODO: Rewrite CLI with Typer (https://typer.tiangolo.com/)
# https://github.com/snailUlitka/chromatica/issues/5
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

    _ = Food101(
        root=DATASET_FOLDER,
        split="train",
        download=True,
    )
    _ = Food101(
        root=DATASET_FOLDER,
        split="test",
        download=True,
    )

    source_path = Path(OLD_DATASET_FOLDER + "images/")
    dest_path = Path(NEW_DATASET_FOLDER)
    dataset_folders = [
        subfolder for subfolder in source_path.iterdir() if subfolder.is_dir()
    ]
    train_len = test_len = 0

    uniq_classes = set()

    for images in tqdm(dataset_folders, desc="Copying"):
        uniq_classes.add(images.name)

        (dest_path / "train" / images.name).mkdir(parents=True, exist_ok=True)
        (dest_path / "test" / images.name).mkdir(parents=True, exist_ok=True)

        for idx, image in enumerate(images.iterdir()):
            images_count = len(list(images.iterdir()))
            split = "train" if idx < int(images_count * 0.7) else "test"

            if split == "train":
                train_len += 1
            else:
                test_len += 1

            new_filename = f"{images.name}_{idx}{image.suffix}"

            shutil.copy2(image, dest_path / split / images.name / new_filename)

    with Path(NEW_DATASET_FOLDER + "METADATA").open(mode="w") as meta:
        metadata = json.dumps(
            {
                "dataset_len": train_len + test_len,
                "train_len": train_len,
                "test_len": test_len,
                "classes_names": sorted(uniq_classes),
                "classes_count": len(uniq_classes),
            },
        )
        meta.write(metadata)

    for item in Path(DATASET_FOLDER).iterdir():
        if item.name != dest_path.name:
            if item.is_dir():
                shutil.rmtree(item.absolute())
            else:
                item.unlink()

    dataset = Dataset.create(
        dataset_name=NEW_DATASET_NAME,
        dataset_project="Chromatica",
        dataset_version="1.1.2",
        description="Food101 loaded from torchvision datasets.",
    )

    dataset.add_files(path=DATASET_FOLDER)

    dataset.upload()
    dataset.finalize()
