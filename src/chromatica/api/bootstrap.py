"""Bootstrap helpers for seeding the API database."""

from __future__ import annotations

import logging
import mimetypes
from pathlib import Path
from typing import TYPE_CHECKING

from sqlalchemy import func, select

from chromatica.api.config import get_settings
from chromatica.api.db import session_scope
from chromatica.api.models import Architecture, Dataset, DatasetImage

if TYPE_CHECKING:
    from collections.abc import Iterable

    from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

DEFAULT_ARCHITECTURES = (
    {
        "code": "u_net_v1",
        "label": "U-Net (v1, no skip connections)",
        "notes": None,
    },
    {
        "code": "u_net_v2",
        "label": "U-Net (v2, skip connections)",
        "notes": None,
    },
)


def _ensure_rows(
    session: Session, model: type[Architecture | Dataset], rows: Iterable[dict]
) -> None:
    for row in rows:
        if session.scalar(select(model).where(model.code == row["code"])):
            continue
        session.add(model(**row))


def ensure_seed_records() -> None:
    """Insert default architectures if they are missing."""
    with session_scope() as session:
        before_count = session.scalar(select(func.count()).select_from(Architecture))
        _ensure_rows(session, Architecture, DEFAULT_ARCHITECTURES)
        session.commit()
        after_count = session.scalar(select(func.count()).select_from(Architecture))
        inserted = (after_count or 0) - (before_count or 0)
        if inserted:
            logger.info("Inserted %s default architectures", inserted)
        else:
            logger.info("Default architectures already present")


def _title_from_code(code: str) -> str:
    titleized = code.replace("_", " ").strip()
    if not titleized:
        return "Dataset"
    return titleized.title()


def sync_datasets_from_disk(dataset_root: Path) -> list[str]:
    """Load datasets from the given root directory into the database.

    Each subdirectory is treated as a dataset code. Files inside are stored in
    the ``dataset_images`` table; duplicates (by filename) are skipped to keep
    the operation idempotent.
    """
    if not dataset_root.exists():
        logger.warning("Dataset root %s does not exist", dataset_root)
        return []

    loaded: list[str] = []
    dataset_dirs = sorted(
        path
        for path in dataset_root.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )
    if not dataset_dirs:
        logger.info("No dataset directories found under %s", dataset_root)
        return loaded

    logger.info(
        "Found %s dataset directories under %s", len(dataset_dirs), dataset_root
    )

    with session_scope() as session:
        for dataset_dir in dataset_dirs:
            code = dataset_dir.name
            dataset = session.scalar(select(Dataset).where(Dataset.code == code))
            if dataset is None:
                dataset = Dataset(code=code, title=_title_from_code(code))
                session.add(dataset)
                session.flush()

            existing_files = set(
                session.scalars(
                    select(DatasetImage.filename).where(
                        DatasetImage.dataset_id == dataset.id
                    )
                ).all()
            )

            added = 0
            for file_path in sorted(dataset_dir.iterdir()):
                if not file_path.is_file() or file_path.name.startswith("."):
                    continue
                if file_path.name in existing_files:
                    continue
                mime_type, _ = mimetypes.guess_type(file_path.name)
                session.add(
                    DatasetImage(
                        dataset_id=dataset.id,
                        filename=file_path.name,
                        mime_type=mime_type,
                        content=file_path.read_bytes(),
                    )
                )
                added += 1
            loaded.append(code)
            logger.info(
                "Dataset %s synced: %s new files (existing: %s)",
                code,
                added,
                len(existing_files),
            )
        session.commit()

    return loaded


def main() -> None:
    """CLI entrypoint used by the container entrypoint."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ensure_seed_records()
    dataset_root = Path(get_settings().datasets_path)
    sync_datasets_from_disk(dataset_root)


if __name__ == "__main__":
    main()
