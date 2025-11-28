"""Shared fixtures for API tests."""

from __future__ import annotations

from io import BytesIO
from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from chromatica.api.config import get_settings

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register custom markers and CLI flags."""
    parser.addoption(
        "--runs-continues",
        action="store_true",
        default=False,
        help="Run tests marked as continues (long-running).",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Expose custom marker to pytest."""
    config.addinivalue_line(
        "markers",
        "continues: long-running training/inference tests that are skipped by default.",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip continues tests unless the CLI flag is set."""
    if config.getoption("--runs-continues"):
        return
    skip_continues = pytest.mark.skip(
        reason="Skipping continues tests; enable with --runs-continues."
    )
    for item in items:
        if "continues" in item.keywords:
            item.add_marker(skip_continues)


@pytest.fixture
def api_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """Provide an API client backed by a temporary SQLite database."""
    db_path = tmp_path / "api.sqlite3"
    dataset_root = tmp_path / "datasets"
    dataset_root.mkdir()
    sample_dataset = dataset_root / "demo"
    sample_dataset.mkdir()

    for idx, color in enumerate(((120, 130, 140), (180, 50, 60), (90, 200, 40))):
        image = Image.new("RGB", (32, 32), color)
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        sample_dataset.joinpath(f"demo_{idx}.png").write_bytes(buffer.getvalue())

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("DATASETS_PATH", str(dataset_root))
    monkeypatch.setenv("MODEL_STORE_PATH", str(tmp_path / "models"))
    get_settings.cache_clear()

    # Local imports to apply environment overrides before initialization.
    from chromatica.api import db as db_module  # noqa: PLC0415
    from chromatica.api import models as models_module  # noqa: PLC0415
    from chromatica.api.bootstrap import (  # noqa: PLC0415
        ensure_seed_records,
        sync_datasets_from_disk,
    )

    models_module.Base.metadata.create_all(db_module.engine)
    ensure_seed_records()
    sync_datasets_from_disk(dataset_root)

    from chromatica.api.app import build_app  # noqa: PLC0415

    app = build_app()
    with TestClient(app) as client:
        yield client
