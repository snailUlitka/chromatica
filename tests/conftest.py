"""Shared fixtures for API tests."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from chromatica.api.app import ModelRegistry, build_app


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
def api_client(tmp_path: Path) -> Iterator[TestClient]:
    """Provide an API client backed by a temporary registry."""
    registry = ModelRegistry(tmp_path)
    app = build_app(registry)
    with TestClient(app) as client:
        yield client
