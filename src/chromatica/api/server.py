"""CLI entrypoint for launching the Chromatica FastAPI app."""

from __future__ import annotations

import os
from pathlib import Path

import uvicorn

from chromatica.api.config import get_settings

DEV_DEFAULTS = {
    "API_HOST": "127.0.0.1",
    "API_PORT": "8000",
}


def _apply_dev_defaults() -> None:
    """Set sensible local defaults without overriding provided env vars."""
    base_path = Path.cwd()
    defaults = {
        **DEV_DEFAULTS,
        "DATABASE_URL": (
            "postgresql://chromatica:chromatica@localhost:5432/chromatica"
        ),
        "DATASETS_PATH": str(base_path / ".data"),
        "MODEL_STORE_PATH": str(base_path / ".models"),
    }

    for key, value in defaults.items():
        os.environ.setdefault(key, value)

    Path(os.environ["DATASETS_PATH"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["MODEL_STORE_PATH"]).mkdir(parents=True, exist_ok=True)


def main() -> None:
    """Start the FastAPI app with uvicorn."""
    settings = get_settings()
    host = settings.api_host
    port = settings.api_port
    uvicorn.run("chromatica.api.app:app", host=host, port=port)


def dev() -> None:
    """Start the FastAPI app with auto-reload and local defaults."""
    _apply_dev_defaults()
    get_settings.cache_clear()
    settings = get_settings()
    uvicorn.run(
        "chromatica.api.app:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        reload_dirs=[str(Path(__file__).parents[2])],
    )


if __name__ == "__main__":
    main()
