"""CLI entrypoint for launching the Chromatica FastAPI app."""

from __future__ import annotations

import uvicorn

from chromatica.api.config import get_settings


def main() -> None:
    """Start the FastAPI app with uvicorn."""
    settings = get_settings()
    host = settings.api_host
    port = settings.api_port
    uvicorn.run("chromatica.api.app:app", host=host, port=port)


if __name__ == "__main__":
    main()
