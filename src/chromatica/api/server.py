"""CLI entrypoint for launching the Chromatica FastAPI app."""

from __future__ import annotations

from os import getenv

import uvicorn


def main() -> None:
    """Start the FastAPI app with uvicorn."""
    host = getenv("API_HOST", "127.0.0.1")
    port = int(getenv("API_PORT", "8000"))
    uvicorn.run("chromatica.api.app:app", host=host, port=port)


if __name__ == "__main__":
    main()
