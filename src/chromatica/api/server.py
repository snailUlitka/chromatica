"""CLI entrypoint for launching the Chromatica FastAPI app."""

from __future__ import annotations

import uvicorn


def main() -> None:
    """Start the FastAPI app with uvicorn."""
    uvicorn.run("chromatica.api.app:app", host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
