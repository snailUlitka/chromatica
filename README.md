## Local development

- Install dependencies with `uv sync --all-extras --all-groups`.
- Run the API with auto-reload via `uv run dev`.
  - Defaults to `postgresql://chromatica:chromatica@localhost:5432/chromatica`.
  - Uses local `.data` and `.models` directories under the repo root.
  - Watches `src/chromatica` for changes and restarts automatically.
