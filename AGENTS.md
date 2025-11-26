# Repository Guidelines

## Project Structure & Module Organization
- Core code lives in `src/chromatica`: `datasets/` (LAB loaders and transforms), `nn/` (model contracts/loaders), `metrics/` (color-diversity metrics), `charts/` (visualization helpers), plus empty `api/` and `cli/` scaffolds for future services.
- Experiments and exploratory runs sit in `notebooks/` (two U-Net variants and metric tests); keep large outputs out of git.
- Reports, figures, and derived assets belong in `reports/`; scripts for tooling live in `scripts/`.

## Build, Test, and Development Commands
- Install deps with `uv sync --all-extras --all-groups` (Python 3.12 pinned).
- Run lint/format: `uv run ruff format .` then `uv run ruff check . --fix` (config in `pyproject.toml`).
- Type-check if available: `uv run pyright` (uses `.venv` by default).
- Launch notebooks: `./scripts/lab.sh` (opens JupyterLab in Chrome).
- After any sweeping change, run the toolchain without activating the env: `uvx ruff format . && uvx ruff check .` to catch drift early.

## Coding Style & Naming Conventions
- Follow Ruff settings: 88-char lines, double quotes, spaces for indent, numpy-style docstrings, and comprehensive linting (most rules enabled; see `tool.ruff.*`).
- Prefer explicit type hints and torch tensor shapes in docstrings.
- Module/file naming: snake_case for modules and functions; classes in CapWords; avoid introducing non-ASCII unless necessary.

## Testing Guidelines
- No automated test suite yet; add `pytest`-style tests under `tests/` using `test_*.py` and cover new logic (especially data transforms, metrics, and model loaders).
- Keep tests deterministic (set seeds for torch/numpy) and small; prefer CPU-friendly fixtures and synthetic tensors.
- For exploratory checks, place notebooks under `notebooks/<feature>/` with clear names like `metrics_validation.ipynb`.

## Commit & Pull Request Guidelines
- Commit subjects should be auto-prefixed with the issue/ticket from the branch (see `scripts/github_commit_prefix.py`); example: `[#123]: add delta-a metric coverage`.
- Commit in small, reviewable chunks and keep messages imperative.
- PRs should describe intent, key changes, testing performed (commands/notebooks), and any open risks; link issues and attach before/after visuals for charts or qualitative colorization results.

## Security & Data Handling
- Do not commit datasets or generated images; reference local paths or object storage instead.
- Keep secrets and API keys out of the repo and notebooks; rely on environment variables or local `.env` files that stay untracked.
