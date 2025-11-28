FROM ghcr.io/astral-sh/uv:0.4.17-python3.12-bookworm AS builder
WORKDIR /app

COPY pyproject.toml uv.lock ./
# Install only default (non-dev) dependencies; notebook extras are skipped by default
RUN uv sync --frozen --no-install-project --no-dev

# Add source and build wheel
COPY README.md ./README.md
COPY src ./src
RUN uv build --wheel --out-dir /dist

# Runtime stage: minimal Python image, install built wheel
FROM python:3.12-slim AS runner
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

# Install runtime dependencies from built wheel
COPY --from=builder /dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && rm /tmp/*.whl

# Copy Alembic configuration and migrations
COPY alembic.ini ./alembic.ini
COPY migrations ./migrations
COPY entrypoint.sh ./entrypoint.sh
COPY .env.example ./.env

# Run as non-root user for safety
RUN useradd --system --create-home --uid 1000 chromatica
RUN mkdir -p /app/.data /app/.models && chown -R chromatica:chromatica /app
USER chromatica

EXPOSE 8000
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["api"]
