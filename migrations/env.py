"""Alembic environment configuration."""

from __future__ import annotations

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from chromatica.api.config import get_settings
from chromatica.api.server import _apply_dev_defaults

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

_apply_dev_defaults()
get_settings.cache_clear()
try:
    database_url = get_settings().database_url
except Exception as exc:  # noqa: BLE001
    msg = (
        "DATABASE_URL is required for migrations. "
        "Set it explicitly or provide a .env file."
    )
    raise RuntimeError(msg) from exc

config.set_main_option("sqlalchemy.url", database_url)

target_metadata = None


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    context.configure(
        url=database_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
