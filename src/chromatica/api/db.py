"""Database engine/session setup."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, scoped_session, sessionmaker

from chromatica.api.config import get_settings

Base = declarative_base()

if TYPE_CHECKING:
    from collections.abc import Iterator


def _make_engine():
    return create_engine(get_settings().database_url, pool_pre_ping=True, future=True)


engine = _make_engine()
SessionLocal = scoped_session(
    sessionmaker(bind=engine, autocommit=False, autoflush=False, future=True)
)


@contextmanager
def session_scope() -> Iterator:
    """Provide a transactional scope around a series of operations."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:  # pragma: no cover
        session.rollback()
        raise
    finally:
        session.close()
