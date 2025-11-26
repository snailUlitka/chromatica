"""SQLAlchemy models for Chromatica API."""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    CheckConstraint,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from chromatica.api.db import Base


class Status(enum.StrEnum):
    """Lifecycle status for trained models."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class Dataset(Base):
    """Dataset entry with optional bundled images."""

    __tablename__ = "datasets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    code: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow
    )

    images: Mapped[list[DatasetImage]] = relationship(
        "DatasetImage", back_populates="dataset", cascade="all, delete-orphan"
    )
    models: Mapped[list[TrainedModel]] = relationship(
        "TrainedModel", back_populates="dataset"
    )


class Architecture(Base):
    """Model architecture definitions."""

    __tablename__ = "architectures"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    code: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    label: Mapped[str] = mapped_column(String(255), nullable=False)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow
    )

    models: Mapped[list[TrainedModel]] = relationship(
        "TrainedModel", back_populates="architecture"
    )


class TrainedModel(Base):
    """Stored trained model metadata and optional binary weights."""

    __tablename__ = "trained_models"
    __table_args__ = (
        CheckConstraint(
            "status in ('running','completed','failed')",
            name="ck_trained_models_status",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    dataset_id: Mapped[int] = mapped_column(
        ForeignKey("datasets.id"), nullable=False, index=True
    )
    architecture_id: Mapped[int] = mapped_column(
        ForeignKey("architectures.id"), nullable=False, index=True
    )
    status: Mapped[Status] = mapped_column(
        Enum(Status), default=Status.RUNNING, nullable=False
    )
    model_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    weights_blob: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    dataset: Mapped[Dataset] = relationship("Dataset", back_populates="models")
    architecture: Mapped[Architecture] = relationship(
        "Architecture", back_populates="models"
    )
    metrics: Mapped[TrainingMetrics] = relationship(
        "TrainingMetrics",
        back_populates="model",
        uselist=False,
        cascade="all, delete-orphan",
    )


class TrainingMetrics(Base):
    """One-to-one metrics attached to a trained model."""

    __tablename__ = "training_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_id: Mapped[str] = mapped_column(
        ForeignKey("trained_models.id", ondelete="CASCADE"), unique=True
    )
    train_error: Mapped[float] = mapped_column(Float, nullable=False)
    test_error: Mapped[float] = mapped_column(Float, nullable=False)
    history: Mapped[Any | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow
    )

    model: Mapped[TrainedModel] = relationship("TrainedModel", back_populates="metrics")


class DatasetImage(Base):
    """Binary images bundled with a dataset entry."""

    __tablename__ = "dataset_images"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_id: Mapped[int] = mapped_column(
        ForeignKey("datasets.id", ondelete="CASCADE"), index=True
    )
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    mime_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    content: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow
    )

    dataset: Mapped[Dataset] = relationship("Dataset", back_populates="images")
