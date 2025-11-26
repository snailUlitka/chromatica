"""Initial schema for datasets, architectures, models, metrics, and assets."""

from __future__ import annotations

from typing import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "202503041200"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


datasets_table = sa.table(
    "datasets",
    sa.column("code", sa.String()),
    sa.column("title", sa.String()),
    sa.column("notes", sa.Text()),
)

architectures_table = sa.table(
    "architectures",
    sa.column("code", sa.String()),
    sa.column("label", sa.String()),
    sa.column("notes", sa.Text()),
)


def upgrade() -> None:
    op.create_table(
        "datasets",
        sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("code", sa.String(length=64), nullable=False, unique=True),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
    )

    op.create_table(
        "architectures",
        sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("code", sa.String(length=64), nullable=False, unique=True),
        sa.Column("label", sa.String(length=255), nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
    )

    op.create_table(
        "trained_models",
        sa.Column("id", sa.String(length=36), primary_key=True, nullable=False),
        sa.Column(
            "dataset_id", sa.Integer(), sa.ForeignKey("datasets.id"), nullable=False
        ),
        sa.Column(
            "architecture_id",
            sa.Integer(),
            sa.ForeignKey("architectures.id"),
            nullable=False,
        ),
        sa.Column(
            "status",
            sa.String(length=32),
            nullable=False,
            server_default="running",
        ),
        sa.Column("model_path", sa.Text(), nullable=True),
        sa.Column("weights_blob", sa.LargeBinary(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "status in ('running','completed','failed')",
            name="ck_trained_models_status",
        ),
    )
    op.create_index(
        op.f("ix_trained_models_dataset_arch"),
        "trained_models",
        ["dataset_id", "architecture_id"],
    )

    op.create_table(
        "training_metrics",
        sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column(
            "model_id",
            sa.String(length=36),
            sa.ForeignKey("trained_models.id", ondelete="CASCADE"),
            nullable=False,
            unique=True,
        ),
        sa.Column("train_error", sa.Float(), nullable=False),
        sa.Column("test_error", sa.Float(), nullable=False),
        sa.Column("history", sa.JSON(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
    )

    op.create_table(
        "dataset_images",
        sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column(
            "dataset_id",
            sa.Integer(),
            sa.ForeignKey("datasets.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("filename", sa.String(length=255), nullable=False),
        sa.Column("mime_type", sa.String(length=64), nullable=True),
        sa.Column("content", sa.LargeBinary(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
    )
    op.create_index(
        op.f("ix_dataset_images_dataset_id"),
        "dataset_images",
        ["dataset_id"],
    )

    op.bulk_insert(
        datasets_table,
        [
            {"code": "COCO", "title": "COCO", "notes": None},
            {"code": "FOOD101", "title": "Food-101", "notes": None},
        ],
    )
    op.bulk_insert(
        architectures_table,
        [
            {
                "code": "u_net_v1",
                "label": "U-Net (v1, no skip connections)",
                "notes": None,
            },
            {
                "code": "u_net_v2",
                "label": "U-Net (v2, skip connections)",
                "notes": None,
            },
        ],
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_dataset_images_dataset_id"), table_name="dataset_images")
    op.drop_table("dataset_images")
    op.drop_table("training_metrics")
    op.drop_index(op.f("ix_trained_models_dataset_arch"), table_name="trained_models")
    op.drop_table("trained_models")
    op.drop_table("architectures")
    op.drop_table("datasets")
