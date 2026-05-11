"""add weight_kg to evidences

Revision ID: a3f1c2d4e5b6
Revises: 608b0e6b5598
Create Date: 2026-05-11 17:00:00.000000
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "a3f1c2d4e5b6"
down_revision: str | None = "608b0e6b5598"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "evidences",
        sa.Column("weight_kg", sa.Numeric(), nullable=True),
    )
    op.create_check_constraint(
        "evidences_weight_kg_check",
        "evidences",
        "weight_kg IS NULL OR weight_kg > 0",
    )


def downgrade() -> None:
    op.drop_constraint("evidences_weight_kg_check", "evidences", type_="check")
    op.drop_column("evidences", "weight_kg")
