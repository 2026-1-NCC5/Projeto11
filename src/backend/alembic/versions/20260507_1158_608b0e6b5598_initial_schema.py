"""initial schema

Recria o schema completo (extensions + enums + tables + indexes + trigger
updated_at). Espelha src/backend/sql/init.sql. Usado em ambientes novos;
o Supabase atual já tem o schema aplicado e é marcado via `alembic stamp head`.

Revision ID: 608b0e6b5598
Revises:
Create Date: 2026-05-07 11:58:10.198511
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "608b0e6b5598"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")

    food_category = postgresql.ENUM(
        "arroz", "feijao", "acucar", "macarrao", "oleo", "fuba",
        name="food_category",
    )
    user_role = postgresql.ENUM("professor", "aluno", name="user_role")
    period_type = postgresql.ENUM("matutino", "noturno", name="period_type")
    food_category.create(op.get_bind(), checkfirst=True)
    user_role.create(op.get_bind(), checkfirst=True)
    period_type.create(op.get_bind(), checkfirst=True)

    op.create_table(
        "users",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("email", sa.Text(), nullable=False, unique=True),
        sa.Column(
            "role",
            postgresql.ENUM(name="user_role", create_type=False),
            nullable=False,
        ),
        sa.Column("ra", sa.Text(), nullable=False, unique=True),
        sa.Column("full_name", sa.Text(), nullable=False),
        sa.Column("course", sa.Text(), nullable=True),
        sa.Column("semester", sa.SmallInteger(), nullable=True),
        sa.Column(
            "period",
            postgresql.ENUM(name="period_type", create_type=False),
            nullable=False,
        ),
        sa.Column("password_hash", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.CheckConstraint(
            "course IN ('Administração','Ciências Contábeis','Ciências Econômicas')",
            name="users_course_check",
        ),
        sa.CheckConstraint("semester BETWEEN 1 AND 8", name="users_semester_check"),
        sa.CheckConstraint(
            "CASE role "
            "WHEN 'professor' THEN email LIKE '%@fecap.br' AND ra ~ '^\\d{6}$' "
            "WHEN 'aluno' THEN email LIKE '%@edu.fecap.br' AND ra ~ '^\\d{8}$' "
            "END",
            name="users_role_email_ra_check",
        ),
    )
    op.create_index("users_email_idx", "users", ["email"])

    op.create_table(
        "groups",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("name", sa.Text(), nullable=False, unique=True),
        sa.Column(
            "created_by",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )

    op.create_table(
        "group_members",
        sa.Column(
            "group_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("groups.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column(
            "joined_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )

    op.create_table(
        "detection_sessions",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "group_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("groups.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "started_by",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
    )

    op.create_table(
        "evidences",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "session_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("detection_sessions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "group_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("groups.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "category",
            postgresql.ENUM(name="food_category", create_type=False),
            nullable=False,
        ),
        sa.Column("frame_url", sa.Text(), nullable=False),
        sa.Column("confidence", sa.Numeric(4, 3), nullable=False),
        sa.Column("detected_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("dedup_hash", sa.Text(), nullable=False, unique=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.CheckConstraint(
            "confidence >= 0 AND confidence <= 1", name="evidences_confidence_check"
        ),
    )
    op.create_index(
        "evidences_group_detected_idx",
        "evidences",
        ["group_id", sa.text("detected_at DESC")],
    )
    op.create_index("evidences_category_idx", "evidences", ["category"])
    op.create_index("evidences_session_idx", "evidences", ["session_id"])

    op.create_table(
        "refresh_tokens",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("token_hash", sa.Text(), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "revoked", sa.Boolean(), nullable=False, server_default=sa.text("false")
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index(
        "refresh_tokens_user_active_idx", "refresh_tokens", ["user_id", "revoked"]
    )

    op.execute(
        """
        CREATE OR REPLACE FUNCTION public.set_updated_at()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at := NOW();
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
        """
    )
    op.execute(
        """
        DROP TRIGGER IF EXISTS users_set_updated_at ON public.users;
        CREATE TRIGGER users_set_updated_at
            BEFORE UPDATE ON public.users
            FOR EACH ROW EXECUTE FUNCTION public.set_updated_at();
        """
    )


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS users_set_updated_at ON public.users")
    op.execute("DROP FUNCTION IF EXISTS public.set_updated_at()")

    op.drop_index("refresh_tokens_user_active_idx", table_name="refresh_tokens")
    op.drop_table("refresh_tokens")

    op.drop_index("evidences_session_idx", table_name="evidences")
    op.drop_index("evidences_category_idx", table_name="evidences")
    op.drop_index("evidences_group_detected_idx", table_name="evidences")
    op.drop_table("evidences")

    op.drop_table("detection_sessions")
    op.drop_table("group_members")
    op.drop_table("groups")

    op.drop_index("users_email_idx", table_name="users")
    op.drop_table("users")

    op.execute("DROP TYPE IF EXISTS period_type")
    op.execute("DROP TYPE IF EXISTS user_role")
    op.execute("DROP TYPE IF EXISTS food_category")
