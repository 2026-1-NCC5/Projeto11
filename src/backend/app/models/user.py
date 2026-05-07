from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import CheckConstraint, DateTime, Index, SmallInteger, Text, text
from sqlalchemy.dialects.postgresql import ENUM as PgEnum
from sqlalchemy.dialects.postgresql import UUID as PgUUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.models._enums import PeriodType, UserRole

user_role_enum = PgEnum(
    UserRole,
    name="user_role",
    create_type=False,
    values_callable=lambda e: [m.value for m in e],
)
period_type_enum = PgEnum(
    PeriodType,
    name="period_type",
    create_type=False,
    values_callable=lambda e: [m.value for m in e],
)


class User(Base):
    __tablename__ = "users"
    __table_args__ = (
        CheckConstraint(
            "course IN ('Administração','Ciências Contábeis','Ciências Econômicas')",
            name="users_course_check",
        ),
        CheckConstraint("semester BETWEEN 1 AND 8", name="users_semester_check"),
        Index("users_email_idx", "email"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    email: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    role: Mapped[UserRole] = mapped_column(user_role_enum, nullable=False)
    ra: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    full_name: Mapped[str] = mapped_column(Text, nullable=False)
    course: Mapped[str | None] = mapped_column(Text, nullable=True)
    semester: Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    period: Mapped[PeriodType] = mapped_column(period_type_enum, nullable=False)
    password_hash: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("now()"), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("now()"), nullable=False
    )
