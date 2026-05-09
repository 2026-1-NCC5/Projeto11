from __future__ import annotations

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models._enums import UserRole
from app.models.group_member import GroupMember
from app.models.user import User


async def search_students(
    session: AsyncSession, q: str, limit: int = 30
) -> list[dict]:
    stmt = select(User).where(User.role == UserRole.aluno)
    needle = q.strip()
    if needle:
        like = f"%{needle.lower()}%"
        stmt = stmt.where(
            or_(
                User.full_name.ilike(like),
                User.email.ilike(like),
                User.ra.ilike(like),
            )
        )
    stmt = stmt.order_by(User.full_name.asc()).limit(limit)
    students = list((await session.execute(stmt)).scalars().all())

    if not students:
        return []

    grouped = set(
        (
            await session.execute(
                select(GroupMember.user_id).where(
                    GroupMember.user_id.in_([s.id for s in students])
                )
            )
        ).scalars().all()
    )

    return [
        {
            "id": u.id,
            "full_name": u.full_name,
            "email": u.email,
            "ra": u.ra,
            "course": u.course,
            "semester": u.semester,
            "has_group": u.id in grouped,
        }
        for u in students
    ]
