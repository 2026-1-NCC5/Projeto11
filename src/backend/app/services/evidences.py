from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import case, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models._enums import FoodCategory
from app.models.evidence import Evidence
from app.models.group import Group
from app.models.group_member import GroupMember


async def list_evidences(
    session: AsyncSession,
    *,
    group_id: uuid.UUID | None,
    category: FoodCategory | None,
    since: datetime | None,
    limit: int,
) -> list[Evidence]:
    stmt = select(Evidence)
    if group_id is not None:
        stmt = stmt.where(Evidence.group_id == group_id)
    if category is not None:
        stmt = stmt.where(Evidence.category == category)
    if since is not None:
        stmt = stmt.where(Evidence.detected_at >= since)
    stmt = stmt.order_by(Evidence.detected_at.desc()).limit(limit)
    return list((await session.execute(stmt)).scalars().all())


async def aggregate_by_category(
    session: AsyncSession,
    *,
    group_id: uuid.UUID | None,
    since: datetime | None = None,
) -> dict:
    weight_expr = func.coalesce(Evidence.weight_kg, 1)
    counts_expr = {
        c: func.sum(case((Evidence.category == c, weight_expr), else_=0))
        for c in FoodCategory
    }
    stmt = select(*counts_expr.values())
    if group_id is not None:
        stmt = stmt.where(Evidence.group_id == group_id)
    if since is not None:
        stmt = stmt.where(Evidence.detected_at >= since)
    row = (await session.execute(stmt)).one()
    counts = {c.value: float(v or 0) for c, v in zip(counts_expr.keys(), row, strict=True)}
    return {"counts": counts, "total": sum(counts.values())}


async def list_groups_ranking(
    session: AsyncSession,
    *,
    course: str | None = None,
    since: datetime | None = None,
) -> list[dict]:
    from app.models.group_member import GroupMember
    from app.models.user import User

    kg_stmt = select(
        Evidence.group_id,
        func.sum(func.coalesce(Evidence.weight_kg, 1)).label("kg"),
    )
    if since is not None:
        kg_stmt = kg_stmt.where(Evidence.detected_at >= since)
    kg_count = kg_stmt.group_by(Evidence.group_id).subquery()

    base = (
        select(Group, func.coalesce(kg_count.c.kg, 0))
        .outerjoin(kg_count, kg_count.c.group_id == Group.id)
    )
    if course is not None:
        base = base.where(
            Group.id.in_(
                select(GroupMember.group_id)
                .join(User, User.id == GroupMember.user_id)
                .where(User.course == course)
            )
        )
    base = base.order_by(func.coalesce(kg_count.c.kg, 0).desc(), Group.name.asc())

    rows = (await session.execute(base)).all()
    return [
        {"id": g.id, "name": g.name, "created_at": g.created_at, "kg": float(kg)}
        for (g, kg) in rows
    ]


async def get_user_group_id(
    session: AsyncSession, user_id: uuid.UUID
) -> uuid.UUID | None:
    return await session.scalar(
        select(GroupMember.group_id).where(GroupMember.user_id == user_id)
    )
