from __future__ import annotations

import uuid

from fastapi import HTTPException, status
from sqlalchemy import and_, func, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models._enums import UserRole
from app.models.evidence import Evidence
from app.models.group import Group
from app.models.group_member import GroupMember
from app.models.user import User


async def list_groups_admin(session: AsyncSession) -> dict:
    member_count = (
        select(GroupMember.group_id, func.count().label("c"))
        .group_by(GroupMember.group_id)
        .subquery()
    )
    kg_count = (
        select(Evidence.group_id, func.count().label("kg"))
        .group_by(Evidence.group_id)
        .subquery()
    )

    rows = (
        await session.execute(
            select(
                Group,
                func.coalesce(member_count.c.c, 0),
                func.coalesce(kg_count.c.kg, 0),
            )
            .outerjoin(member_count, member_count.c.group_id == Group.id)
            .outerjoin(kg_count, kg_count.c.group_id == Group.id)
            .order_by(Group.created_at.desc())
        )
    ).all()

    groups = [
        {
            "id": g.id,
            "name": g.name,
            "created_at": g.created_at,
            "created_by": g.created_by,
            "member_count": int(c),
            "kg": int(kg),
        }
        for (g, c, kg) in rows
    ]

    total_students = await session.scalar(
        select(func.count(func.distinct(GroupMember.user_id)))
    ) or 0
    complete_groups = sum(1 for g in groups if g["member_count"] >= 4)

    return {
        "groups": groups,
        "total_groups": len(groups),
        "total_students": int(total_students),
        "complete_groups": complete_groups,
    }


async def get_group(session: AsyncSession, group_id: uuid.UUID) -> Group:
    g = await session.scalar(select(Group).where(Group.id == group_id))
    if g is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="grupo não encontrado")
    return g


async def get_group_members(
    session: AsyncSession, group_id: uuid.UUID, leader_id: uuid.UUID
) -> list[dict]:
    rows = (
        await session.execute(
            select(GroupMember.joined_at, User)
            .join(User, User.id == GroupMember.user_id)
            .where(GroupMember.group_id == group_id)
            .order_by(GroupMember.joined_at.asc())
        )
    ).all()
    return [
        {
            "user_id": u.id,
            "full_name": u.full_name,
            "email": u.email,
            "ra": u.ra,
            "course": u.course,
            "semester": u.semester,
            "joined_at": joined_at,
            "is_leader": u.id == leader_id,
        }
        for (joined_at, u) in rows
    ]


async def get_group_kg(session: AsyncSession, group_id: uuid.UUID) -> int:
    return int(
        await session.scalar(
            select(func.count()).select_from(Evidence).where(Evidence.group_id == group_id)
        )
        or 0
    )


async def get_recent_evidences(
    session: AsyncSession, group_id: uuid.UUID, limit: int = 10
) -> list[Evidence]:
    return list(
        (
            await session.execute(
                select(Evidence)
                .where(Evidence.group_id == group_id)
                .order_by(Evidence.detected_at.desc())
                .limit(limit)
            )
        )
        .scalars()
        .all()
    )


async def get_group_for_user(
    session: AsyncSession, user_id: uuid.UUID
) -> Group | None:
    return await session.scalar(
        select(Group)
        .join(GroupMember, GroupMember.group_id == Group.id)
        .where(GroupMember.user_id == user_id)
    )


async def is_user_in_group(
    session: AsyncSession, group_id: uuid.UUID, user_id: uuid.UUID
) -> bool:
    return (
        await session.scalar(
            select(func.count())
            .select_from(GroupMember)
            .where(
                and_(
                    GroupMember.group_id == group_id,
                    GroupMember.user_id == user_id,
                )
            )
        )
    ) > 0


async def _validate_students(
    session: AsyncSession, ids: list[uuid.UUID]
) -> list[User]:
    if not ids:
        return []
    rows = (
        await session.execute(select(User).where(User.id.in_(ids)))
    ).scalars().all()
    found_ids = {u.id for u in rows}
    missing = [str(i) for i in ids if i not in found_ids]
    if missing:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail=f"usuários não encontrados: {', '.join(missing)}",
        )
    not_aluno = [u.full_name for u in rows if u.role != UserRole.aluno]
    if not_aluno:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail=f"apenas alunos podem ser membros: {', '.join(not_aluno)}",
        )
    return list(rows)


async def _ensure_no_existing_group(
    session: AsyncSession, user_ids: list[uuid.UUID]
) -> None:
    if not user_ids:
        return
    rows = (
        await session.execute(
            select(GroupMember.user_id).where(GroupMember.user_id.in_(user_ids))
        )
    ).scalars().all()
    if rows:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            detail=f"{len(rows)} aluno(s) já fazem parte de outro grupo",
        )


async def create_group(
    session: AsyncSession,
    *,
    name: str,
    member_ids: list[uuid.UUID],
    created_by: uuid.UUID,
) -> Group:
    await _validate_students(session, member_ids)
    await _ensure_no_existing_group(session, member_ids)

    group = Group(name=name, created_by=created_by)
    session.add(group)
    try:
        await session.flush()
    except IntegrityError as exc:
        await session.rollback()
        raise HTTPException(
            status.HTTP_409_CONFLICT, detail="já existe um grupo com esse nome"
        ) from exc

    for uid in member_ids:
        session.add(GroupMember(group_id=group.id, user_id=uid))

    try:
        await session.commit()
    except IntegrityError as exc:
        await session.rollback()
        raise HTTPException(
            status.HTTP_409_CONFLICT, detail="conflito ao adicionar membros"
        ) from exc
    await session.refresh(group)
    return group


async def add_member(
    session: AsyncSession, group_id: uuid.UUID, user_id: uuid.UUID
) -> None:
    await get_group(session, group_id)
    await _validate_students(session, [user_id])
    await _ensure_no_existing_group(session, [user_id])
    session.add(GroupMember(group_id=group_id, user_id=user_id))
    try:
        await session.commit()
    except IntegrityError as exc:
        await session.rollback()
        raise HTTPException(
            status.HTTP_409_CONFLICT, detail="aluno já está no grupo"
        ) from exc


async def remove_member(
    session: AsyncSession, group_id: uuid.UUID, user_id: uuid.UUID
) -> None:
    member = await session.scalar(
        select(GroupMember).where(
            and_(
                GroupMember.group_id == group_id,
                GroupMember.user_id == user_id,
            )
        )
    )
    if member is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, detail="aluno não está no grupo"
        )
    await session.delete(member)
    await session.commit()
