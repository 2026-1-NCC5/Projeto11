from __future__ import annotations

import uuid
from datetime import datetime

from fastapi import APIRouter, HTTPException, status

from app.core.deps import CurrentProfessor, CurrentUser, SessionDep
from app.models._enums import UserRole
from app.schemas.evidences import GroupRankingItem
from app.schemas.groups import (
    EvidenceBriefOut,
    GroupCreateIn,
    GroupDetailOut,
    GroupListOut,
    GroupMemberAddIn,
    GroupMemberOut,
    GroupSummaryOut,
)
from app.services import evidences as evidences_service
from app.services import groups as groups_service

router = APIRouter(prefix="/groups", tags=["groups"])


@router.get("", response_model=GroupListOut)
async def list_groups(
    session: SessionDep, _: CurrentProfessor
) -> GroupListOut:
    data = await groups_service.list_groups_admin(session)
    return GroupListOut(
        groups=[GroupSummaryOut(**g) for g in data["groups"]],
        total_groups=data["total_groups"],
        total_students=data["total_students"],
        complete_groups=data["complete_groups"],
    )


@router.post("", response_model=GroupSummaryOut, status_code=status.HTTP_201_CREATED)
async def create_group(
    payload: GroupCreateIn,
    session: SessionDep,
    professor: CurrentProfessor,
) -> GroupSummaryOut:
    group = await groups_service.create_group(
        session,
        name=payload.name,
        member_ids=payload.member_ids,
        created_by=professor.id,
    )
    return GroupSummaryOut(
        id=group.id,
        name=group.name,
        created_at=group.created_at,
        created_by=group.created_by,
        member_count=len(payload.member_ids),
        kg=0,
    )


@router.get("/ranking", response_model=list[GroupRankingItem])
async def list_ranking(
    session: SessionDep,
    _: CurrentUser,
    course: str | None = None,
    since: datetime | None = None,
) -> list[GroupRankingItem]:
    rows = await evidences_service.list_groups_ranking(
        session, course=course, since=since
    )
    return [GroupRankingItem(**r) for r in rows]


@router.get("/me", response_model=GroupDetailOut | None)
async def get_my_group(
    session: SessionDep, current: CurrentUser
) -> GroupDetailOut | None:
    group = await groups_service.get_group_for_user(session, current.id)
    if group is None:
        return None
    members = await groups_service.get_group_members(session, group.id, group.created_by)
    kg = await groups_service.get_group_kg(session, group.id)
    evidences = await groups_service.get_recent_evidences(session, group.id, limit=10)
    return GroupDetailOut(
        id=group.id,
        name=group.name,
        created_at=group.created_at,
        created_by=group.created_by,
        members=[GroupMemberOut(**m) for m in members],
        kg=kg,
        recent_evidences=[EvidenceBriefOut.model_validate(e) for e in evidences],
    )


@router.get("/{group_id}", response_model=GroupDetailOut)
async def get_group_detail(
    group_id: uuid.UUID,
    session: SessionDep,
    current: CurrentUser,
) -> GroupDetailOut:
    group = await groups_service.get_group(session, group_id)
    if current.role != UserRole.professor:
        is_member = await groups_service.is_user_in_group(session, group_id, current.id)
        if not is_member:
            raise HTTPException(
                status.HTTP_403_FORBIDDEN, detail="acesso negado"
            )
    members = await groups_service.get_group_members(session, group.id, group.created_by)
    kg = await groups_service.get_group_kg(session, group.id)
    evidences = await groups_service.get_recent_evidences(session, group.id, limit=10)
    return GroupDetailOut(
        id=group.id,
        name=group.name,
        created_at=group.created_at,
        created_by=group.created_by,
        members=[GroupMemberOut(**m) for m in members],
        kg=kg,
        recent_evidences=[EvidenceBriefOut.model_validate(e) for e in evidences],
    )


@router.post(
    "/{group_id}/members",
    response_model=GroupMemberOut,
    status_code=status.HTTP_201_CREATED,
)
async def add_member(
    group_id: uuid.UUID,
    payload: GroupMemberAddIn,
    session: SessionDep,
    _: CurrentProfessor,
) -> GroupMemberOut:
    await groups_service.add_member(session, group_id, payload.user_id)
    group = await groups_service.get_group(session, group_id)
    members = await groups_service.get_group_members(session, group.id, group.created_by)
    new = next((m for m in members if m["user_id"] == payload.user_id), None)
    if new is None:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail="erro interno")
    return GroupMemberOut(**new)


@router.delete(
    "/{group_id}/members/{user_id}", status_code=status.HTTP_204_NO_CONTENT
)
async def remove_member(
    group_id: uuid.UUID,
    user_id: uuid.UUID,
    session: SessionDep,
    _: CurrentProfessor,
) -> None:
    await groups_service.remove_member(session, group_id, user_id)
