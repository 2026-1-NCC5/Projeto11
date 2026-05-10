from __future__ import annotations

import uuid
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query, status  # noqa: F401

from app.core.deps import CurrentUser, SessionDep
from app.models._enums import FoodCategory, UserRole
from app.schemas.evidences import (
    CategoryCounts,
    EvidenceAggregateOut,
    EvidenceFeedOut,
)
from app.services import evidences as evidences_service

router = APIRouter(prefix="/evidences", tags=["evidences"])


async def _resolve_group_id(
    session, current, requested: uuid.UUID | None
) -> tuple[uuid.UUID | None, bool]:
    """Return (group_id_filter, has_access). has_access=False means caller has
    no data to see (e.g. aluno without group) — endpoint should respond empty."""
    if current.role == UserRole.aluno:
        own = await evidences_service.get_user_group_id(session, current.id)
        if own is None:
            return (None, False)
        if requested is not None and requested != own:
            raise HTTPException(
                status.HTTP_403_FORBIDDEN,
                detail="aluno só pode ver evidências do próprio grupo",
            )
        return (own, True)
    return (requested, True)


@router.get("", response_model=list[EvidenceFeedOut])
async def list_feed(
    session: SessionDep,
    current: CurrentUser,
    group_id: uuid.UUID | None = None,
    category: FoodCategory | None = None,
    since: datetime | None = None,
    limit: int = Query(default=200, ge=1, le=1000),
) -> list[EvidenceFeedOut]:
    effective_group, has_access = await _resolve_group_id(session, current, group_id)
    if not has_access:
        return []
    rows = await evidences_service.list_evidences(
        session,
        group_id=effective_group,
        category=category,
        since=since,
        limit=limit,
    )
    return [EvidenceFeedOut.model_validate(r) for r in rows]


@router.get("/aggregate", response_model=EvidenceAggregateOut)
async def aggregate(
    session: SessionDep,
    current: CurrentUser,
    group_id: uuid.UUID | None = None,
    since: datetime | None = None,
) -> EvidenceAggregateOut:
    effective_group, has_access = await _resolve_group_id(session, current, group_id)
    if not has_access:
        return EvidenceAggregateOut(counts=CategoryCounts(), total=0)
    data = await evidences_service.aggregate_by_category(
        session, group_id=effective_group, since=since,
    )
    return EvidenceAggregateOut(
        counts=CategoryCounts(**data["counts"]),
        total=data["total"],
    )
