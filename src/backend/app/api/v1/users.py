from __future__ import annotations

from fastapi import APIRouter, Query

from app.core.deps import CurrentProfessor, SessionDep
from app.schemas.users import StudentSearchOut
from app.services import users as users_service

router = APIRouter(prefix="/users", tags=["users"])


@router.get("/students", response_model=list[StudentSearchOut])
async def search_students(
    session: SessionDep,
    _: CurrentProfessor,
    q: str = Query(default="", max_length=120),
    limit: int = Query(default=30, ge=1, le=100),
) -> list[StudentSearchOut]:
    rows = await users_service.search_students(session, q=q, limit=limit)
    return [StudentSearchOut(**r) for r in rows]
