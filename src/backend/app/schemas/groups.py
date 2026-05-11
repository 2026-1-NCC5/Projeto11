from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.models._enums import FoodCategory


class GroupSummaryOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    created_at: datetime
    created_by: uuid.UUID
    member_count: int
    kg: float


class GroupListOut(BaseModel):
    groups: list[GroupSummaryOut]
    total_groups: int
    total_students: int
    complete_groups: int


class GroupMemberOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    user_id: uuid.UUID
    full_name: str
    email: str
    ra: str
    course: str | None
    semester: int | None
    joined_at: datetime
    is_leader: bool


class EvidenceBriefOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    category: FoodCategory
    detected_at: datetime
    confidence: float
    frame_url: str


class GroupDetailOut(BaseModel):
    id: uuid.UUID
    name: str
    created_at: datetime
    created_by: uuid.UUID
    members: list[GroupMemberOut]
    kg: float
    recent_evidences: list[EvidenceBriefOut]


class GroupCreateIn(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    name: str = Field(min_length=2, max_length=80)
    member_ids: list[uuid.UUID] = Field(min_length=4, max_length=5)


class GroupMemberAddIn(BaseModel):
    user_id: uuid.UUID
