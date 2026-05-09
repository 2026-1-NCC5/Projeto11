from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.models._enums import FoodCategory


class EvidenceFeedOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    category: FoodCategory
    detected_at: datetime
    confidence: float
    frame_url: str
    group_id: uuid.UUID


class CategoryCounts(BaseModel):
    arroz: int = 0
    feijao: int = 0
    acucar: int = 0
    macarrao: int = 0
    oleo: int = 0
    fuba: int = 0


class EvidenceAggregateOut(BaseModel):
    counts: CategoryCounts
    total: int


class GroupRankingItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    created_at: datetime
    kg: int
