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
    arroz: float = 0
    feijao: float = 0
    acucar: float = 0
    macarrao: float = 0
    oleo: float = 0
    fuba: float = 0


class EvidenceAggregateOut(BaseModel):
    counts: CategoryCounts
    total: float


class GroupRankingItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    created_at: datetime
    kg: float
