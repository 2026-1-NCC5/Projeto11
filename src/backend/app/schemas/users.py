from __future__ import annotations

import uuid

from pydantic import BaseModel, ConfigDict


class StudentSearchOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    full_name: str
    email: str
    ra: str
    course: str | None
    semester: int | None
    has_group: bool
