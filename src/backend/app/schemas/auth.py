from __future__ import annotations

import uuid
from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Discriminator, EmailStr, Field, field_validator

from app.core.security import (
    ALUNO_EMAIL_RE,
    ALUNO_RA_RE,
    PROFESSOR_EMAIL_RE,
    PROFESSOR_RA_RE,
    is_strong_password,
)
from app.models._enums import PeriodType, UserRole

CURSOS = ("Administração", "Ciências Contábeis", "Ciências Econômicas")


class _RegisterBase(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    full_name: str = Field(min_length=3, max_length=120)
    password: str = Field(min_length=8, max_length=128)
    period: PeriodType

    @field_validator("password")
    @classmethod
    def _validate_password(cls, v: str) -> str:
        if not is_strong_password(v):
            raise ValueError(
                "senha deve ter 8+ caracteres com maiúscula, minúscula, dígito e símbolo"
            )
        return v


class RegisterProfessorIn(_RegisterBase):
    role: Literal[UserRole.professor] = UserRole.professor
    email: EmailStr
    ra: str

    @field_validator("email")
    @classmethod
    def _validate_email(cls, v: str) -> str:
        if not PROFESSOR_EMAIL_RE.match(v):
            raise ValueError("e-mail de professor deve terminar em @fecap.br")
        return v.lower()

    @field_validator("ra")
    @classmethod
    def _validate_ra(cls, v: str) -> str:
        if not PROFESSOR_RA_RE.match(v):
            raise ValueError("RA de professor deve ter 6 dígitos")
        return v


class RegisterAlunoIn(_RegisterBase):
    role: Literal[UserRole.aluno] = UserRole.aluno
    email: EmailStr
    ra: str
    course: Literal["Administração", "Ciências Contábeis", "Ciências Econômicas"]
    semester: int = Field(ge=1, le=8)

    @field_validator("email")
    @classmethod
    def _validate_email(cls, v: str) -> str:
        if not ALUNO_EMAIL_RE.match(v):
            raise ValueError("e-mail de aluno deve terminar em @edu.fecap.br")
        return v.lower()

    @field_validator("ra")
    @classmethod
    def _validate_ra(cls, v: str) -> str:
        if not ALUNO_RA_RE.match(v):
            raise ValueError("RA de aluno deve ter 8 dígitos (AAMMXXXX)")
        return v


RegisterIn = Annotated[
    RegisterProfessorIn | RegisterAlunoIn,
    Discriminator("role"),
]


class LoginIn(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    email: EmailStr
    password: str

    @field_validator("email")
    @classmethod
    def _normalize_email(cls, v: str) -> str:
        return v.lower()


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    email: EmailStr
    role: UserRole
    ra: str
    full_name: str
    course: str | None
    semester: int | None
    period: PeriodType
    created_at: datetime


class TokenResponse(BaseModel):
    user: UserOut
    access_token: str
    token_type: Literal["bearer"] = "bearer"


class MessageResponse(BaseModel):
    message: str
