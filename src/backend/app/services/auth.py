from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from fastapi import HTTPException, status
from sqlalchemy import or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import (
    create_access_token,
    create_refresh_token,
    hash_password,
    hash_refresh_token,
    new_jti,
    verify_password,
)
from app.models.refresh_token import RefreshToken
from app.models.user import User
from app.schemas.auth import RegisterAlunoIn, RegisterProfessorIn


async def register_user(
    session: AsyncSession,
    payload: RegisterProfessorIn | RegisterAlunoIn,
) -> User:
    existing = await session.scalar(
        select(User).where(or_(User.email == payload.email, User.ra == payload.ra))
    )
    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="e-mail ou RA já cadastrado",
        )

    course = getattr(payload, "course", None)
    semester = getattr(payload, "semester", None)

    user = User(
        email=payload.email,
        role=payload.role,
        ra=payload.ra,
        full_name=payload.full_name,
        course=course,
        semester=semester,
        period=payload.period,
        password_hash=hash_password(payload.password),
    )
    session.add(user)
    try:
        await session.commit()
    except IntegrityError as exc:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="e-mail ou RA já cadastrado",
        ) from exc
    await session.refresh(user)
    return user


async def authenticate(session: AsyncSession, email: str, password: str) -> User:
    user = await session.scalar(select(User).where(User.email == email))
    if user is None or not verify_password(password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="credenciais inválidas",
        )
    return user


async def issue_token_pair(session: AsyncSession, user: User) -> tuple[str, str]:
    access = create_access_token(subject=user.id, role=user.role.value)
    jti = new_jti()
    refresh, expires_at = create_refresh_token(subject=user.id, jti=jti)

    session.add(
        RefreshToken(
            user_id=user.id,
            token_hash=hash_refresh_token(refresh),
            expires_at=expires_at,
        )
    )
    await session.commit()
    return access, refresh


async def rotate_refresh(
    session: AsyncSession, refresh_token: str
) -> tuple[User, str, str]:
    from app.core.security import decode_token  # local import — segurança

    try:
        payload = decode_token(refresh_token)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="refresh token inválido",
        ) from exc

    if payload.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="token de tipo incorreto",
        )

    sub = payload.get("sub")
    if not sub:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="refresh token sem subject",
        )

    token_hash = hash_refresh_token(refresh_token)
    rt = await session.scalar(
        select(RefreshToken).where(RefreshToken.token_hash == token_hash)
    )
    if rt is None or rt.revoked or rt.expires_at <= datetime.now(UTC):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="refresh token expirado ou revogado",
        )

    user = await session.scalar(select(User).where(User.id == UUID(sub)))
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="usuário não encontrado",
        )

    rt.revoked = True
    await session.flush()

    return (user, *await issue_token_pair(session, user))


async def revoke_refresh(session: AsyncSession, refresh_token: str | None) -> None:
    if not refresh_token:
        return
    token_hash = hash_refresh_token(refresh_token)
    rt = await session.scalar(
        select(RefreshToken).where(RefreshToken.token_hash == token_hash)
    )
    if rt is not None and not rt.revoked:
        rt.revoked = True
        await session.commit()
