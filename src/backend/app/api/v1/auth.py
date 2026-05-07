from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Body, Cookie, Response, status

from app.core.config import settings
from app.core.deps import CurrentUser, SessionDep
from app.schemas.auth import (
    LoginIn,
    MessageResponse,
    RegisterIn,
    TokenResponse,
    UserOut,
)
from app.services import auth as auth_service

router = APIRouter(prefix="/auth", tags=["auth"])

_ACCESS_COOKIE = "access_token"
_REFRESH_COOKIE = "refresh_token"
_REFRESH_PATH = "/api/v1/auth/refresh"


def _set_auth_cookies(response: Response, access: str, refresh: str) -> None:
    secure = settings.is_production
    response.set_cookie(
        key=_ACCESS_COOKIE,
        value=access,
        max_age=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        httponly=True,
        secure=secure,
        samesite="lax",
        path="/",
    )
    response.set_cookie(
        key=_REFRESH_COOKIE,
        value=refresh,
        max_age=settings.REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60 * 60,
        httponly=True,
        secure=secure,
        samesite="lax",
        path=_REFRESH_PATH,
    )


def _clear_auth_cookies(response: Response) -> None:
    response.delete_cookie(_ACCESS_COOKIE, path="/")
    response.delete_cookie(_REFRESH_COOKIE, path=_REFRESH_PATH)


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(
    payload: Annotated[RegisterIn, Body(discriminator="role")],
    response: Response,
    session: SessionDep,
) -> TokenResponse:
    user = await auth_service.register_user(session, payload)
    access, refresh = await auth_service.issue_token_pair(session, user)
    _set_auth_cookies(response, access, refresh)
    return TokenResponse(user=UserOut.model_validate(user), access_token=access)


@router.post("/login", response_model=TokenResponse)
async def login(
    payload: LoginIn,
    response: Response,
    session: SessionDep,
) -> TokenResponse:
    user = await auth_service.authenticate(session, payload.email, payload.password)
    access, refresh = await auth_service.issue_token_pair(session, user)
    _set_auth_cookies(response, access, refresh)
    return TokenResponse(user=UserOut.model_validate(user), access_token=access)


@router.post("/refresh", response_model=TokenResponse)
async def refresh(
    response: Response,
    session: SessionDep,
    refresh_token: Annotated[str | None, Cookie()] = None,
) -> TokenResponse:
    from fastapi import HTTPException
    from fastapi import status as http_status

    if not refresh_token:
        raise HTTPException(
            status_code=http_status.HTTP_401_UNAUTHORIZED,
            detail="refresh token ausente",
        )
    user, access, new_refresh = await auth_service.rotate_refresh(session, refresh_token)
    _set_auth_cookies(response, access, new_refresh)
    return TokenResponse(user=UserOut.model_validate(user), access_token=access)


@router.post("/logout", response_model=MessageResponse)
async def logout(
    response: Response,
    session: SessionDep,
    refresh_token: Annotated[str | None, Cookie()] = None,
) -> MessageResponse:
    await auth_service.revoke_refresh(session, refresh_token)
    _clear_auth_cookies(response)
    return MessageResponse(message="logout realizado")


@router.get("/me", response_model=UserOut)
async def me(current: CurrentUser) -> UserOut:
    return UserOut.model_validate(current)
