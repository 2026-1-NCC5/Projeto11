"""Smoke test ponta-a-ponta de auth contra o Supabase real.

Cria um usuário aluno temporário, executa register/me/refresh/logout/login
e remove o usuário ao final via SQL direto.

Pula automaticamente se DATABASE_URL não estiver configurado.
"""
from __future__ import annotations

import secrets

import httpx
import pytest
from app.core.config import settings
from app.db.session import AsyncSessionLocal
from app.main import app
from sqlalchemy import text

pytestmark = pytest.mark.skipif(
    not settings.DATABASE_URL,
    reason="precisa de DATABASE_URL apontando para Supabase",
)


def _aluno_payload(email: str, ra: str) -> dict:
    return {
        "role": "aluno",
        "email": email,
        "ra": ra,
        "full_name": "Smoke Test",
        "password": "Senha@123",
        "period": "matutino",
        "course": "Administração",
        "semester": 1,
    }


async def _delete_user(email: str) -> None:
    async with AsyncSessionLocal() as s:
        await s.execute(
            text(
                "DELETE FROM refresh_tokens WHERE user_id IN "
                "(SELECT id FROM users WHERE email = :e)"
            ),
            {"e": email},
        )
        await s.execute(text("DELETE FROM users WHERE email = :e"), {"e": email})
        await s.commit()


@pytest.mark.asyncio
async def test_register_login_me_logout_e2e():
    suffix = secrets.token_hex(3)
    email = f"smoke{suffix}@edu.fecap.br"
    ra = f"2400{secrets.randbelow(10_000):04d}"
    payload = _aluno_payload(email, ra)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver"
    ) as client:
        try:
            r = await client.post("/api/v1/auth/register", json=payload)
            assert r.status_code == 201, r.text
            body = r.json()
            assert body["user"]["email"] == email
            assert body["token_type"] == "bearer"
            assert "access_token" in body
            assert client.cookies.get("access_token")
            assert client.cookies.get("refresh_token")

            r = await client.get("/api/v1/auth/me")
            assert r.status_code == 200, r.text
            assert r.json()["email"] == email

            old_refresh = client.cookies.get("refresh_token")
            r = await client.post("/api/v1/auth/refresh")
            assert r.status_code == 200, r.text
            assert client.cookies.get("refresh_token") != old_refresh

            r = await client.post("/api/v1/auth/logout")
            assert r.status_code == 200, r.text

            r = await client.post(
                "/api/v1/auth/login",
                json={"email": email, "password": payload["password"]},
            )
            assert r.status_code == 200, r.text
        finally:
            await _delete_user(email)
