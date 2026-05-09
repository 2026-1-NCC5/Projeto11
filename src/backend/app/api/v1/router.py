from fastapi import APIRouter

from app.api.v1 import auth, evidences, groups, users

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(auth.router)
api_router.include_router(groups.router)
api_router.include_router(users.router)
api_router.include_router(evidences.router)
