"""API v1 router — aggregates all endpoint routers."""

from fastapi import APIRouter

from app.api.v1.endpoints.configurations import router as configurations_router
from app.api.v1.endpoints.conversations import router as conversations_router
from app.api.v1.endpoints.health import router as health_router

api_v1_router = APIRouter(prefix="/api/v1")

api_v1_router.include_router(health_router)
api_v1_router.include_router(configurations_router)
api_v1_router.include_router(conversations_router)
