"""Health check endpoint.

Provides a simple liveness/readiness probe for monitoring and judges.
"""

from fastapi import APIRouter

from app.core.config import settings

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    summary="Health check",
    description="Returns application health status, version, and loaded model.",
)
async def health_check():
    """Liveness probe — always returns 200 if the service is up."""
    return {
        "status": "healthy",
        "version": settings.APP_VERSION,
        "model": settings.GEMINI_MODEL,
        "environment": settings.APP_ENV,
    }
