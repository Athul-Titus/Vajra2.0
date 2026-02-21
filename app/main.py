"""FastAPI application factory and configuration.

Creates the app with:
  - CORS middleware (permissive for hackathon demo)
  - Database lifecycle (init on startup, dispose on shutdown)
  - Structured logging initialisation
  - Global exception handlers for custom errors
  - API v1 router with all endpoints
"""

import time
import uuid
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.v1.router import api_v1_router
from app.core.config import settings
from app.core.exceptions import (
    AIEngineError,
    AudioProcessingError,
    ConfigurationNotFoundError,
    FileTooLargeError,
    InvalidInputError,
    UnsupportedFileTypeError,
)
from app.core.logging_config import setup_logging
from app.models.database import close_database, init_database

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Application lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events."""
    # Startup
    setup_logging()
    logger.info("app_starting", version=settings.APP_VERSION, env=settings.APP_ENV)
    await init_database()
    logger.info("database_ready")
    yield
    # Shutdown
    await close_database()
    logger.info("app_shutdown")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description=(
            "Multimodal Telecom Conversation Intelligence Backend — "
            "Analyzes voice and text customer interactions using AI "
            "to extract structured, evidence-based insights.\n\n"
            "**Key capabilities:**\n"
            "- Text & audio conversation analysis\n"
            "- TRAI compliance violation detection with evidence\n"
            "- Agent quality scoring across 5 dimensions\n"
            "- Risk assessment with trigger phrase detection\n"
            "- Configurable per-client analysis policies\n\n"
            "All AI outputs include confidence scores and exact "
            "conversation excerpts as evidence."
        ),
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # --- CORS (permissive for hackathon demo) ---
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- Request ID + timing middleware ---
    @app.middleware("http")
    async def add_request_context(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        start = time.perf_counter()

        # Bind request_id to structlog for correlated logging
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)

        response = await call_next(request)

        elapsed_ms = int((time.perf_counter() - start) * 1000)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Processing-Time-Ms"] = str(elapsed_ms)
        return response

    # --- Global exception handlers ---
    @app.exception_handler(ConfigurationNotFoundError)
    async def handle_config_not_found(
        request: Request, exc: ConfigurationNotFoundError
    ):
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={
                "success": False,
                "error": {"code": "CONFIG_NOT_FOUND", "message": str(exc)},
            },
        )

    @app.exception_handler(InvalidInputError)
    async def handle_invalid_input(request: Request, exc: InvalidInputError):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "success": False,
                "error": {"code": "INVALID_INPUT", "message": str(exc)},
            },
        )

    @app.exception_handler(FileTooLargeError)
    async def handle_file_too_large(request: Request, exc: FileTooLargeError):
        return JSONResponse(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            content={
                "success": False,
                "error": {"code": "FILE_TOO_LARGE", "message": str(exc)},
            },
        )

    @app.exception_handler(UnsupportedFileTypeError)
    async def handle_unsupported_type(request: Request, exc: UnsupportedFileTypeError):
        return JSONResponse(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            content={
                "success": False,
                "error": {"code": "UNSUPPORTED_FILE_TYPE", "message": str(exc)},
            },
        )

    @app.exception_handler(AIEngineError)
    async def handle_ai_error(request: Request, exc: AIEngineError):
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "error": {"code": "AI_ENGINE_ERROR", "message": str(exc)},
            },
        )

    @app.exception_handler(AudioProcessingError)
    async def handle_audio_error(request: Request, exc: AudioProcessingError):
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "success": False,
                "error": {"code": "AUDIO_PROCESSING_ERROR", "message": str(exc)},
            },
        )

    # --- Routes ---
    app.include_router(api_v1_router)

    # Mount static files for the frontend portal
    import os
    static_path = os.path.join(os.path.dirname(__file__), "static")
    if not os.path.exists(static_path):
        os.makedirs(static_path)
    app.mount("/static", StaticFiles(directory=static_path), name="static")

    @app.get("/", include_in_schema=False)
    async def root():
        index_file = os.path.join(static_path, "index.html")
        if os.path.exists(index_file):
            return FileResponse(index_file)
        return {
            "name": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "docs": "/docs",
            "health": "/api/v1/health",
            "message": "Vajra 2.0 Backend is running. Please add app/static/index.html to see the portal."
        }

    return app


app = create_app()
