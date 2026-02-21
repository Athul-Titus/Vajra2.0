"""Conversation analysis endpoints.

The core feature — submit text or audio and get structured,
evidence-based intelligence back.
"""

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.core.config import settings
from app.core.exceptions import (
    AIEngineError,
    ConfigurationNotFoundError,
    FileTooLargeError,
    InvalidInputError,
    UnsupportedFileTypeError,
)
from app.core.security import verify_api_key
from app.models.database import get_db_session
from app.models.schemas.requests import TextAnalysisRequest
from app.models.schemas.responses import (
    ConversationAnalysisResponse,
    ErrorResponse,
)
from app.services.conversation_analyzer import analyze_audio, analyze_text

router = APIRouter(
    prefix="/conversations",
    tags=["Conversations"],
    dependencies=[Depends(verify_api_key)],
)


@router.post(
    "/analyze/text",
    response_model=ConversationAnalysisResponse,
    summary="Analyze a text conversation",
    description=(
        "Submit a text transcript for comprehensive AI analysis. "
        "Returns sentiment, compliance violations, quality assessment, "
        "risk scoring, and more — all with confidence scores and evidence."
    ),
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        404: {"model": ErrorResponse, "description": "Configuration not found"},
        500: {"model": ErrorResponse, "description": "AI engine error"},
    },
)
async def analyze_text_conversation(
    request: TextAnalysisRequest,
    db=Depends(get_db_session),
):
    """Analyze a text transcript and return structured insights.

    The analysis is performed in a single AI call that extracts:
    - Summary and language detection
    - Sentiment analysis (overall + per-speaker)
    - Customer intent classification
    - Entity extraction
    - Compliance violation detection with evidence
    - Agent quality assessment (5 dimensions)
    - Call outcome classification
    - Risk assessment with trigger detection
    - Speaker timeline
    """
    try:
        return await analyze_text(
            transcript=request.transcript,
            config_id=request.config_id,
            db_session=db,
        )
    except InvalidInputError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except ConfigurationNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        )
    except AIEngineError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"AI analysis failed: {exc}",
        )


@router.post(
    "/analyze/audio",
    response_model=ConversationAnalysisResponse,
    summary="Analyze an audio conversation",
    description=(
        "Upload an audio file for multimodal AI analysis. "
        "The AI transcribes and analyzes the conversation in a single call. "
        f"Max file size: {settings.MAX_AUDIO_SIZE_MB} MB."
    ),
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        404: {"model": ErrorResponse, "description": "Configuration not found"},
        413: {"model": ErrorResponse, "description": "File too large"},
        415: {"model": ErrorResponse, "description": "Unsupported file type"},
        500: {"model": ErrorResponse, "description": "AI engine error"},
    },
)
async def analyze_audio_conversation(
    file: UploadFile = File(..., description="Audio file to analyze"),
    config_id: str = Form(
        "telecom_default",
        description="Client configuration ID",
        pattern=r"^[a-z0-9_]+$",
    ),
    db=Depends(get_db_session),
):
    """Upload an audio file for comprehensive AI analysis.

    Supports: WAV, MP3, OGG, WebM, M4A, MP4 audio.

    The AI model natively processes the audio — it handles transcription
    and analysis in a single call, leveraging tone and speech patterns
    for richer sentiment and quality insights.
    """
    # Validate content type
    content_type = file.content_type or "application/octet-stream"
    if content_type not in settings.ALLOWED_AUDIO_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported audio type: '{content_type}'. "
                f"Allowed: {', '.join(settings.ALLOWED_AUDIO_TYPES)}"
            ),
        )

    # Read file content
    try:
        audio_bytes = await file.read()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded file: {exc}",
        )

    # Validate size
    if len(audio_bytes) > settings.max_audio_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"File size ({len(audio_bytes) // (1024 * 1024)} MB) "
                f"exceeds maximum of {settings.MAX_AUDIO_SIZE_MB} MB"
            ),
        )

    try:
        return await analyze_audio(
            audio_bytes=audio_bytes,
            audio_mime_type=content_type,
            filename=file.filename or "unknown",
            config_id=config_id,
            db_session=db,
        )
    except ConfigurationNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        )
    except FileTooLargeError as exc:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=str(exc),
        )
    except UnsupportedFileTypeError as exc:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=str(exc),
        )
    except AIEngineError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"AI analysis failed: {exc}",
        )
