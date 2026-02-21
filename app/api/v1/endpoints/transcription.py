"""Transcription endpoints.

POST /api/v1/transcribe/audio         — single-speaker transcription
POST /api/v1/transcribe/audio/diarize — multi-speaker with speaker labels

Both use Gemini's File Upload API (no additional dependencies).
Requires X-API-Key header. Supports any language + translate-to-English.
"""

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel

from app.core.config import settings
from app.core.exceptions import AIEngineError, AudioProcessingError
from app.core.security import verify_api_key
from app.services.transcription_service import transcribe_audio, transcribe_with_diarization

router = APIRouter(
    prefix="/transcribe",
    tags=["Transcription"],
    dependencies=[Depends(verify_api_key)],
)

# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class TranscriptionResponse(BaseModel):
    transcript: str
    language_hint: str | None
    task: str
    processing_time_ms: int


class DiarizedSegment(BaseModel):
    speaker: str
    speaker_label: str
    text: str


class DiarizedTranscriptionResponse(BaseModel):
    transcript: str
    language_hint: str | None
    task: str
    segments: list[DiarizedSegment]
    speakers: list[str]
    processing_time_ms: int


# ---------------------------------------------------------------------------
# Shared validation helpers
# ---------------------------------------------------------------------------


def _validate_mime(file: UploadFile) -> str:
    content_type = file.content_type or "application/octet-stream"
    if content_type not in settings.ALLOWED_AUDIO_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported type: '{content_type}'. "
                f"Allowed: {', '.join(settings.ALLOWED_AUDIO_TYPES)}"
            ),
        )
    return content_type


async def _read_audio(file: UploadFile) -> tuple[bytes, str]:
    mime = _validate_mime(file)
    try:
        data = await file.read()
    except Exception as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"Failed to read file: {exc}")
    if len(data) > settings.max_audio_size_bytes:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            f"File exceeds {settings.MAX_AUDIO_SIZE_MB} MB limit",
        )
    return data, mime


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post(
    "/audio",
    response_model=TranscriptionResponse,
    summary="Transcribe audio (single-speaker)",
    description=(
        "Upload an audio file and get back a full transcript. "
        "Set `task=translate` to convert any language (e.g. Malayalam, Hindi) → English. "
        "Powered by **Gemini**'s native multimodal audio understanding. "
        f"Max file size: {settings.MAX_AUDIO_SIZE_MB} MB."
    ),
)
async def transcribe_audio_endpoint(
    file: Annotated[UploadFile, File(description="Audio file (wav/mp3/ogg/webm/m4a)")],
    language: Annotated[
        str | None,
        Form(description="Language hint e.g. 'Malayalam', 'Hindi'. Auto-detected if omitted."),
    ] = None,
    task: Annotated[
        Literal["transcribe", "translate"],
        Form(description="'transcribe' keeps original language. 'translate' → English."),
    ] = "transcribe",
):
    audio_bytes, mime_type = await _read_audio(file)
    try:
        result = await transcribe_audio(
            audio_bytes=audio_bytes,
            mime_type=mime_type,
            language=language or None,
            task=task,
        )
        return result
    except AudioProcessingError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc))
    except AIEngineError as exc:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, str(exc))


@router.post(
    "/audio/diarize",
    response_model=DiarizedTranscriptionResponse,
    summary="Transcribe audio with speaker diarization (multi-speaker)",
    description=(
        "Upload a multi-speaker audio file. Each segment is labelled with a speaker "
        "(User 1, User 2, …). Set `task=translate` for Malayalam/Hindi → English. "
        "Powered by **Gemini**'s native audio + speaker understanding — no HuggingFace token needed. "
        f"Max file size: {settings.MAX_AUDIO_SIZE_MB} MB."
    ),
)
async def diarize_audio_endpoint(
    file: Annotated[UploadFile, File(description="Multi-speaker audio file")],
    language: Annotated[
        str | None,
        Form(description="Language hint e.g. 'Malayalam'. Auto-detected if omitted."),
    ] = None,
    task: Annotated[
        Literal["transcribe", "translate"],
        Form(description="'transcribe' (default) | 'translate' → English"),
    ] = "transcribe",
):
    audio_bytes, mime_type = await _read_audio(file)
    try:
        result = await transcribe_with_diarization(
            audio_bytes=audio_bytes,
            mime_type=mime_type,
            language=language or None,
            task=task,
        )
        return result
    except AudioProcessingError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc))
    except AIEngineError as exc:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, str(exc))
