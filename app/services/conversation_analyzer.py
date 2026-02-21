"""Conversation analyzer — the main orchestration service.

This is the single entry-point for all conversation analysis. It:
  1. Loads the client configuration
  2. Builds the dynamic prompts
  3. Calls the AI engine(s):
     - PRIMARY: Two parallel Groq calls (8B extraction + 70B judgment)
     - FALLBACK: Single Gemini call (if Groq unavailable or fails)
  4. Merges parallel results into a unified analysis
  5. Validates the response against Pydantic schemas
  6. Returns a fully typed ConversationAnalysisResponse

Architecture:
  ┌─────────────────────────────────┐
  │      analyze_text()             │
  │                                 │
  │  ┌── asyncio.gather() ────────┐ │
  │  │                            │ │
  │  │  Groq 8B (extraction)  ──┐ │ │
  │  │  ~300 ms                 │ │ │  → merge → validate → respond
  │  │                          │ │ │
  │  │  Groq 70B (judgment)  ───┘ │ │
  │  │  ~1.5 s                    │ │
  │  └────────────────────────────┘ │
  │                                 │
  │  Gemini fallback (single call)  │
  │  if Groq fails                  │
  └─────────────────────────────────┘
"""

import asyncio
import json
import time
import uuid
from pathlib import Path

import structlog

from app.core.config import settings
from app.core.exceptions import (
    AIEngineError,
    ConfigurationNotFoundError,
    FileTooLargeError,
    InvalidInputError,
    UnsupportedFileTypeError,
)
from app.models.schemas.configuration import ClientConfiguration
from app.models.schemas.responses import (
    AudioFullAnalysisResponse,
    ConversationAnalysis,
    ConversationAnalysisResponse,
    ResponseMetadata,
    TranscriptionResult,
    TranscriptionSegment,
)
from app.services.ai_engine import generate_analysis, generate_analysis_with_audio
from app.services.groq_engine import generate_groq_analysis, is_groq_available
from app.services.prompt_builder import (
    build_analysis_prompt,
    build_extraction_prompt,
    build_extraction_system_instruction,
    build_judgment_prompt,
    build_judgment_system_instruction,
    build_system_instruction,
)

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration cache with TTL
# ---------------------------------------------------------------------------

_CACHE_TTL_SECONDS = 300  # 5-minute TTL

_config_cache: dict[str, tuple[ClientConfiguration, float]] = {}


def _cache_get(config_id: str) -> ClientConfiguration | None:
    """Get a config from cache if it exists and isn't expired."""
    entry = _config_cache.get(config_id)
    if entry is None:
        return None
    config, cached_at = entry
    if time.time() - cached_at > _CACHE_TTL_SECONDS:
        del _config_cache[config_id]
        logger.debug("config_cache_expired", config_id=config_id)
        return None
    return config


def _cache_set(config_id: str, config: ClientConfiguration) -> None:
    """Store a config in cache with current timestamp."""
    _config_cache[config_id] = (config, time.time())


def invalidate_config_cache(config_id: str | None = None) -> None:
    """Invalidate cached configs.

    Args:
        config_id: Specific config to invalidate. If None, clears all.
    """
    if config_id is None:
        _config_cache.clear()
        logger.info("config_cache_cleared_all")
    elif config_id in _config_cache:
        del _config_cache[config_id]
        logger.info("config_cache_invalidated", config_id=config_id)


def _load_default_config(config_id: str) -> ClientConfiguration | None:
    """Load a configuration from the defaults directory (JSON files)."""
    cached = _cache_get(config_id)
    if cached is not None:
        return cached

    config_path = Path(__file__).parent.parent / "configs" / f"{config_id}.json"
    if not config_path.exists():
        return None

    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
        config = ClientConfiguration(**data)
        _cache_set(config_id, config)
        logger.info("config_loaded_from_file", config_id=config_id)
        return config
    except Exception as exc:
        logger.error("config_load_failed", config_id=config_id, error=str(exc))
        return None


async def get_config(config_id: str, db_session=None) -> ClientConfiguration:
    """Resolve a configuration by ID.

    Lookup order:
      1. Database (if db_session provided and row exists)
      2. Default JSON files in app/configs/
      3. Raise ConfigurationNotFoundError

    Args:
        config_id: The configuration identifier.
        db_session: Optional async DB session for database lookup.

    Returns:
        Resolved ClientConfiguration.

    Raises:
        ConfigurationNotFoundError: If no configuration found.
    """
    # Try database first
    if db_session is not None:
        from sqlalchemy import select

        from app.models.database import ClientConfigurationModel

        stmt = select(ClientConfigurationModel).where(
            ClientConfigurationModel.client_id == config_id
        )
        result = await db_session.execute(stmt)
        row = result.scalar_one_or_none()
        if row is not None:
            config = ClientConfiguration(
                client_id=row.client_id,
                client_name=row.client_name,
                domain=row.domain,
                description=row.description,
                **row.config_data,
            )
            _cache_set(config_id, config)
            logger.info("config_loaded_from_db", config_id=config_id)
            return config

    # Try default files
    config = _load_default_config(config_id)
    if config is not None:
        return config

    raise ConfigurationNotFoundError(config_id)


# ---------------------------------------------------------------------------
# Parallel Groq analysis (primary path)
# ---------------------------------------------------------------------------


async def _run_groq_parallel(
    config: ClientConfiguration,
    transcript: str,
) -> tuple[dict, dict, dict]:
    """Run extraction + judgment in parallel on Groq.

    Returns:
        Tuple of (extraction_result, judgment_result, timing_info).

    Raises:
        AIEngineError: If either Groq call fails.
    """
    extraction_sys = build_extraction_system_instruction()
    extraction_prompt = build_extraction_prompt(config, transcript)

    judgment_sys = build_judgment_system_instruction()
    judgment_prompt = build_judgment_prompt(config, transcript)

    start = time.perf_counter()

    extraction_result, judgment_result = await asyncio.gather(
        generate_groq_analysis(
            system_instruction=extraction_sys,
            user_prompt=extraction_prompt,
            model=settings.GROQ_EXTRACTION_MODEL,
            temperature=0.1,
            task_label="extraction",
        ),
        generate_groq_analysis(
            system_instruction=judgment_sys,
            user_prompt=judgment_prompt,
            model=settings.GROQ_JUDGMENT_MODEL,
            temperature=0.2,
            task_label="judgment",
        ),
    )

    elapsed_ms = int((time.perf_counter() - start) * 1000)

    timing = {
        "provider": "groq",
        "parallel_elapsed_ms": elapsed_ms,
        "extraction_model": settings.GROQ_EXTRACTION_MODEL,
        "judgment_model": settings.GROQ_JUDGMENT_MODEL,
    }

    logger.info(
        "groq_parallel_complete",
        elapsed_ms=elapsed_ms,
        extraction_keys=list(extraction_result.keys()),
        judgment_keys=list(judgment_result.keys()),
    )

    return extraction_result, judgment_result, timing


def _merge_results(extraction: dict, judgment: dict) -> dict:
    """Merge extraction (8B) and judgment (70B) into a single dict."""
    merged = {}

    for key in [
        "summary",
        "languages_detected",
        "sentiment",
        "customer_intents",
        "topics",
        "entities",
        "speaker_timeline",
    ]:
        if key in extraction:
            merged[key] = extraction[key]

    for key in [
        "agent_quality",
        "call_outcome",
        "risk_assessment",
    ]:
        if key in judgment:
            merged[key] = judgment[key]

    return merged


# ---------------------------------------------------------------------------
# Gemini fallback (single call)
# ---------------------------------------------------------------------------


async def _run_gemini_fallback(
    config: ClientConfiguration,
    transcript: str,
) -> tuple[dict, dict]:
    """Run single-call analysis on Gemini as a fallback."""
    system_instruction = build_system_instruction()
    analysis_prompt = build_analysis_prompt(config, transcript, is_audio=False)

    start = time.perf_counter()

    raw_result = await generate_analysis(
        system_instruction=system_instruction,
        user_prompt=analysis_prompt,
    )

    elapsed_ms = int((time.perf_counter() - start) * 1000)
    timing = {"provider": "gemini", "elapsed_ms": elapsed_ms, "model": settings.GEMINI_MODEL}

    logger.info("gemini_fallback_complete", elapsed_ms=elapsed_ms)
    return raw_result, timing


# ---------------------------------------------------------------------------
# Text analysis
# ---------------------------------------------------------------------------


async def analyze_text(
    transcript: str,
    config_id: str = "telecom_default",
    db_session=None,
) -> ConversationAnalysisResponse:
    """Analyze a text transcript and return structured insights.

    Strategy:
      1. If Groq is configured → run parallel 8B + 70B calls → merge
      2. If Groq fails or not configured → fall back to single Gemini call

    Args:
        transcript: The conversation transcript to analyze.
        config_id: Configuration ID driving analysis behaviour.
        db_session: Optional DB session for config lookup.

    Returns:
        Complete ConversationAnalysisResponse with metadata.

    Raises:
        InvalidInputError: If transcript is empty or too long.
        ConfigurationNotFoundError: If config_id doesn't resolve.
        AIEngineError: If all AI providers fail.
    """
    request_id = str(uuid.uuid4())
    start_time = time.perf_counter()

    logger.info(
        "text_analysis_start",
        request_id=request_id,
        config_id=config_id,
        transcript_chars=len(transcript),
        groq_available=is_groq_available(),
    )

    # Validate input
    if not transcript or not transcript.strip():
        raise InvalidInputError("Transcript cannot be empty")
    if len(transcript) > settings.MAX_TRANSCRIPT_LENGTH:
        raise InvalidInputError(
            f"Transcript exceeds maximum length of {settings.MAX_TRANSCRIPT_LENGTH} characters"
        )

    # Load configuration
    config = await get_config(config_id, db_session)

    # --- AI call strategy ---
    raw_result: dict
    ai_model_used: str
    provider_used: str

    if is_groq_available():
        try:
            from app.services.rag_policy import detect_compliance_violations

            # Run Groq parallel + RAG compliance all concurrently
            # gather returns [groq_result, rag_result] — 2 items
            groq_result, rag_result = await asyncio.gather(
                _run_groq_parallel(config, transcript),
                detect_compliance_violations(transcript),
            )
            # groq_result is a 3-tuple: (extraction_dict, judgment_dict, timing_dict)
            extraction_result, judgment_result, timing = groq_result
            raw_result = _merge_results(extraction_result, judgment_result)
            # Inject RAG violations (replace any LLM-generated ones)
            raw_result["compliance_violations"] = rag_result.get(
                "compliance_violations", []
            )
            ai_model_used = (
                f"{settings.GROQ_EXTRACTION_MODEL} + {settings.GROQ_JUDGMENT_MODEL}"
            )
            provider_used = "groq"

            logger.info(
                "groq_rag_parallel_success",
                request_id=request_id,
                parallel_ms=timing["parallel_elapsed_ms"],
                rag_violations=len(raw_result["compliance_violations"]),
                rag_policies_retrieved=rag_result.get("retrieved_policies", []),
            )


        except Exception as groq_exc:
            logger.warning(
                "groq_failed_falling_back_to_gemini",
                request_id=request_id,
                groq_error=str(groq_exc),
                groq_error_type=type(groq_exc).__name__,
            )

            if settings.GEMINI_API_KEY:
                raw_result, timing = await _run_gemini_fallback(config, transcript)
                ai_model_used = settings.GEMINI_MODEL
                provider_used = "gemini"
            else:
                raise AIEngineError(
                    f"Groq failed and Gemini not configured: {groq_exc}"
                ) from groq_exc
    else:
        if settings.GEMINI_API_KEY:
            raw_result, timing = await _run_gemini_fallback(config, transcript)
            ai_model_used = settings.GEMINI_MODEL
            provider_used = "gemini"
        else:
            raise AIEngineError(
                "No AI provider configured. Set GROQ_API_KEY or GEMINI_API_KEY."
            )

    # Validate response against Pydantic schema
    try:
        analysis = ConversationAnalysis(**raw_result)
    except Exception as exc:
        logger.error(
            "response_validation_failed",
            request_id=request_id,
            provider=provider_used,
            error=str(exc),
            raw_keys=list(raw_result.keys()) if isinstance(raw_result, dict) else None,
        )
        raise AIEngineError(
            f"AI response did not match expected schema: {exc}"
        ) from exc

    # Build response with metadata
    elapsed_ms = int((time.perf_counter() - start_time) * 1000)

    metadata = ResponseMetadata(
        request_id=request_id,
        processing_time_ms=elapsed_ms,
        ai_model=ai_model_used,
        ai_provider=provider_used,
        input_type="text",
        config_id=config_id,
    )

    response = ConversationAnalysisResponse(
        success=True,
        metadata=metadata,
        analysis=analysis,
    )

    logger.info(
        "text_analysis_complete",
        request_id=request_id,
        provider=provider_used,
        elapsed_ms=elapsed_ms,
        violations_found=len(analysis.compliance_violations),
        risk_level=analysis.risk_assessment.level,
        agent_score=analysis.agent_quality.overall_score,
    )

    return response


# ---------------------------------------------------------------------------
# Audio analysis
# ---------------------------------------------------------------------------


async def analyze_audio(
    audio_bytes: bytes,
    audio_mime_type: str,
    filename: str,
    config_id: str = "telecom_default",
    db_session=None,
) -> ConversationAnalysisResponse:
    """Analyze an audio file and return structured insights.

    Audio always uses Gemini (native multimodal support).
    """
    request_id = str(uuid.uuid4())
    start_time = time.perf_counter()

    logger.info(
        "audio_analysis_start",
        request_id=request_id,
        config_id=config_id,
        filename=filename,
        audio_size_kb=len(audio_bytes) // 1024,
        audio_mime=audio_mime_type,
    )

    # Validate file size
    if len(audio_bytes) > settings.max_audio_size_bytes:
        raise FileTooLargeError(settings.MAX_AUDIO_SIZE_MB)

    # Validate MIME type
    if audio_mime_type not in settings.ALLOWED_AUDIO_TYPES:
        raise UnsupportedFileTypeError(audio_mime_type, settings.ALLOWED_AUDIO_TYPES)

    # Load configuration
    config = await get_config(config_id, db_session)

    # Build prompts
    system_instruction = build_system_instruction()
    analysis_prompt = build_analysis_prompt(
        config,
        transcript="[Audio file provided — transcribe and analyze the conversation]",
        is_audio=True,
    )

    # Call Gemini with audio
    raw_result = await generate_analysis_with_audio(
        system_instruction=system_instruction,
        user_prompt=analysis_prompt,
        audio_bytes=audio_bytes,
        audio_mime_type=audio_mime_type,
    )

    # Validate response
    try:
        analysis = ConversationAnalysis(**raw_result)
    except Exception as exc:
        logger.error(
            "audio_response_validation_failed",
            request_id=request_id,
            error=str(exc),
            raw_keys=list(raw_result.keys()) if isinstance(raw_result, dict) else None,
        )
        raise AIEngineError(
            f"AI response did not match expected schema (audio): {exc}"
        ) from exc

    elapsed_ms = int((time.perf_counter() - start_time) * 1000)

    metadata = ResponseMetadata(
        request_id=request_id,
        processing_time_ms=elapsed_ms,
        ai_model=settings.GEMINI_MODEL,
        ai_provider="gemini",
        input_type="audio",
        config_id=config_id,
    )

    response = ConversationAnalysisResponse(
        success=True,
        metadata=metadata,
        analysis=analysis,
    )

    logger.info(
        "audio_analysis_complete",
        request_id=request_id,
        elapsed_ms=elapsed_ms,
        violations_found=len(analysis.compliance_violations),
        risk_level=analysis.risk_assessment.level,
        agent_score=analysis.agent_quality.overall_score,
    )

    return response


# ---------------------------------------------------------------------------
# Unified audio → transcription + analysis  (the main demo pipeline)
# ---------------------------------------------------------------------------


async def analyze_audio_full(
    audio_bytes: bytes,
    audio_mime_type: str,
    filename: str,
    task: str = "transcribe",
    language: str | None = None,
    config_id: str = "telecom_default",
    db_session=None,
) -> AudioFullAnalysisResponse:
    """Transcribe + analyse an audio call in one request.

    Pipeline:
      1. Gemini diarized transcription → 'User 1: ... / User 2: ...'
      2. analyze_text() → Groq parallel 8B+70B (or Gemini fallback)
      3. Return unified AudioFullAnalysisResponse
    """
    from app.services.transcription_service import transcribe_with_diarization

    request_id = str(uuid.uuid4())
    start_time = time.perf_counter()

    logger.info(
        "audio_full_analysis_start",
        request_id=request_id,
        config_id=config_id,
        filename=filename,
        task=task,
        language=language,
        audio_size_kb=len(audio_bytes) // 1024,
    )

    if len(audio_bytes) > settings.max_audio_size_bytes:
        raise FileTooLargeError(settings.MAX_AUDIO_SIZE_MB)
    if audio_mime_type not in settings.ALLOWED_AUDIO_TYPES:
        raise UnsupportedFileTypeError(audio_mime_type, settings.ALLOWED_AUDIO_TYPES)

    # Step 1 — diarized transcription
    transcription_raw = await transcribe_with_diarization(
        audio_bytes=audio_bytes,
        mime_type=audio_mime_type,
        language=language,
        task=task,
    )
    transcript_text = transcription_raw["transcript"]
    transcription_ms = transcription_raw["processing_time_ms"]

    logger.info(
        "audio_full_transcription_done",
        request_id=request_id,
        chars=len(transcript_text),
        speakers=transcription_raw.get("speakers", []),
        transcription_ms=transcription_ms,
    )

    # Step 2 — full analysis on the transcript
    analysis_response = await analyze_text(
        transcript=transcript_text,
        config_id=config_id,
        db_session=db_session,
    )

    total_ms = int((time.perf_counter() - start_time) * 1000)

    segments = [
        TranscriptionSegment(**seg) for seg in transcription_raw.get("segments", [])
    ]
    transcription_result = TranscriptionResult(
        transcript=transcript_text,
        language_hint=transcription_raw.get("language_hint"),
        task=task,
        segments=segments,
        speakers=transcription_raw.get("speakers", []),
        transcription_time_ms=transcription_ms,
    )

    metadata = ResponseMetadata(
        request_id=request_id,
        processing_time_ms=total_ms,
        ai_model=analysis_response.metadata.ai_model,
        ai_provider=analysis_response.metadata.ai_provider,
        input_type="audio_full",
        config_id=config_id,
    )

    logger.info(
        "audio_full_analysis_complete",
        request_id=request_id,
        total_ms=total_ms,
        transcription_ms=transcription_ms,
        analysis_ms=analysis_response.metadata.processing_time_ms,
        violations_found=len(analysis_response.analysis.compliance_violations),
        risk_level=analysis_response.analysis.risk_assessment.level,
        agent_score=analysis_response.analysis.agent_quality.overall_score,
    )

    return AudioFullAnalysisResponse(
        success=True,
        metadata=metadata,
        transcription=transcription_result,
        analysis=analysis_response.analysis,
    )
