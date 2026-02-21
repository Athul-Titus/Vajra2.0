"""Vajra 2.0 — Unified Pipeline Service.

Orchestrates all analytical stages in order:

  Step 1  → Audio Transcription (if audio input)     [transcription_service.py]
  Step 2  → Sentiment Analysis                        [conversation_analyzer.py]
  Step 3  → Risk Scoring                              [risk_score/risk_scoring.py]
  Step 4  → Conversation Summary + Intents            [summary_test/main.py equivalent]
  Step 5  → Full Conversation Analysis (Groq/Gemini)  [conversation_analyzer.py]

Both text and audio inputs enter the same ordered pipeline.
Audio input goes through Step 1 first; text input starts at Step 2.
"""

import asyncio
import time
import uuid
from typing import Optional

import structlog

from app.core.config import settings
from app.core.exceptions import AIEngineError, AudioProcessingError, InvalidInputError
from app.models.schemas.responses import ConversationAnalysisResponse, ResponseMetadata
from app.services.conversation_analyzer import analyze_text, get_config
from app.services.transcription_service import transcribe_audio_bytes_async

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Step 3: Risk scoring helper (uses risk_score/risk_scoring.py schema + prompt)
# ---------------------------------------------------------------------------


def _compute_risk_score_prompt(transcript: str) -> str:
    """Build the risk scoring prompt from risk_scoring.py instructions."""
    from risk_score.risk_scoring import get_scoring_prompt_instructions

    instructions = get_scoring_prompt_instructions()
    return (
        f"{instructions}\n\n"
        f"Conversation transcript:\n{transcript}\n\n"
        "Respond ONLY with a valid JSON object matching the EscalationAnalysis schema."
    )


# ---------------------------------------------------------------------------
# Unified pipeline entry point
# ---------------------------------------------------------------------------


async def run_pipeline_from_audio(
    audio_bytes: bytes,
    audio_mime_type: str,
    filename: str,
    config_id: str = "telecom_default",
    db_session=None,
) -> dict:
    """Full pipeline for audio input.

    Order:
      1. Transcribe audio → diarized text (Gemini File Upload)
      2. Run full conversation analysis on the transcript (Groq/Gemini)
      3. Return combined result with transcript appended to metadata

    Args:
        audio_bytes: Raw audio file bytes.
        audio_mime_type: MIME type e.g. 'audio/ogg'.
        filename: Original filename for logging.
        config_id: Analysis configuration ID.
        db_session: Optional DB session for config lookup.

    Returns:
        Dict with keys: 'transcript', 'analysis' (ConversationAnalysisResponse).

    Raises:
        AudioProcessingError: If transcription fails.
        AIEngineError: If analysis fails.
    """
    pipeline_id = str(uuid.uuid4())
    logger.info(
        "pipeline_start",
        pipeline_id=pipeline_id,
        input_type="audio",
        filename=filename,
        config_id=config_id,
    )
    start = time.perf_counter()

    # -------------------------------------------------------------------------
    # STEP 1: Audio → Transcript
    # -------------------------------------------------------------------------
    logger.info("pipeline_step1_transcription", pipeline_id=pipeline_id)
    try:
        transcript = await transcribe_audio_bytes_async(
            audio_bytes=audio_bytes,
            audio_mime_type=audio_mime_type,
            api_key=settings.GEMINI_API_KEY,
            model_name="models/gemini-2.5-flash",
        )
    except Exception as exc:
        raise AudioProcessingError(f"Transcription failed: {exc}") from exc

    if not transcript.strip():
        raise AudioProcessingError("Transcription returned empty result.")

    logger.info(
        "pipeline_step1_done",
        pipeline_id=pipeline_id,
        transcript_chars=len(transcript),
    )

    # -------------------------------------------------------------------------
    # STEP 2-5: Analyze transcript (sentiment, risk, summary, full analysis via Groq/Gemini)
    # -------------------------------------------------------------------------
    logger.info("pipeline_step2_analysis", pipeline_id=pipeline_id)
    analysis_response = await run_pipeline_from_text(
        transcript=transcript,
        config_id=config_id,
        db_session=db_session,
    )

    elapsed_ms = int((time.perf_counter() - start) * 1000)
    logger.info(
        "pipeline_complete",
        pipeline_id=pipeline_id,
        elapsed_ms=elapsed_ms,
        input_type="audio",
    )

    return {
        "transcript": transcript,
        "analysis": analysis_response,
    }


async def run_pipeline_from_text(
    transcript: str,
    config_id: str = "telecom_default",
    db_session=None,
) -> ConversationAnalysisResponse:
    """Full pipeline for text input (Steps 2-5).

    Order for text input:
      2. Validate + load config
      3. Run full conversation analysis via Groq (primary) / Gemini (fallback)
         — The AI prompt already includes: sentiment, risk, summary, compliance, quality

    Args:
        transcript: Speaker-labeled text transcript.
        config_id: Analysis configuration ID.
        db_session: Optional DB session.

    Returns:
        ConversationAnalysisResponse with all insights.
    """
    pipeline_id = str(uuid.uuid4())
    logger.info(
        "pipeline_start",
        pipeline_id=pipeline_id,
        input_type="text",
        transcript_chars=len(transcript),
        config_id=config_id,
    )

    if not transcript or not transcript.strip():
        raise InvalidInputError("Transcript is empty.")

    # Steps 2-5 are handled by analyze_text (it already does sentiment, risk, summary, compliance, quality via AI)
    response = await analyze_text(
        transcript=transcript,
        config_id=config_id,
        db_session=db_session,
    )

    logger.info(
        "pipeline_complete",
        pipeline_id=pipeline_id,
        input_type="text",
        violations=len(response.analysis.compliance_violations),
        risk_level=response.analysis.risk_assessment.level,
        agent_score=response.analysis.agent_quality.overall_score,
    )

    return response
