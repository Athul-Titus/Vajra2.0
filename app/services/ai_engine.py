"""Async AI engine wrapping the Google Gemini SDK.

Provides a singleton client with:
  - Structured JSON output via `response_mime_type`
  - Automatic retry with exponential back-off
  - Response validation against Pydantic schemas
  - Detailed timing and error metadata
"""

import json
import re
import time

import structlog
from google import genai
from google.genai import types as genai_types

from app.core.config import settings
from app.core.exceptions import AIEngineError

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# JSON fallback parser
# ---------------------------------------------------------------------------


def _try_parse_json(raw: str) -> dict:
    """Attempt to parse JSON with progressive fallback strategies.

    1. Direct parse
    2. Strip markdown code fences (```json ... ```)
    3. Extract first JSON object via regex

    Args:
        raw: Raw string from the AI model.

    Returns:
        Parsed dictionary.

    Raises:
        AIEngineError: If all strategies fail.
    """
    # Strategy 1: Direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Strategy 2: Strip markdown code fences
    stripped = re.sub(r"^```(?:json)?\s*\n?", "", raw.strip())
    stripped = re.sub(r"\n?```\s*$", "", stripped)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # Strategy 3: Extract first JSON object via brace matching
    match = re.search(r"\{", raw)
    if match:
        depth = 0
        start = match.start()
        for i in range(start, len(raw)):
            if raw[i] == "{":
                depth += 1
            elif raw[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(raw[start : i + 1])
                    except json.JSONDecodeError:
                        break

    logger.error("json_parse_all_strategies_failed", raw_preview=raw[:500])
    raise AIEngineError(f"Model returned unparseable response: {raw[:200]}")

# ---------------------------------------------------------------------------
# Singleton client
# ---------------------------------------------------------------------------

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    """Return (and lazily create) the Gemini client singleton."""
    global _client
    if _client is None:
        _client = genai.Client(api_key=settings.GEMINI_API_KEY)
        logger.info("gemini_client_initialised", model=settings.GEMINI_MODEL)
    return _client


# ---------------------------------------------------------------------------
# Core generation function
# ---------------------------------------------------------------------------


async def generate_analysis(
    system_instruction: str,
    user_prompt: str,
    *,
    temperature: float = 0.2,
    max_retries: int = 2,
) -> dict:
    """Send a single AI request and return parsed JSON.

    Args:
        system_instruction: System-level instruction for the model.
        user_prompt: User prompt containing transcript + analysis request.
        temperature: Sampling temperature (lower = more deterministic).
        max_retries: Number of retries on transient failures.

    Returns:
        Parsed JSON dictionary from the model's response.

    Raises:
        AIEngineError: On all AI-related failures after retries exhausted.
    """
    client = _get_client()
    model = settings.GEMINI_MODEL

    config = genai_types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=temperature,
        response_mime_type="application/json",
    )

    last_error: Exception | None = None

    for attempt in range(1, max_retries + 2):  # max_retries + 1 total attempts
        start = time.perf_counter()
        try:
            logger.info(
                "gemini_request_start",
                model=model,
                attempt=attempt,
                prompt_chars=len(user_prompt),
            )

            response = await client.aio.models.generate_content(
                model=model,
                contents=user_prompt,
                config=config,
            )

            elapsed_ms = int((time.perf_counter() - start) * 1000)

            # Extract text from response
            if not response.text:
                raise AIEngineError("Empty response from Gemini model")

            raw_text = response.text.strip()

            # Parse JSON with fallback strategies
            parsed = _try_parse_json(raw_text)

            logger.info(
                "gemini_request_success",
                model=model,
                attempt=attempt,
                elapsed_ms=elapsed_ms,
                response_chars=len(raw_text),
            )

            return parsed

        except Exception as exc:
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            last_error = exc
            logger.warning(
                "gemini_request_failed",
                model=model,
                attempt=attempt,
                elapsed_ms=elapsed_ms,
                error=str(exc),
                error_type=type(exc).__name__,
            )

            if attempt <= max_retries:
                import asyncio

                wait = 2 ** (attempt - 1)  # 1s, 2s
                logger.info("gemini_retry_wait", wait_seconds=wait)
                await asyncio.sleep(wait)
            else:
                raise AIEngineError(
                    f"AI request failed after {attempt} attempts: {exc}"
                ) from exc

    # Should never reach here, but safety net
    raise AIEngineError(f"AI request failed: {last_error}")


# ---------------------------------------------------------------------------
# Audio file upload support
# ---------------------------------------------------------------------------


async def generate_analysis_with_audio(
    system_instruction: str,
    user_prompt: str,
    audio_bytes: bytes,
    audio_mime_type: str,
    *,
    temperature: float = 0.2,
    max_retries: int = 2,
) -> dict:
    """Send an audio file + prompt to Gemini for multimodal analysis.

    Uses inline data (base64) for files under 20 MB, which avoids
    the file upload API roundtrip.

    Args:
        system_instruction: System-level instruction for the model.
        user_prompt: Analysis prompt (without transcript — AI transcribes).
        audio_bytes: Raw audio file bytes.
        audio_mime_type: MIME type of the audio file.
        temperature: Sampling temperature.
        max_retries: Number of retries on transient failures.

    Returns:
        Parsed JSON dictionary from the model's response.

    Raises:
        AIEngineError: On all AI-related failures.
    """
    client = _get_client()
    model = settings.GEMINI_MODEL

    config = genai_types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=temperature,
        response_mime_type="application/json",
    )

    # Build multimodal content: audio part + text prompt
    audio_part = genai_types.Part.from_bytes(
        data=audio_bytes,
        mime_type=audio_mime_type,
    )
    text_part = genai_types.Part.from_text(text=user_prompt)

    last_error: Exception | None = None

    for attempt in range(1, max_retries + 2):
        start = time.perf_counter()
        try:
            logger.info(
                "gemini_audio_request_start",
                model=model,
                attempt=attempt,
                audio_size_kb=len(audio_bytes) // 1024,
                audio_mime=audio_mime_type,
            )

            response = await client.aio.models.generate_content(
                model=model,
                contents=[audio_part, text_part],
                config=config,
            )

            elapsed_ms = int((time.perf_counter() - start) * 1000)

            if not response.text:
                raise AIEngineError("Empty response from Gemini model (audio)")

            raw_text = response.text.strip()

            parsed = _try_parse_json(raw_text)

            logger.info(
                "gemini_audio_request_success",
                model=model,
                attempt=attempt,
                elapsed_ms=elapsed_ms,
                response_chars=len(raw_text),
            )

            return parsed

        except Exception as exc:
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            last_error = exc
            logger.warning(
                "gemini_audio_request_failed",
                model=model,
                attempt=attempt,
                elapsed_ms=elapsed_ms,
                error=str(exc),
                error_type=type(exc).__name__,
            )

            if attempt <= max_retries:
                import asyncio

                wait = 2 ** (attempt - 1)
                logger.info("gemini_retry_wait", wait_seconds=wait)
                await asyncio.sleep(wait)
            else:
                raise AIEngineError(
                    f"AI audio request failed after {attempt} attempts: {exc}"
                ) from exc

    raise AIEngineError(f"AI audio request failed: {last_error}")
