"""Async AI engine wrapping the Groq SDK.

Provides a singleton client with:
  - Structured JSON output via response_format
  - Automatic retry with exponential back-off
  - Model-agnostic interface (used for both 8B and 70B calls)
  - Detailed timing and error metadata

Designed to be called in parallel via asyncio.gather() from the
conversation analyzer for maximum throughput.
"""

import json
import re
import time

import structlog
from groq import AsyncGroq, APIStatusError, RateLimitError

from app.core.config import settings
from app.core.exceptions import AIEngineError

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# JSON fallback parser (shared logic with Gemini engine)
# ---------------------------------------------------------------------------


def _try_parse_json(raw: str) -> dict:
    """Attempt to parse JSON with progressive fallback strategies.

    1. Direct parse
    2. Strip markdown code fences
    3. Extract first JSON object via regex
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

    logger.error("groq_json_parse_all_strategies_failed", raw_preview=raw[:500])
    raise AIEngineError(f"Groq model returned unparseable response: {raw[:200]}")


# ---------------------------------------------------------------------------
# Singleton client
# ---------------------------------------------------------------------------

_client: AsyncGroq | None = None


def _get_client() -> AsyncGroq:
    """Return (and lazily create) the Groq async client singleton."""
    global _client
    if _client is None:
        if not settings.GROQ_API_KEY:
            raise AIEngineError("GROQ_API_KEY is not configured")
        _client = AsyncGroq(api_key=settings.GROQ_API_KEY)
        logger.info("groq_client_initialised")
    return _client


def is_groq_available() -> bool:
    """Check whether the Groq provider is configured and available."""
    return bool(settings.GROQ_API_KEY)


# ---------------------------------------------------------------------------
# Core generation function
# ---------------------------------------------------------------------------


async def generate_groq_analysis(
    system_instruction: str,
    user_prompt: str,
    *,
    model: str,
    temperature: float = 0.2,
    max_retries: int = 1,
    task_label: str = "groq_analysis",
) -> dict:
    """Send a chat completion request to Groq and return parsed JSON.

    Args:
        system_instruction: System message establishing the AI's role.
        user_prompt: User message with transcript + analysis request.
        model: Groq model ID (e.g. llama-3.1-8b-instant).
        temperature: Sampling temperature (lower = more deterministic).
        max_retries: Number of retries on transient failures.
        task_label: Label for structured logging (extraction / judgment).

    Returns:
        Parsed JSON dictionary from the model's response.

    Raises:
        AIEngineError: On all AI-related failures after retries exhausted.
    """
    client = _get_client()
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 2):
        start = time.perf_counter()
        try:
            logger.info(
                f"{task_label}_request_start",
                model=model,
                attempt=attempt,
                prompt_chars=len(user_prompt),
            )

            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                response_format={"type": "json_object"},
                max_tokens=4096,
            )

            elapsed_ms = int((time.perf_counter() - start) * 1000)

            # Extract text
            raw_text = response.choices[0].message.content
            if not raw_text:
                raise AIEngineError(f"Empty response from Groq model {model}")

            raw_text = raw_text.strip()

            # Parse JSON
            parsed = _try_parse_json(raw_text)

            # Log usage stats
            usage = response.usage
            logger.info(
                f"{task_label}_request_success",
                model=model,
                attempt=attempt,
                elapsed_ms=elapsed_ms,
                response_chars=len(raw_text),
                prompt_tokens=usage.prompt_tokens if usage else None,
                completion_tokens=usage.completion_tokens if usage else None,
                total_tokens=usage.total_tokens if usage else None,
            )

            return parsed

        except RateLimitError as exc:
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            last_error = exc
            logger.warning(
                f"{task_label}_rate_limited",
                model=model,
                attempt=attempt,
                elapsed_ms=elapsed_ms,
                error=str(exc),
            )
            if attempt <= max_retries:
                import asyncio
                wait = 2 ** attempt  # 2s, 4s
                logger.info(f"{task_label}_retry_wait", wait_seconds=wait)
                await asyncio.sleep(wait)
            else:
                raise AIEngineError(
                    f"Groq rate limited after {attempt} attempts: {exc}"
                ) from exc

        except APIStatusError as exc:
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            last_error = exc
            logger.warning(
                f"{task_label}_api_error",
                model=model,
                attempt=attempt,
                elapsed_ms=elapsed_ms,
                status_code=exc.status_code,
                error=str(exc),
            )
            # Don't retry on 4xx client errors (except 429 handled above)
            if 400 <= exc.status_code < 500:
                raise AIEngineError(
                    f"Groq API error ({exc.status_code}): {exc}"
                ) from exc
            # Retry on 5xx
            if attempt <= max_retries:
                import asyncio
                wait = 2 ** attempt
                await asyncio.sleep(wait)
            else:
                raise AIEngineError(
                    f"Groq request failed after {attempt} attempts: {exc}"
                ) from exc

        except AIEngineError:
            raise

        except Exception as exc:
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            last_error = exc
            logger.warning(
                f"{task_label}_request_failed",
                model=model,
                attempt=attempt,
                elapsed_ms=elapsed_ms,
                error=str(exc),
                error_type=type(exc).__name__,
            )
            if attempt <= max_retries:
                import asyncio
                wait = 2 ** (attempt - 1)
                await asyncio.sleep(wait)
            else:
                raise AIEngineError(
                    f"Groq request failed after {attempt} attempts: {exc}"
                ) from exc

    raise AIEngineError(f"Groq request failed: {last_error}")
