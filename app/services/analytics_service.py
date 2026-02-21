"""Analytics service — focused sentiment and risk analysis.

Uses the existing Groq/Gemini engines with smaller, faster prompts
that only extract what's needed rather than running full analysis.

Incorporates the escalation scoring matrix from risk_score/risk_scoring.py:
  - Legal threat:        40 pts
  - Churn risk:          30 pts
  - Toxicity:            15 pts (0/5/8/15)
  - Resolution failure:  15 pts
"""

import time

import structlog

from app.core.config import settings
from app.core.exceptions import AIEngineError, InvalidInputError
from app.services.ai_engine import generate_analysis
from app.services.groq_engine import generate_groq_analysis, is_groq_available

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Sentiment analysis
# ---------------------------------------------------------------------------

_SENTIMENT_SYSTEM = """You are a sentiment analysis expert for telecom customer calls.
Analyze the transcript and return ONLY valid JSON — no markdown, no explanation."""

_SENTIMENT_PROMPT = """Analyze the sentiment of this telecom conversation transcript.

TRANSCRIPT:
{transcript}

Return ONLY this JSON (no extra text):
{{
  "overall": "negative|positive|neutral|mixed",
  "overall_confidence": 0.0,
  "customer": {{
    "label": "frustrated|angry|neutral|satisfied|upset|calm",
    "confidence": 0.0
  }},
  "agent": {{
    "label": "professional|empathetic|rude|neutral|friendly",
    "confidence": 0.0
  }},
  "tone_shift": "none|improving|worsening",
  "key_emotions": ["list", "of", "emotions"],
  "escalation_risk": true
}}"""


async def analyze_sentiment(transcript: str) -> dict:
    """Run a focused sentiment-only analysis on a transcript.

    Much faster than full analysis — single small model call (~300ms).

    Returns:
        dict with overall, customer, agent sentiment + escalation_risk flag
    """
    if not transcript or not transcript.strip():
        raise InvalidInputError("Transcript cannot be empty")

    start = time.perf_counter()
    prompt = _SENTIMENT_PROMPT.format(transcript=transcript[:8000])

    if is_groq_available():
        raw = await generate_groq_analysis(
            system_instruction=_SENTIMENT_SYSTEM,
            user_prompt=prompt,
            model=settings.GROQ_EXTRACTION_MODEL,  # fast 8B model
            temperature=0.1,
            task_label="sentiment",
        )
    elif settings.GEMINI_API_KEY:
        raw = await generate_analysis(
            system_instruction=_SENTIMENT_SYSTEM,
            user_prompt=prompt,
        )
    else:
        raise AIEngineError("No AI provider configured.")

    elapsed_ms = int((time.perf_counter() - start) * 1000)
    raw["processing_time_ms"] = elapsed_ms

    logger.info(
        "sentiment_analysis_complete",
        overall=raw.get("overall"),
        elapsed_ms=elapsed_ms,
    )
    return raw


# ---------------------------------------------------------------------------
# Risk / Escalation scoring
# ---------------------------------------------------------------------------

_RISK_SYSTEM = """You are a risk assessment expert for telecom customer calls.
Use the exact weighted scoring matrix provided. Return ONLY valid JSON."""

_RISK_PROMPT = """Score the escalation risk of this telecom conversation using this exact weighted matrix:

SCORING MATRIX:
- Legal Threat:        40 points (threatens lawsuit, regulators, consumer forum)
- Churn Risk:          30 points (threatens to cancel, mentions competitor)
- Toxicity:            15 points (15=severe abuse, 8=moderate, 5=mild frustration, 0=polite)
- Resolution Failure:  15 points (issue unresolved, demands supervisor, repeat complaint)

Sum = escalation_score (0-100).
Risk category: 0-25=Low, 26-50=Monitor, 51-75=High Risk, 76-100=Critical.

TRANSCRIPT:
{transcript}

Return ONLY this JSON (no extra text):
{{
  "escalation_score": 0,
  "risk_category": "Low|Monitor|High Risk|Critical",
  "risk_level": "low|medium|high|critical",
  "escalation_reason": "brief explanation of scoring",
  "breakdown": {{
    "legal_points": 0,
    "churn_points": 0,
    "toxicity_points": 0,
    "resolution_points": 0
  }},
  "triggers_detected": ["exact trigger phrases"],
  "recommended_actions": ["next steps"]
}}"""


async def analyze_risk(transcript: str) -> dict:
    """Run focused escalation risk scoring on a transcript.

    Uses the weighted matrix from risk_score/risk_scoring.py:
    Legal(40) + Churn(30) + Toxicity(15) + Resolution(15) = 100.

    Returns:
        dict with escalation_score, risk_category, breakdown, triggers
    """
    if not transcript or not transcript.strip():
        raise InvalidInputError("Transcript cannot be empty")

    start = time.perf_counter()
    prompt = _RISK_PROMPT.format(transcript=transcript[:8000])

    if is_groq_available():
        raw = await generate_groq_analysis(
            system_instruction=_RISK_SYSTEM,
            user_prompt=prompt,
            model=settings.GROQ_JUDGMENT_MODEL,  # 70B for better scoring accuracy
            temperature=0.1,
            task_label="risk",
        )
    elif settings.GEMINI_API_KEY:
        raw = await generate_analysis(
            system_instruction=_RISK_SYSTEM,
            user_prompt=prompt,
        )
    else:
        raise AIEngineError("No AI provider configured.")

    elapsed_ms = int((time.perf_counter() - start) * 1000)
    raw["processing_time_ms"] = elapsed_ms

    logger.info(
        "risk_analysis_complete",
        score=raw.get("escalation_score"),
        category=raw.get("risk_category"),
        elapsed_ms=elapsed_ms,
    )
    return raw
