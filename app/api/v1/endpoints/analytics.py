"""Analytics endpoints — standalone sentiment and risk analysis.

POST /api/v1/analytics/sentiment  → customer + agent sentiment breakdown
POST /api/v1/analytics/risk       → escalation risk score with breakdown

Faster alternatives to the full /analyze/text when you only need
one dimension of analysis (e.g. real-time dashboard widgets).
"""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.core.exceptions import AIEngineError, InvalidInputError
from app.core.security import verify_api_key
from app.services.analytics_service import analyze_risk, analyze_sentiment

router = APIRouter(
    prefix="/analytics",
    tags=["Analytics"],
    dependencies=[Depends(verify_api_key)],
)


# ---------------------------------------------------------------------------
# Shared request schema
# ---------------------------------------------------------------------------


class TranscriptRequest(BaseModel):
    transcript: str = Field(
        ...,
        min_length=10,
        description="Conversation transcript to analyse",
        examples=["Agent: Hello, how can I help?\nCustomer: My internet is down again!"],
    )


# ---------------------------------------------------------------------------
# Sentiment response schemas
# ---------------------------------------------------------------------------


class SpeakerSentimentResult(BaseModel):
    label: str
    confidence: float


class SentimentResponse(BaseModel):
    overall: str
    overall_confidence: float
    customer: SpeakerSentimentResult
    agent: SpeakerSentimentResult
    tone_shift: str | None = None
    key_emotions: list[str] = []
    escalation_risk: bool
    processing_time_ms: int


# ---------------------------------------------------------------------------
# Risk response schemas
# ---------------------------------------------------------------------------


class RiskBreakdown(BaseModel):
    legal_points: int
    churn_points: int
    toxicity_points: int
    resolution_points: int


class RiskResponse(BaseModel):
    escalation_score: int = Field(..., ge=0, le=100)
    risk_category: str = Field(..., description="Low | Monitor | High Risk | Critical")
    risk_level: str = Field(..., description="low | medium | high | critical")
    escalation_reason: str
    breakdown: RiskBreakdown
    triggers_detected: list[str] = []
    recommended_actions: list[str] = []
    processing_time_ms: int


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post(
    "/sentiment",
    response_model=SentimentResponse,
    summary="Sentiment analysis of a conversation transcript",
    description=(
        "Fast focused sentiment analysis — customer + agent emotional state, "
        "overall tone, tone shift direction, and escalation risk flag. "
        "Uses Groq `llama-3.1-8b-instant` (~300 ms)."
    ),
)
async def get_sentiment(request: TranscriptRequest):
    """Return sentiment breakdown for the provided transcript."""
    try:
        return await analyze_sentiment(request.transcript)
    except InvalidInputError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc))
    except AIEngineError as exc:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, str(exc))
    except Exception as exc:
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            f"Sentiment analysis failed: {exc}",
        )


@router.post(
    "/risk",
    response_model=RiskResponse,
    summary="Escalation risk score for a conversation",
    description=(
        "Weighted escalation risk scoring using a 4-factor matrix:\n\n"
        "| Factor | Max Points |\n|---|---|\n"
        "| Legal threat | 40 |\n"
        "| Churn risk | 30 |\n"
        "| Toxicity | 15 |\n"
        "| Resolution failure | 15 |\n\n"
        "**Score ranges:** 0–25 Low · 26–50 Monitor · 51–75 High Risk · 76–100 Critical"
    ),
)
async def get_risk(request: TranscriptRequest):
    """Return escalation risk score and breakdown for the provided transcript."""
    try:
        return await analyze_risk(request.transcript)
    except InvalidInputError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc))
    except AIEngineError as exc:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, str(exc))
    except Exception as exc:
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            f"Risk analysis failed: {exc}",
        )
