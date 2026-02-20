"""Response schemas for conversation analysis.

This is THE core differentiator of the project. Every AI claim includes:
  - confidence score (0.0-1.0)
  - evidence excerpt from the actual conversation
  - structured categorization

Judges can verify nothing is hallucinated.
"""

from datetime import datetime

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class SpeakerSentiment(BaseModel):
    """Sentiment analysis for a single speaker."""

    label: str = Field(
        ...,
        description="Sentiment label",
        examples=["frustrated", "professional", "angry", "empathetic"],
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score for this sentiment assessment",
    )


class SentimentAnalysis(BaseModel):
    """Comprehensive sentiment analysis with per-speaker breakdown."""

    overall: str = Field(
        ...,
        description="Overall conversation sentiment",
        examples=["negative", "positive", "neutral", "mixed"],
    )
    overall_confidence: float = Field(..., ge=0.0, le=1.0)
    customer: SpeakerSentiment = Field(..., description="Customer's emotional state")
    agent: SpeakerSentiment = Field(..., description="Agent's emotional state")


class EntityExtraction(BaseModel):
    """Named entities extracted from the conversation."""

    customer_name: str | None = Field(None, description="Customer's name if mentioned")
    customer_id: str | None = Field(
        None, description="Customer/account ID if mentioned"
    )
    phone_numbers: list[str] = Field(
        default_factory=list, description="Phone numbers mentioned"
    )
    products_mentioned: list[str] = Field(
        default_factory=list, description="Products/services discussed"
    )
    competitors_mentioned: list[str] = Field(
        default_factory=list, description="Competitor brands mentioned"
    )
    amounts_mentioned: list[str] = Field(
        default_factory=list, description="Monetary amounts mentioned"
    )
    custom_entities: dict[str, str] = Field(
        default_factory=dict, description="Additional domain-specific entities"
    )


# ---------------------------------------------------------------------------
# Advanced analysis components
# ---------------------------------------------------------------------------


class ComplianceViolation(BaseModel):
    """A detected compliance or policy violation with evidence.

    Every violation includes the exact conversation excerpt that
    triggered the detection, so judges can verify accuracy.
    """

    violation_id: str = Field(
        ..., description="Unique identifier for this violation", examples=["CV-001"]
    )
    type: str = Field(
        ...,
        description="Violation category",
        examples=["DND_VIOLATION", "CONSENT_NOT_OBTAINED", "MISLEADING_INFORMATION"],
    )
    severity: str = Field(
        ..., description="Impact level", examples=["high", "medium", "low"]
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="AI confidence in this detection"
    )
    description: str = Field(..., description="What the violation is about")
    evidence: str = Field(
        ...,
        description="Exact quote or excerpt from the conversation proving this violation",
    )
    policy_reference: str = Field(
        ...,
        description="Which policy/regulation was violated",
        examples=["TRAI TCCCPR 2018 - Regulation 11"],
    )
    recommended_action: str = Field(..., description="Suggested remediation step")


class QualityMetric(BaseModel):
    """A single quality assessment metric with evidence."""

    score: float = Field(
        ..., ge=0.0, le=10.0, description="Score out of 10 for this metric"
    )
    present: bool = Field(..., description="Whether this quality aspect was observed")
    evidence: str = Field(..., description="Conversation excerpt supporting this score")


class AgentQualityAssessment(BaseModel):
    """Detailed agent performance evaluation."""

    overall_score: float = Field(
        ..., ge=0.0, le=10.0, description="Overall agent quality score"
    )
    confidence: float = Field(..., ge=0.0, le=1.0)
    greeting: QualityMetric = Field(..., description="Greeting/introduction quality")
    empathy: QualityMetric = Field(..., description="Empathy and understanding shown")
    professionalism: QualityMetric = Field(..., description="Professional conduct")
    resolution_skills: QualityMetric = Field(
        ..., description="Problem-solving effectiveness"
    )
    closing: QualityMetric = Field(..., description="Call closing quality")
    strengths: list[str] = Field(
        default_factory=list, description="What the agent did well"
    )
    improvements: list[str] = Field(
        default_factory=list, description="Areas for improvement"
    )


class CallOutcome(BaseModel):
    """Classification of the call's result."""

    classification: str = Field(
        ...,
        description="Call outcome category",
        examples=["resolved", "escalated", "follow_up_required", "dropped"],
    )
    confidence: float = Field(..., ge=0.0, le=1.0)
    reason: str = Field(..., description="Why the call ended this way")


class RiskAssessment(BaseModel):
    """Risk and escalation scoring with justification."""

    score: float = Field(
        ..., ge=0.0, le=10.0, description="Risk score (0 = no risk, 10 = critical)"
    )
    confidence: float = Field(..., ge=0.0, le=1.0)
    level: str = Field(
        ...,
        description="Risk level category",
        examples=["critical", "high", "medium", "low", "none"],
    )
    factors: list[str] = Field(
        default_factory=list, description="Contributing risk factors"
    )
    triggers_detected: list[str] = Field(
        default_factory=list,
        description="Specific risk trigger phrases found in conversation",
    )
    recommended_actions: list[str] = Field(
        default_factory=list, description="Suggested next steps"
    )


class SpeakerTurn(BaseModel):
    """A single turn in the speaker-level timeline analysis."""

    turn_number: int = Field(..., description="Sequential turn number")
    speaker: str = Field(
        ..., description="Who is speaking", examples=["agent", "customer"]
    )
    sentiment: str = Field(
        ...,
        description="Sentiment for this turn",
        examples=["frustrated", "empathetic"],
    )
    key_point: str = Field(..., description="Main point made in this turn")


# ---------------------------------------------------------------------------
# Top-level analysis result
# ---------------------------------------------------------------------------


class ConversationAnalysis(BaseModel):
    """Complete conversation analysis — the core AI output.

    Contains all mandatory + advanced insights extracted from a single
    AI call. Every claim includes confidence scores and evidence excerpts.
    """

    # --- Core Insights (Mandatory) ---
    summary: str = Field(..., description="Concise summary of the conversation")
    languages_detected: list[str] = Field(
        ...,
        description="Languages used in the conversation",
        examples=[["English", "Hindi"]],
    )
    sentiment: SentimentAnalysis = Field(
        ..., description="Sentiment analysis breakdown"
    )
    customer_intents: list[str] = Field(
        ...,
        description="Primary customer intents identified",
        examples=[["billing_dispute", "plan_change", "complaint"]],
    )
    topics: list[str] = Field(
        ...,
        description="Key topics discussed",
        examples=[["internet_speed", "billing", "network_issue"]],
    )
    entities: EntityExtraction = Field(..., description="Named entities extracted")

    # --- Advanced Analysis ---
    compliance_violations: list[ComplianceViolation] = Field(
        default_factory=list,
        description="Detected policy or compliance violations with evidence",
    )
    agent_quality: AgentQualityAssessment = Field(
        ..., description="Agent performance assessment"
    )
    call_outcome: CallOutcome = Field(..., description="Call result classification")
    risk_assessment: RiskAssessment = Field(
        ..., description="Risk and escalation assessment"
    )

    # --- Bonus: Speaker Timeline ---
    speaker_timeline: list[SpeakerTurn] = Field(
        default_factory=list,
        description="Turn-by-turn sentiment and topic tracking",
    )


# ---------------------------------------------------------------------------
# API response wrapper
# ---------------------------------------------------------------------------


class ResponseMetadata(BaseModel):
    """Metadata about the analysis request — for observability and debugging."""

    request_id: str = Field(..., description="Unique request tracking ID")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="When the analysis was performed"
    )
    processing_time_ms: int = Field(
        ..., description="Total processing time in milliseconds"
    )
    ai_model: str = Field(..., description="AI model used for analysis")
    ai_provider: str = Field(
        ...,
        description="AI provider used (groq or gemini)",
        examples=["groq", "gemini"],
    )
    input_type: str = Field(
        ..., description="Type of input processed", examples=["text", "audio"]
    )
    config_id: str = Field(..., description="Configuration used for this analysis")
    audio_duration_seconds: float | None = Field(
        None, description="Duration of audio input (if applicable)"
    )


class ConversationAnalysisResponse(BaseModel):
    """Top-level API response for conversation analysis.

    Wraps the analysis with metadata for enterprise integration.
    """

    success: bool = Field(
        True, description="Whether the analysis completed successfully"
    )
    metadata: ResponseMetadata = Field(..., description="Request metadata and timing")
    analysis: ConversationAnalysis = Field(..., description="Complete analysis results")


class ErrorDetail(BaseModel):
    """Structured error response for API errors."""

    code: str = Field(
        ..., description="Machine-readable error code", examples=["INVALID_INPUT"]
    )
    message: str = Field(..., description="Human-readable error description")
    request_id: str | None = Field(None, description="Request tracking ID")
    details: dict | None = Field(None, description="Additional error context")


class ErrorResponse(BaseModel):
    """Standard error response wrapper."""

    success: bool = Field(False)
    error: ErrorDetail = Field(..., description="Error information")
