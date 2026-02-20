"""Client configuration schemas.

Defines the structure for client-specific configuration that drives
how conversations are analyzed. Each client can customize compliance
policies, quality criteria, risk triggers, and more.
"""

from pydantic import BaseModel, Field


class CompliancePolicy(BaseModel):
    """A single compliance policy to check against conversations."""

    id: str = Field(..., description="Unique policy identifier", examples=["TRAI-001"])
    name: str = Field(
        ..., description="Human-readable policy name", examples=["DND Compliance"]
    )
    description: str = Field(
        ...,
        description="What this policy requires",
        examples=[
            "Agents must not make unsolicited marketing calls to DND-registered customers"
        ],
    )
    severity: str = Field(
        "high",
        description="Default severity when this policy is violated",
        examples=["high", "medium", "low"],
    )


class QualityCriteria(BaseModel):
    """Criteria for evaluating agent quality during conversations."""

    greeting_required: bool = Field(True, description="Agent must greet the customer")
    closing_required: bool = Field(
        True, description="Agent must close the call properly"
    )
    empathy_keywords: list[str] = Field(
        default=["understand", "sorry", "apologize", "appreciate", "thank you"],
        description="Keywords indicating empathy from the agent",
    )
    resolution_confirmation_required: bool = Field(
        True, description="Agent must confirm resolution with customer"
    )
    prohibited_phrases: list[str] = Field(
        default=["calm down", "that's not my problem", "nothing I can do"],
        description="Phrases the agent should never use",
    )


class ClientConfiguration(BaseModel):
    """Full client configuration that drives conversation analysis.

    This configuration is injected into the AI prompt dynamically,
    so different clients get different analysis behavior without
    code changes.
    """

    client_id: str = Field(
        ...,
        description="Unique client identifier",
        examples=["jio_telecom"],
        pattern=r"^[a-z0-9_]+$",
    )
    client_name: str = Field(
        ...,
        description="Display name for the client",
        examples=["Jio Telecommunications"],
    )
    domain: str = Field(
        "telecom",
        description="Business domain",
        examples=["telecom", "banking", "insurance"],
    )
    description: str = Field(
        "",
        description="Brief description of this configuration",
    )
    products: list[str] = Field(
        default_factory=list,
        description="Products and services offered by this client",
        examples=[["prepaid", "postpaid", "fiber", "DTH"]],
    )
    compliance_policies: list[CompliancePolicy] = Field(
        default_factory=list,
        description="Compliance policies to check against conversations",
    )
    risk_triggers: list[str] = Field(
        default_factory=list,
        description="Keywords/phrases that indicate risk or escalation",
        examples=[["TRAI complaint", "legal action", "port out", "consumer forum"]],
    )
    quality_criteria: QualityCriteria = Field(
        default_factory=QualityCriteria,
        description="Criteria for evaluating agent quality",
    )
    call_outcome_categories: list[str] = Field(
        default=[
            "resolved",
            "escalated",
            "follow_up_required",
            "transferred",
            "dropped",
            "customer_callback_scheduled",
        ],
        description="Possible call outcome classifications",
    )


class ClientConfigurationCreate(BaseModel):
    """Schema for creating a new client configuration."""

    client_id: str = Field(
        ...,
        description="Unique client identifier (lowercase, underscores only)",
        examples=["jio_telecom"],
        pattern=r"^[a-z0-9_]+$",
    )
    client_name: str = Field(..., examples=["Jio Telecommunications"])
    domain: str = Field("telecom", examples=["telecom", "banking"])
    description: str = Field("")
    products: list[str] = Field(default_factory=list)
    compliance_policies: list[CompliancePolicy] = Field(default_factory=list)
    risk_triggers: list[str] = Field(default_factory=list)
    quality_criteria: QualityCriteria = Field(default_factory=QualityCriteria)
    call_outcome_categories: list[str] = Field(
        default=[
            "resolved",
            "escalated",
            "follow_up_required",
            "transferred",
            "dropped",
            "customer_callback_scheduled",
        ]
    )


class ClientConfigurationUpdate(BaseModel):
    """Schema for updating an existing client configuration.

    All fields are optional — only provided fields are updated.
    """

    client_name: str | None = None
    domain: str | None = None
    description: str | None = None
    products: list[str] | None = None
    compliance_policies: list[CompliancePolicy] | None = None
    risk_triggers: list[str] | None = None
    quality_criteria: QualityCriteria | None = None
    call_outcome_categories: list[str] | None = None


class ClientConfigurationResponse(BaseModel):
    """API response wrapper for client configuration."""

    client_id: str
    client_name: str
    domain: str
    description: str
    products: list[str]
    compliance_policies: list[CompliancePolicy]
    risk_triggers: list[str]
    quality_criteria: QualityCriteria
    call_outcome_categories: list[str]
    created_at: str | None = None
    updated_at: str | None = None
