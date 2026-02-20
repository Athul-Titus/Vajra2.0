"""Dynamic prompt builder.

Takes a ClientConfiguration and a transcript (or audio flag) and
produces the final prompt strings ready to send to AI providers.

Supports two modes:
  1. Single-call (Gemini fallback): One prompt for everything.
  2. Split-call (Groq parallel): Extraction prompt (8B) + Judgment prompt (70B).

The builder injects client-specific policies, quality criteria,
and risk triggers directly into the prompt templates so the AI
analyses against the correct ruleset.
"""

from app.models.schemas.configuration import ClientConfiguration
from app.prompts.analysis_prompt import (
    ANALYSIS_PROMPT_TEMPLATE,
    AUDIO_ANALYSIS_ADDENDUM,
    SYSTEM_INSTRUCTION,
)
from app.prompts.extraction_prompt import (
    EXTRACTION_PROMPT_TEMPLATE,
    EXTRACTION_SYSTEM_INSTRUCTION,
)
from app.prompts.judgment_prompt import (
    JUDGMENT_PROMPT_TEMPLATE,
    JUDGMENT_SYSTEM_INSTRUCTION,
)


def _format_compliance_policies(config: ClientConfiguration) -> str:
    """Render compliance policies as a numbered checklist for the AI."""
    if not config.compliance_policies:
        return "No specific compliance policies configured. Perform general compliance review."

    lines: list[str] = []
    for p in config.compliance_policies:
        lines.append(
            f"- **{p.id} — {p.name}** (severity: {p.severity})\n" f"  {p.description}"
        )
    return "\n".join(lines)


def _format_risk_triggers(config: ClientConfiguration) -> str:
    """Render risk triggers as a bullet list."""
    if not config.risk_triggers:
        return "No specific risk triggers configured. Watch for general escalation signals."

    return ", ".join(f'"{t}"' for t in config.risk_triggers)


def _format_quality_criteria(config: ClientConfiguration) -> str:
    """Render quality criteria as a readable block."""
    qc = config.quality_criteria
    lines = [
        f"- Greeting required: {qc.greeting_required}",
        f"- Closing required: {qc.closing_required}",
        f"- Resolution confirmation required: {qc.resolution_confirmation_required}",
        f"- Empathy keywords to look for: {', '.join(qc.empathy_keywords)}",
        f"- Prohibited phrases (agent should NEVER use): {', '.join(qc.prohibited_phrases)}",
    ]
    return "\n".join(lines)


def build_system_instruction() -> str:
    """Return the system instruction prompt for the AI model."""
    return SYSTEM_INSTRUCTION


def build_analysis_prompt(
    config: ClientConfiguration,
    transcript: str,
    *,
    is_audio: bool = False,
) -> str:
    """Build the complete analysis prompt from config + transcript.

    Args:
        config: Client configuration driving analysis behaviour.
        transcript: The conversation text to analyse.
        is_audio: Whether the original input was an audio file
                  (adds audio-specific analysis instructions).

    Returns:
        Fully formatted prompt string ready for the AI model.
    """
    prompt = ANALYSIS_PROMPT_TEMPLATE.format(
        domain=config.domain,
        client_name=config.client_name,
        products=", ".join(config.products) if config.products else "Not specified",
        compliance_policies_block=_format_compliance_policies(config),
        risk_triggers_block=_format_risk_triggers(config),
        quality_criteria_block=_format_quality_criteria(config),
        call_outcome_categories=", ".join(config.call_outcome_categories),
        transcript=transcript,
    )

    if is_audio:
        prompt += AUDIO_ANALYSIS_ADDENDUM

    return prompt


# ---------------------------------------------------------------------------
# Split prompts for Groq parallel architecture
# ---------------------------------------------------------------------------


def build_extraction_system_instruction() -> str:
    """Return the system instruction for the extraction model (8B)."""
    return EXTRACTION_SYSTEM_INSTRUCTION


def build_judgment_system_instruction() -> str:
    """Return the system instruction for the judgment model (70B)."""
    return JUDGMENT_SYSTEM_INSTRUCTION


def build_extraction_prompt(
    config: ClientConfiguration,
    transcript: str,
) -> str:
    """Build the extraction prompt (summary, sentiment, entities, etc.).

    Designed for LLaMA 3.1 8B — fast, factual extraction.
    """
    return EXTRACTION_PROMPT_TEMPLATE.format(
        domain=config.domain,
        client_name=config.client_name,
        products=", ".join(config.products) if config.products else "Not specified",
        transcript=transcript,
    )


def build_judgment_prompt(
    config: ClientConfiguration,
    transcript: str,
) -> str:
    """Build the judgment prompt (compliance, quality, risk, outcome).

    Designed for LLaMA 3.3 70B — deep reasoning and policy matching.
    """
    return JUDGMENT_PROMPT_TEMPLATE.format(
        domain=config.domain,
        client_name=config.client_name,
        compliance_policies_block=_format_compliance_policies(config),
        risk_triggers_block=_format_risk_triggers(config),
        quality_criteria_block=_format_quality_criteria(config),
        call_outcome_categories=", ".join(config.call_outcome_categories),
        transcript=transcript,
    )
