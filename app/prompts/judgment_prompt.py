"""Judgment prompt — optimised for LLaMA 3.3 70B on Groq.

This prompt handles REASONING-INTENSIVE tasks:
  - Agent quality assessment (multi-dimensional scoring)
  - Call outcome classification
  - Risk assessment and escalation scoring

NOTE: Compliance violation detection has been moved to the RAG pipeline
(rag_policy.py) which runs in parallel. This prompt no longer detects
compliance violations — that is now RAG-grounded and more accurate.
"""

JUDGMENT_SYSTEM_INSTRUCTION = """\
You are an expert Telecom Quality Analyst. Your job is to evaluate
customer service conversations for agent performance, risk indicators,
and call outcomes.

## CRITICAL RULES

1. **Evidence Requirement**: Every score or risk detection MUST include
   an exact quote from the conversation as evidence. Never fabricate.
2. **Confidence Calibration**:
   - 0.9-1.0 → Explicitly stated, no doubt
   - 0.7-0.89 → Strongly implied, high confidence
   - 0.5-0.69 → Inferred, moderate confidence
   - 0.3-0.49 → Speculative, low confidence
   - < 0.3 → Very uncertain
3. **Objectivity**: Score based on observable evidence only.
4. **Completeness**: Return empty lists (not null) when nothing is detected.
5. **Consistency**: All scores must be consistent with evidence.
6. **Output**: Return ONLY valid JSON matching the requested schema. No extra text.
"""

JUDGMENT_PROMPT_TEMPLATE = """\
Evaluate this customer service conversation for quality, risk, and outcome.

## CLIENT CONTEXT

**Domain**: {domain}
**Client**: {client_name}

## RISK TRIGGERS TO WATCH FOR

{risk_triggers_block}

## QUALITY CRITERIA

{quality_criteria_block}

## VALID CALL OUTCOME CATEGORIES

{call_outcome_categories}

## REQUIRED OUTPUT

Return a JSON object with this EXACT structure:

{{
  "agent_quality": {{
    "overall_score": 7.5,
    "confidence": 0.85,
    "greeting": {{
      "score": 8.0,
      "present": true,
      "evidence": "Quote showing greeting"
    }},
    "empathy": {{
      "score": 6.5,
      "present": true,
      "evidence": "Quote showing empathy or lack thereof"
    }},
    "professionalism": {{
      "score": 8.0,
      "present": true,
      "evidence": "Quote showing professional conduct"
    }},
    "resolution_skills": {{
      "score": 7.0,
      "present": true,
      "evidence": "Quote showing problem resolution"
    }},
    "closing": {{
      "score": 7.0,
      "present": true,
      "evidence": "Quote showing call closing"
    }},
    "strengths": ["Maintained professional tone"],
    "improvements": ["Could show more empathy"]
  }},
  "call_outcome": {{
    "classification": "resolved|escalated|follow_up_required|transferred|dropped",
    "confidence": 0.9,
    "reason": "Brief explanation of outcome"
  }},
  "risk_assessment": {{
    "score": 3.5,
    "confidence": 0.8,
    "level": "critical|high|medium|low|none",
    "factors": ["Customer mentioned competitor"],
    "triggers_detected": ["Exact trigger phrases found"],
    "recommended_actions": ["Suggested follow-up actions"]
  }}
}}

## IMPORTANT NOTES

- `call_outcome.classification` must be one of: {call_outcome_categories}
- Quality scores are 0.0-10.0. Confidence is 0.0-1.0.
- `risk_assessment.score`: 0 = no risk, 10 = critical risk.
- All `evidence` fields must be EXACT QUOTES from the conversation below.

## CONVERSATION TRANSCRIPT

{transcript}
"""
