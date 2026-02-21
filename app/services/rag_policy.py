"""RAG pipeline for compliance/policy violation detection.

Architecture:
- Loads policy JSON (documentation/config.json) at startup
- Embeds rules with sentence-transformers + FAISS for vector search
- At runtime: retrieve top-5 relevant policy chunks for the transcript,
  then ask Groq to check which ones are violated (with evidence)
- All sync FAISS/embedding work runs in a thread pool to avoid blocking

This replaces the LLM-only compliance detection in the judgment prompt.
Every violation is grounded in actual policy text — not just LLM knowledge.
"""

import asyncio
import json
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Policy data structures
# ---------------------------------------------------------------------------


@dataclass
class PolicyChunk:
    section: str       # e.g. "data_privacy"
    rule_id: str       # e.g. "data_privacy_1"
    text: str          # The actual policy rule text

    def __repr__(self) -> str:
        return f"[{self.section}] {self.rule_id}: {self.text}"


# ---------------------------------------------------------------------------
# Load + embed index (built once at startup via get_rag_index())
# ---------------------------------------------------------------------------


def _load_policy_chunks(json_path: str) -> list[PolicyChunk]:
    """Parse config.json policy rules into flat list of PolicyChunks."""
    path = Path(json_path)
    if not path.exists():
        logger.warning("policy_json_not_found", path=str(path))
        return []
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    chunks = []
    for section, rules in doc.get("policies_or_rules", {}).items():
        for idx, rule in enumerate(rules, 1):
            chunks.append(PolicyChunk(
                section=section,
                rule_id=f"{section}_{idx}",
                text=rule,
            ))
    logger.info("policy_chunks_loaded", count=len(chunks))
    return chunks


class PolicyRAGIndex:
    """FAISS-backed semantic search over policy rules.

    Built once at startup; all heavy work stays sync (called in thread pool).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        import faiss  # noqa: F401  (ensures it's importable)

        self._model = SentenceTransformer(model_name)
        self._chunks: list[PolicyChunk] = []
        self._index = None

    def build(self, chunks: list[PolicyChunk]) -> None:
        import faiss

        self._chunks = chunks
        if not chunks:
            return
        texts = [c.text for c in chunks]
        embeddings = self._model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatL2(dim)
        self._index.add(embeddings.astype(np.float32))
        logger.info("rag_index_built", chunks=len(chunks), dim=dim)

    def retrieve_sync(self, query: str, top_k: int = 5) -> list[PolicyChunk]:
        """Synchronous retrieval — call this inside asyncio.to_thread()."""
        if self._index is None or not self._chunks:
            return []
        q_emb = self._model.encode([query], convert_to_numpy=True).astype(np.float32)
        _, indices = self._index.search(q_emb, min(top_k, len(self._chunks)))
        return [self._chunks[i] for i in indices[0] if i < len(self._chunks)]


@lru_cache(maxsize=1)
def get_rag_index() -> PolicyRAGIndex:
    """Singleton RAG index — built once, reused across requests."""
    index = PolicyRAGIndex()
    chunks = _load_policy_chunks("documentation/config.json")
    index.build(chunks)
    return index


# ---------------------------------------------------------------------------
# Async RAG compliance detection
# ---------------------------------------------------------------------------

_COMPLIANCE_SYSTEM = """You are a telecom compliance auditor.
Given a set of policy rules and a conversation transcript, detect any violations.
Return ONLY valid JSON — no markdown, no explanation."""

_COMPLIANCE_PROMPT = """Check this conversation for violations of the following policy rules.

POLICY RULES (retrieved for relevance):
{policy_context}

CONVERSATION TRANSCRIPT:
{transcript}

Return ONLY this JSON (no extra text):
{{
  "compliance_violations": [
    {{
      "violation_id": "CV-001",
      "type": "POLICY_RULE_ID",
      "severity": "high|medium|low",
      "confidence": 0.9,
      "description": "What the violation is",
      "evidence": "Exact quote from the conversation",
      "policy_reference": "Policy section and rule text",
      "recommended_action": "What should be done"
    }}
  ]
}}

If no violations are found, return: {{"compliance_violations": []}}
Only include violations with confidence >= 0.6."""


async def detect_compliance_violations(transcript: str) -> dict:
    """RAG-grounded async compliance detection.

    Steps:
      1. FAISS vector search for top-5 relevant policy chunks (in thread pool)
      2. Build prompt with retrieved policy text + transcript
      3. Call Groq 70B (small focused prompt — fast)

    Returns:
        dict with "compliance_violations" list and "processing_time_ms"
    """
    from app.core.config import settings
    from app.services.ai_engine import generate_analysis
    from app.services.groq_engine import generate_groq_analysis, is_groq_available

    start = time.perf_counter()

    # Step 1 — retrieve relevant policy chunks in thread pool
    rag_index = get_rag_index()
    retrieved: list[PolicyChunk] = await asyncio.to_thread(
        rag_index.retrieve_sync, transcript, 5
    )

    if not retrieved:
        logger.warning("rag_no_chunks_retrieved")
        return {"compliance_violations": [], "processing_time_ms": 0}

    # Build context block
    policy_context = "\n".join(
        f"[{c.rule_id}] {c.text}" for c in retrieved
    )

    prompt = _COMPLIANCE_PROMPT.format(
        policy_context=policy_context,
        transcript=transcript[:6000],
    )

    # Step 2 — ask LLM to check which rules are violated
    if is_groq_available():
        result = await generate_groq_analysis(
            system_instruction=_COMPLIANCE_SYSTEM,
            user_prompt=prompt,
            model=settings.GROQ_JUDGMENT_MODEL,
            temperature=0.1,
            task_label="rag_compliance",
        )
    else:
        result = await generate_analysis(
            system_instruction=_COMPLIANCE_SYSTEM,
            user_prompt=prompt,
        )

    elapsed_ms = int((time.perf_counter() - start) * 1000)
    result["processing_time_ms"] = elapsed_ms
    result["retrieved_policies"] = [c.rule_id for c in retrieved]

    violations = result.get("compliance_violations", [])
    logger.info(
        "rag_compliance_complete",
        violations_found=len(violations),
        policies_retrieved=len(retrieved),
        elapsed_ms=elapsed_ms,
    )
    return result
