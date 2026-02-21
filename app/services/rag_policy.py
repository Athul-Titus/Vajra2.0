"""
RAG pipeline for compliance/policy violation detection (production-ready).
- Loads policy JSON (documentation/config.json)
- Embeds and indexes policy rules for retrieval
- At runtime: retrieves relevant policies for a transcript, passes both to LLM
- Minimizes hallucination, maximizes traceability
"""
import json
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# --- 1. Load and chunk policy document ---

class PolicyChunk:
    def __init__(self, section: str, rule: str, text: str):
        self.section = section
        self.rule = rule
        self.text = text

    def __repr__(self):
        return f"[{self.section}] {self.rule}: {self.text}"


def load_policy_chunks(json_path: str) -> List[PolicyChunk]:
    with open(json_path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    chunks = []
    for section, rules in doc.get("policies_or_rules", {}).items():
        for idx, rule in enumerate(rules, 1):
            chunks.append(PolicyChunk(section, f"{section}_{idx}", rule))
    return chunks

# --- 2. Embed and index chunks (FAISS) ---

class PolicyRAGIndex:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.chunks: List[PolicyChunk] = []
        self.index = None
        self.embeddings = None

    def build(self, chunks: List[PolicyChunk]):
        self.chunks = chunks
        texts = [c.text for c in chunks]
        self.embeddings = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(self.embeddings)

    def retrieve(self, query: str, top_k=5) -> List[PolicyChunk]:
        q_emb = self.model.encode([query], convert_to_numpy=True)
        D, I = self.index.search(q_emb, top_k)
        return [self.chunks[i] for i in I[0]]

# --- 3. RAG pipeline: retrieve + LLM prompt ---

def build_policy_context(chunks: List[PolicyChunk]) -> str:
    return "\n".join(f"[{c.section}] {c.text}" for c in chunks)


def rag_policy_violation_detection(transcript: str, rag_index: PolicyRAGIndex, llm_call) -> Any:
    # 1. Retrieve relevant policy chunks
    retrieved = rag_index.retrieve(transcript, top_k=5)
    policy_context = build_policy_context(retrieved)
    # 2. Build prompt for LLM
    prompt = (
        "You are a compliance analyst. Given the following company policy rules:\n"
        f"{policy_context}\n\n"
        "Analyze the following conversation transcript for any policy or compliance violations. "
        "For each violation, cite the policy section and explain the reasoning.\n\n"
        f"Transcript:\n{transcript}\n"
        "Return a JSON list of violations, each with: section, rule, description."
    )
    # 3. Call LLM (Groq, Gemini, etc.)
    return llm_call(prompt)

# --- Example usage (for integration) ---
if __name__ == "__main__":
    # 1. Build RAG index
    chunks = load_policy_chunks("documentation/config.json")
    rag = PolicyRAGIndex()
    rag.build(chunks)
    # 2. Example transcript
    transcript = "Customer: I want to know why my bill is so high.\nAgent: I can check that for you. May I have your account PIN and billing address?\n..."
    # 3. Dummy LLM call (replace with real call)
    def dummy_llm(prompt):
        print("PROMPT TO LLM:\n", prompt)
        return []
    rag_policy_violation_detection(transcript, rag, dummy_llm)
