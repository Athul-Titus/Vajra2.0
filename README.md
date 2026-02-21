# 🎙️ Multimodal Telecom Conversation Intelligence

> **Transight Hackathon** — Production-grade FastAPI backend that analyses telecom customer calls (voice + text) and returns structured, evidence-based intelligence in a single API call.

---

## ✨ What It Does

Upload a raw call recording → get back everything in **~12 seconds**:

| Output | Detail |
|---|---|
| 🎙️ **Diarized Transcript** | Per-speaker segments (`[User 1]` / `[User 2]`) — supports Malayalam, Hindi, English |
| ⚖️ **Compliance Violations** | RAG-grounded — violations cite exact policy text from `config.json` |
| 🧑‍💼 **Agent Quality Score** | 0–10 across greeting, empathy, professionalism, resolution, closing |
| 💬 **Sentiment Analysis** | Customer + agent sentiment with confidence scores |
| ⚠️ **Risk Assessment** | Escalation level (none/low/medium/high/critical) + churn trigger detection |
| 📋 **Call Outcome** | Resolved / escalated / follow-up required / transferred / dropped |
| 🗂️ **Speaker Timeline** | Turn-by-turn key points and sentiment |

---

## 🏗️ Architecture

```
Audio File (OGG / MP4 / WAV / MP3)
        │
        ▼
[1] Gemini Diarized Transcription
        │  "User 1: ... / User 2: ..."
        │
        ▼
[2] Three parallel async tasks:
        ├── Groq llama-3.1-8b-instant   → Extraction (sentiment, entities, intents, topics, timeline)
        ├── Groq llama-3.3-70b-versatile → Judgment (quality, risk, outcome)
        └── RAG Compliance               → FAISS vector search → Groq checks policy violations
                │                            (grounded in documentation/config.json)
        ▼
[3] Unified JSON Response
```

**No extra latency from RAG** — all three tasks run concurrently via `asyncio.gather`.

---

## 🧠 AI Usage Approach

Our approach leverages specific models for the tasks they perform best, optimizing for both speed and reasoning quality:

- **Google Gemini**: Handles diarized audio transcription, providing highly accurate speaker separation (e.g., distinguishing between Customer and Agent).
- **Groq (Llama-3.1-8b-instant)**: Utilized for rapid extraction tasks such as generating summaries, analyzing sentiment, identifying entities, and extracting topics.
- **Groq (Llama-3.3-70b-versatile)**: Responsible for complex reasoning, including agent quality scoring, nuanced risk assessment, and determining the call outcome.
- **RAG (Retrieval-Augmented Generation)**: Uses FAISS and sentence transformers to embed client policies, grounding compliance violation checks in actual documentation rather than relying on LLM memory.

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- `GEMINI_API_KEY` — for diarized transcription
- `GROQ_API_KEY` — for parallel LLM analysis

### Setup

```bash
# Clone
git clone https://github.com/Athul-Titus/Vajra2.0.git
cd Vajra2.0

# Install dependencies (using uv)
pip install uv
uv sync

# Or using pip
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your API keys
```

### Environment Variables

```env
GEMINI_API_KEY=your_gemini_api_key
GROQ_API_KEY=your_groq_api_key
API_KEY=ci-dev-key-2026
DATABASE_URL=sqlite+aiosqlite:///./conversation_intelligence.db
MAX_AUDIO_SIZE_MB=25
```

### Run

```bash
uvicorn app.main:app --reload
```

API docs: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 📡 API Endpoints

All endpoints require the header: `X-API-Key: ci-dev-key-2026`

### ⭐ Main Pipeline — Audio → Full Analysis

```
POST /api/v1/conversations/analyze/audio/full
```

**Form fields:**

| Field | Type | Default | Description |
|---|---|---|---|
| `file` | File | required | Audio file (ogg/mp3/wav/m4a/webm/mpeg) |
| `task` | string | `transcribe` | `transcribe` or `translate` (→ English) |
| `language` | string | auto | Language hint e.g. `Malayalam` |
| `config_id` | string | `telecom_default` | Client config profile |

**Example:**

```bash
curl -X POST "http://localhost:8000/api/v1/conversations/analyze/audio/full" \
  -H "X-API-Key: ci-dev-key-2026" \
  -F "file=@call_recording.ogg;type=audio/ogg" \
  -F "task=translate" \
  -F "config_id=telecom_default"
```

**Windows CMD:**

```cmd
curl -X POST "http://localhost:8000/api/v1/conversations/analyze/audio/full" ^
  -H "X-API-Key: ci-dev-key-2026" ^
  -F "file=@call_recording.ogg;type=audio/ogg" ^
  -F "task=translate" ^
  -F "config_id=telecom_default"
```

---

### 📝 Text Transcript Analysis

```
POST /api/v1/conversations/analyze/text
```

```bash
curl -X POST "http://localhost:8000/api/v1/conversations/analyze/text" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: ci-dev-key-2026" \
  -d '{"transcript": "Agent: How can I help you?\nCustomer: My internet is down."}'
```

### 🎙️ Transcription Only

```
POST /api/v1/transcribe/audio/diarize    # With speaker separation
POST /api/v1/transcribe/audio            # Simple transcription
```

### 📊 Standalone Analytics

```
POST /api/v1/analytics/sentiment    # Sentiment only (fast, 8B model)
POST /api/v1/analytics/risk         # Risk score only (weighted matrix)
```

---

## 📦 Sample Response

```json
{
  "success": true,
  "metadata": {
    "processing_time_ms": 11186,
    "ai_model": "llama-3.1-8b-instant + llama-3.3-70b-versatile",
    "ai_provider": "groq",
    "input_type": "audio_full"
  },
  "transcription": {
    "transcript": "[User 1]: Hello, my internet has been slow...\n[User 2]: I'm sorry to hear that...",
    "segments": [
      { "speaker": "SPEAKER_01", "speaker_label": "User 1", "text": "Hello, my internet has been slow..." },
      { "speaker": "SPEAKER_02", "speaker_label": "User 2", "text": "I'm sorry to hear that..." }
    ],
    "speakers": ["SPEAKER_01", "SPEAKER_02"],
    "transcription_time_ms": 9814
  },
  "analysis": {
    "summary": "Customer frustrated with slow internet, threatens to switch providers.",
    "sentiment": {
      "overall": "negative",
      "customer": { "label": "frustrated", "confidence": 0.95 },
      "agent": { "label": "professional", "confidence": 0.90 }
    },
    "compliance_violations": [
      {
        "violation_id": "CV-001",
        "type": "authentication_1",
        "severity": "high",
        "confidence": 0.9,
        "evidence": "Could you please provide your registered mobile number?",
        "policy_reference": "[authentication_1] Verify caller identity using two of: Account PIN, Last 4 of SSN, DOB, or Billing Address before discussing account details.",
        "recommended_action": "Request two verification factors before proceeding."
      }
    ],
    "agent_quality": {
      "overall_score": 8.0,
      "greeting": { "score": 7.0, "present": true },
      "empathy": { "score": 8.0, "present": true },
      "professionalism": { "score": 9.0, "present": true },
      "resolution_skills": { "score": 8.0, "present": true },
      "closing": { "score": 6.0, "present": false }
    },
    "risk_assessment": {
      "score": 6.0,
      "level": "medium",
      "triggers_detected": ["switch to another service provider"],
      "recommended_actions": ["Follow up within 24 hours", "Offer retention package"]
    },
    "call_outcome": { "classification": "follow_up_required", "confidence": 0.9 }
  }
}
```

---

## ⚙️ Configuration (Multi-Tenant)

Compliance rules, risk triggers, and quality criteria are fully config-driven. Edit `documentation/config.json` to customise for any telecom operator:

```json
{
  "policies_or_rules": {
    "authentication": [
      "Verify caller identity using two of: Account PIN, Last 4 of SSN, DOB, or Billing Address."
    ],
    "data_privacy": [
      "Never ask for full credit card numbers, CVV, or PINs."
    ]
  },
  "risk_or_compliance_triggers": [
    "Customer explicitly threatens to switch to a competitor.",
    "Customer threatens legal action."
  ]
}
```

No code changes needed — just update the JSON and restart.

---

## 🗂️ Project Structure

```
app/
├── api/v1/endpoints/
│   ├── conversations.py   # Main pipeline + text analysis routes
│   ├── transcription.py   # Standalone transcription routes
│   ├── analytics.py       # Sentiment + risk standalone routes
│   ├── configurations.py  # Client config CRUD
│   └── health.py          # Health check
├── services/
│   ├── conversation_analyzer.py  # Orchestrates the full pipeline
│   ├── transcription_service.py  # Gemini diarized transcription
│   ├── rag_policy.py             # FAISS RAG compliance detection
│   ├── analytics_service.py      # Standalone sentiment + risk
│   ├── ai_engine.py              # Gemini provider
│   ├── groq_engine.py            # Groq provider
│   └── prompt_builder.py         # Config-driven prompt assembly
├── prompts/
│   ├── extraction_prompt.py      # 8B: summary, sentiment, entities
│   └── judgment_prompt.py        # 70B: quality, risk, outcome
├── models/
│   ├── schemas/responses.py      # Pydantic response schemas
│   └── database.py               # SQLAlchemy async DB
└── core/
    ├── config.py                  # Settings (env vars)
    ├── security.py                # API key auth
    └── exceptions.py             # Custom exceptions
documentation/
└── config.json                   # Policy rules + risk triggers
```

---

## 🔑 Key Design Decisions

| Decision | Rationale |
|---|---|
| **Single unified endpoint** | One upload → everything. Best demo UX. |
| **Groq parallel 8B + 70B** | Small model for fast extraction, large for reasoning. ~2-3s total. |
| **RAG compliance** | Violations grounded in policy text — not LLM hallucination. |
| **Config-driven** | Change compliance rules without touching code. Multi-tenant ready. |
| **Async throughout** | FAISS runs in `asyncio.to_thread()`. All three AI tasks run concurrently. |
| **Evidence on every claim** | Every violation, score, and risk trigger includes an exact quote. |

---

## 🛠️ Tech Stack

- **Framework:** FastAPI + Uvicorn
- **AI Providers:** Google Gemini (transcription) · Groq (analysis)
- **LLM Models:** `llama-3.1-8b-instant` · `llama-3.3-70b-versatile`
- **RAG:** `sentence-transformers/all-MiniLM-L6-v2` + FAISS
- **DB:** SQLite (async via SQLAlchemy + aiosqlite)
- **Validation:** Pydantic v2
- **Logging:** structlog (structured JSON logs)

---

## 📌 Assumptions

| Assumption | Detail |
|---|---|
| **Two speakers per call** | System assumes a two-party call (customer + agent). Multi-party calls will still transcribe but speaker labels may not map cleanly. |
| **GROQ_API_KEY is required** | Groq drives all analysis. Gemini is the fallback for analysis only if Groq is unavailable. Without both keys, the server will start but analysis endpoints will fail. |
| **GEMINI_API_KEY is required** | Gemini exclusively handles diarized transcription. There is no fallback for the transcription step. |
| **Audio is a telecom call** | Not designed for music, noisy environments, or multi-person panel discussions. Best results with a 1:1 customer support call. |
| **Language auto-detected** | Gemini detects the language automatically. `language` hint is optional and improves accuracy for regional languages (e.g. `Malayalam`). |
| **`config.json` is source of truth** | Compliance policies and risk triggers are read from `documentation/config.json` at startup. A server restart is needed after editing it. |
| **First-request RAG warmup** | The FAISS index and sentence-transformer model load on the first request (~30s). All subsequent requests use the cached index and run in ~12s total. |
| **SQLite for demo only** | The database is SQLite for simplicity. For production, replace `DATABASE_URL` with a PostgreSQL connection string. |

---

## ⚠️ Limitations

- **Real-time Processing**: The current system relies on full file uploads and does not support real-time audio stream processing.
- **Audio Quality Dependency**: Extremely noisy environments or complex multi-speaker (3+) panels may reduce the accuracy of the model's diarization.
- **External Dependencies**: The system speed and availability are directly reliant on the rate limits and uptimes of the external APIs.

---

## 🔮 Future Improvements

- **Real-Time API Integration**: Transition to processing live call audio streams via real-time APIs, providing immediate feedback during a call.
- **Data Analytics and Charting**: Implement data sorting, aggregations, and visual charts to track agent performance, risk trends, and customer sentiment over time.
- **AI Agent Training Loop**: Utilize the structured call outcomes, charts, and scored data as a robust dataset to securely train AI agents and fine-tune models.

---

## 📝 License

Built for the **Transight Hackathon** · February 2026
