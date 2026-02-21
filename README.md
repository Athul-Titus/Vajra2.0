# Vajra 2.0 — Multimodal Conversation Intelligence

Vajra 2.0 is an end-to-end pipeline that transforms raw audio or text into structured, evidence-based intelligence (Sentiment, Risk, Compliance, and Quality).

## 💻 API SPECIFICATION (Backend Service)

Vajra 2.0 is a proper REST API backend. You do not need the Python files once the server is running.

**Base URL**: `http://127.0.0.1:8000`

### 🎙️ Analyze Audio Endpoint
**`POST /api/v1/conversations/analyze/audio`**

| Requirement | Value |
| :--- | :--- |
| **Method** | `POST` |
| **Auth Header** | `X-API-KEY: ci-dev-key-2026` |
| **Content-Type** | `multipart/form-data` |
| **Body (file)** | Binary audio file (`.ogg`, `.wav`, `.mp3`) |
| **Body (config_id)** | `telecom_default` |

### 📄 Analyze Text Endpoint
**`POST /api/v1/conversations/analyze/text`**
- **Body**: `{ "transcript": "User 1: ... User 2: ...", "config_id": "telecom_default" }`

### 💻 Example Test (CURL)
Run this from the project root to test the audio pipeline:
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/conversations/analyze/audio" \
     -H "X-API-KEY: ci-dev-key-2026" \
     -F "file=@audio_transmition/athul1.ogg" \
     -F "config_id=telecom_default"
```

---

## 📂 Where to put your Audio Files?

If you are using the **Web Portal** or the **API**:
1. Open `http://127.0.0.1:8000` in your browser.
2. Use the **Upload Area** to submit files.

## 🚀 Quick Start (Connected Pipeline)

1. **Setup**: Create a `.env` file in the root with your `GEMINI_API_KEY`.
2. **Run Test**: `python test_full_pipeline.py`
3. **View Results**: Check `full_pipeline_test_result.json` for the deep insights.

---
*Unified Pipeline: Transcription → Sentiment → Risk Scoring → Compliance → Agent Quality*
