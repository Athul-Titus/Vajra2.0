"""Transcription service using Gemini's multimodal audio capabilities.

Wraps the audio upload → transcription pipeline originally prototyped in
audio_transmition/whisper.py, adapted for async FastAPI use.

Two public async functions:
  transcribe_audio          → single response (Gemini handles diarization via prompt)
  transcribe_with_diarization → same but with an explicit diarization prompt

Both use the Gemini File Upload API (google-genai SDK) — no extra dependencies.
"""

import asyncio
import tempfile
import time
from pathlib import Path

import structlog
from google import genai
from google.genai import types

from app.core.config import settings
from app.core.exceptions import AIEngineError, AudioProcessingError

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Lazy Gemini client (reuse project singleton pattern)
# ---------------------------------------------------------------------------

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        if not settings.GEMINI_API_KEY:
            raise AIEngineError("GEMINI_API_KEY is not configured")
        _client = genai.Client(api_key=settings.GEMINI_API_KEY)
        logger.info("transcription_gemini_client_initialised")
    return _client


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MIME_TO_EXT = {
    "audio/wav": ".wav",
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/ogg": ".ogg",
    "audio/webm": ".webm",
    "audio/x-m4a": ".m4a",
    "audio/mp4": ".mp4",
}


def _build_prompt(task: str, language: str | None) -> str:
    """Build the transcription prompt based on task and language."""
    if task == "translate":
        lang_instruction = "translated into English"
    elif language:
        lang_instruction = f"in {language}"
    else:
        lang_instruction = "in the language spoken"

    return (
        f"Please provide a word-for-word transcript of this audio {lang_instruction}. "
        "Do not summarize, add conversational filler, or describe the audio. "
        "Just write the exact spoken words."
    )


def _build_diarization_prompt(task: str, language: str | None) -> str:
    """Build the diarized transcription prompt."""
    if task == "translate":
        lang_instruction = "translated into English"
    elif language:
        lang_instruction = f"in {language}"
    else:
        lang_instruction = "in the language spoken"

    return (
        f"Please provide a word-for-word transcript of this audio {lang_instruction}. "
        "There are multiple speakers. Diarize the conversation by labeling each spoken "
        "segment with 'User 1:', 'User 2:', etc. "
        "Format each line as: 'User N: <spoken words>'. "
        "Do not summarize, add conversational filler, or describe the audio. "
        "Just write the speaker label followed by their exact spoken words."
    )


def _parse_diarized_transcript(raw: str) -> tuple[str, list[dict]]:
    """Parse 'User N: text' lines into structured segments."""
    segments = []
    speakers_seen: dict[str, str] = {}
    speaker_counter = 0

    for line in raw.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        if ":" in line:
            label, _, text = line.partition(":")
            label = label.strip()
            text = text.strip()
            if label not in speakers_seen:
                speaker_counter += 1
                speakers_seen[label] = f"SPEAKER_{speaker_counter:02d}"
            segments.append({
                "speaker": speakers_seen[label],
                "speaker_label": label,
                "text": text,
            })
        else:
            # line with no speaker prefix — attach to last speaker or SPEAKER_00
            last_speaker = segments[-1]["speaker"] if segments else "SPEAKER_00"
            last_label = segments[-1]["speaker_label"] if segments else "User 1"
            segments.append({
                "speaker": last_speaker,
                "speaker_label": last_label,
                "text": line,
            })

    unique_speakers = sorted({s["speaker"] for s in segments})
    full_transcript = "\n".join(
        f"[{s['speaker_label']}]: {s['text']}" for s in segments
    )
    return full_transcript, segments, unique_speakers


async def _upload_and_transcribe(
    audio_bytes: bytes,
    mime_type: str,
    prompt: str,
) -> str:
    """Write bytes to temp file, upload to Gemini, transcribe, delete. Returns raw text."""
    client = _get_client()
    suffix = _MIME_TO_EXT.get(mime_type, ".wav")
    tmp_path = None

    try:
        # Write to temp file (Gemini SDK needs a file path)
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(audio_bytes)
            tmp_path = Path(tmp.name)

        logger.info("transcription_upload_start", size_bytes=len(audio_bytes))

        # Run blocking upload + generate in thread pool
        def _blocking_call():
            # Upload the audio file
            uploaded = client.files.upload(file=tmp_path, config={"mime_type": mime_type})

            try:
                # Generate transcript
                response = client.models.generate_content(
                    model=settings.GEMINI_MODEL,
                    contents=[
                        types.Part.from_uri(
                            file_uri=uploaded.uri,
                            mime_type=mime_type,
                        ),
                        prompt,
                    ],
                )
                return (response.text or "").strip()
            finally:
                # Always clean up the uploaded file
                try:
                    client.files.delete(name=uploaded.name)
                except Exception:
                    pass

        return await asyncio.get_event_loop().run_in_executor(None, _blocking_call)

    except AIEngineError:
        raise
    except Exception as exc:
        raise AudioProcessingError(f"Gemini transcription failed: {exc}") from exc
    finally:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Public async functions
# ---------------------------------------------------------------------------


async def transcribe_audio(
    audio_bytes: bytes,
    mime_type: str = "audio/wav",
    language: str | None = None,
    task: str = "transcribe",
) -> dict:
    """Transcribe audio using Gemini (single-speaker / no explicit diarization).

    Args:
        audio_bytes: Raw audio file bytes.
        mime_type:   MIME type of the audio.
        language:    Language hint (e.g. 'Malayalam', 'Hindi'). None = auto-detect.
        task:        'transcribe' or 'translate' (→ English).

    Returns:
        dict with transcript, language_hint, task, processing_time_ms
    """
    if task not in ("transcribe", "translate"):
        raise AudioProcessingError(
            f"Invalid task '{task}'. Must be 'transcribe' or 'translate'."
        )

    start = time.perf_counter()
    prompt = _build_prompt(task, language)
    transcript = await _upload_and_transcribe(audio_bytes, mime_type, prompt)
    elapsed_ms = int((time.perf_counter() - start) * 1000)

    logger.info(
        "transcription_complete",
        task=task,
        language=language,
        chars=len(transcript),
        elapsed_ms=elapsed_ms,
    )

    return {
        "transcript": transcript,
        "language_hint": language,
        "task": task,
        "processing_time_ms": elapsed_ms,
    }


async def transcribe_with_diarization(
    audio_bytes: bytes,
    mime_type: str = "audio/wav",
    language: str | None = None,
    task: str = "transcribe",
) -> dict:
    """Transcribe audio with speaker diarization using Gemini.

    Gemini labels each segment as 'User 1:', 'User 2:', etc.
    These are parsed into structured segments with speaker IDs.

    Args:
        audio_bytes: Raw audio file bytes.
        mime_type:   MIME type of the audio.
        language:    Language hint. None = auto-detect.
        task:        'transcribe' or 'translate' (→ English).

    Returns:
        dict with transcript, segments (list with speaker), speakers, processing_time_ms
    """
    if task not in ("transcribe", "translate"):
        raise AudioProcessingError(
            f"Invalid task '{task}'. Must be 'transcribe' or 'translate'."
        )

    start = time.perf_counter()
    prompt = _build_diarization_prompt(task, language)
    raw = await _upload_and_transcribe(audio_bytes, mime_type, prompt)
    elapsed_ms = int((time.perf_counter() - start) * 1000)

    full_transcript, segments, unique_speakers = _parse_diarized_transcript(raw)

    logger.info(
        "diarized_transcription_complete",
        task=task,
        language=language,
        speakers=unique_speakers,
        segments=len(segments),
        elapsed_ms=elapsed_ms,
    )

    return {
        "transcript": full_transcript,
        "language_hint": language,
        "task": task,
        "segments": segments,
        "speakers": unique_speakers,
        "processing_time_ms": elapsed_ms,
    }
