"""Audio Transcription Service using Gemini File Upload API.

Wraps the audio_transmition/test_gemini.py logic as a proper async service
that integrates into the main FastAPI pipeline.

Pipeline Step 1: Audio → Speaker-diarized Transcript (User 1/User 2 format)
"""

import tempfile
import time
import os
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)

# Supported audio extensions that can be sent directly
DIRECT_UPLOAD_TYPES = {
    "audio/ogg", "audio/wav", "audio/mp3", "audio/mpeg",
    "audio/webm", "audio/x-m4a", "audio/mp4",
}


def _get_genai():
    """Lazy import of google.generativeai to avoid startup errors if not installed."""
    try:
        import google.generativeai as genai
        return genai
    except ImportError as exc:
        raise RuntimeError(
            "google-generativeai is not installed. Run: pip install google-generativeai"
        ) from exc


def transcribe_audio_bytes(
    audio_bytes: bytes,
    audio_mime_type: str,
    api_key: str,
    model_name: str = "models/gemini-2.5-flash",
) -> str:
    """Transcribe raw audio bytes to a speaker-diarized plain-text transcript.

    Args:
        audio_bytes: Raw bytes of the audio file.
        audio_mime_type: MIME type (e.g. 'audio/ogg').
        api_key: Google Generative AI API key.
        model_name: Gemini model to use for transcription.

    Returns:
        Speaker-diarized transcript string with 'User 1:', 'User 2:' labels.

    Raises:
        RuntimeError: On transcription failure.
    """
    genai = _get_genai()
    genai.configure(api_key=api_key)

    # Detect file extension from MIME type
    ext_map = {
        "audio/ogg": ".ogg",
        "audio/wav": ".wav",
        "audio/mp3": ".mp3",
        "audio/mpeg": ".mp3",
        "audio/webm": ".webm",
        "audio/x-m4a": ".m4a",
        "audio/mp4": ".mp4",
    }
    suffix = ext_map.get(audio_mime_type, ".wav")

    # Write bytes to a temp file for upload
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        logger.info("transcription_upload_start", path=tmp_path, mime=audio_mime_type)
        audio_file = genai.upload_file(path=tmp_path)

        # Poll until ACTIVE
        for _ in range(30):
            file_info = genai.get_file(audio_file.name)
            if file_info.state.name == "ACTIVE":
                break
            if file_info.state.name == "FAILED":
                raise RuntimeError(
                    f"Gemini file upload FAILED for mime={audio_mime_type}. "
                    "Try converting to WAV first."
                )
            time.sleep(2)
        else:
            raise RuntimeError("Gemini file did not become ACTIVE within 60s.")

        model = genai.GenerativeModel(model_name)
        prompt = (
            "Please provide a word-for-word transcript of this audio. "
            "There are multiple speakers in this audio. You must diarize the conversation "
            "by labeling each spoken segment with 'User 1:', 'User 2:', etc. "
            "Do not summarize, add any conversational filler, or describe the audio. "
            "Just write the speaker label followed by their spoken words."
        )

        resp = model.generate_content([audio_file, prompt])
        transcript = (resp.text or "").strip()

        # Cleanup uploaded file
        genai.delete_file(audio_file.name)

        logger.info("transcription_complete", chars=len(transcript))
        return transcript

    except Exception as exc:
        logger.error("transcription_failed", error=str(exc))
        raise RuntimeError(f"Audio transcription failed: {exc}") from exc
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


async def transcribe_audio_bytes_async(
    audio_bytes: bytes,
    audio_mime_type: str,
    api_key: str,
    model_name: str = "models/gemini-2.5-flash",
) -> str:
    """Async wrapper around transcribe_audio_bytes using run_in_executor."""
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        transcribe_audio_bytes,
        audio_bytes,
        audio_mime_type,
        api_key,
        model_name,
    )
