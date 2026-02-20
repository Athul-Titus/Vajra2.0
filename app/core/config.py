"""Application settings loaded from environment variables."""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """Central configuration for the application.

    All values are loaded from environment variables or .env file.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- App ---
    APP_NAME: str = "Conversation Intelligence API"
    APP_VERSION: str = "0.1.0"
    APP_ENV: str = "development"
    APP_DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # --- Gemini AI (fallback) ---
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.0-flash"

    # --- Groq AI (primary — parallel dual-model) ---
    GROQ_API_KEY: str = ""
    GROQ_EXTRACTION_MODEL: str = "llama-3.1-8b-instant"
    GROQ_JUDGMENT_MODEL: str = "llama-3.3-70b-versatile"

    # --- API Security ---
    API_KEY: str = "ci-dev-key-2026"

    # --- Database ---
    DATABASE_URL: str = "sqlite+aiosqlite:///./conversation_intelligence.db"

    # --- File Upload ---
    MAX_AUDIO_SIZE_MB: int = 25
    ALLOWED_AUDIO_TYPES: list[str] = [
        "audio/wav",
        "audio/mpeg",
        "audio/mp3",
        "audio/ogg",
        "audio/webm",
        "audio/x-m4a",
        "audio/mp4",
    ]
    MAX_TRANSCRIPT_LENGTH: int = 50_000

    @property
    def max_audio_size_bytes(self) -> int:
        return self.MAX_AUDIO_SIZE_MB * 1024 * 1024

    @property
    def is_development(self) -> bool:
        return self.APP_ENV == "development"


# Singleton instance
settings = AppSettings()
