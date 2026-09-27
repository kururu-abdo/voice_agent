import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    livekit_url: str = ""
    livekit_api_key: str = ""
    livekit_api_secret: str = ""
    google_api_key: str = ""

    chat_model: str = "gemini-3.8-flash"
    embedding_model: str = "models/gemini-embedding-001"
    stt_model: str = "gemini-3.5-transcribe-live"
    tts_model: str = "gemini-3.8-flash-tts"
    tts_voice: str = "Zephyr"
    company_name: str = "Your Company"
    rag_top_k: int = 5
    rag_min_relevance: float = 0.25
    chroma_path: Path = ROOT / "data" / "chroma"
    upload_path: Path = ROOT / "uploads"
    root_documents_path: Path = ROOT / "documents"


@lru_cache
def get_settings() -> Settings:
    legacy_values: dict[str, str] = {}
    legacy_file = ROOT / "api.txt"
    if not (ROOT / ".env").exists() and legacy_file.exists():
        valid_keys = {
            "LIVEKIT_URL",
            "LIVEKIT_API_KEY",
            "LIVEKIT_API_SECRET",
            "GOOGLE_API_KEY",
        }
        for line in legacy_file.read_text(encoding="utf-8").splitlines():
            key, separator, value = line.strip().partition("=")
            if separator and key in valid_keys:
                legacy_values[key.lower()] = value.strip()
                os.environ.setdefault(key, value.strip())
    return Settings(**legacy_values)
