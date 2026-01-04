# config/settings.py
"""
Central configuration loader using pydantic-settings
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

# Load .env from project root (local dev only)
ENV_PATH = Path(__file__).parent.parent / ".env"


class Settings(BaseSettings):
    """Application settings with validation"""

    # ===============================
    # Supabase (OPTIONAL – backend uses Postgres directly)
    # ===============================
    supabase_url: Optional[str] = Field(default=None, env="SUPABASE_URL")
    supabase_anon_key: Optional[str] = Field(default=None, env="SUPABASE_ANON_KEY")

    # ===============================
    # Recognition
    # ===============================
    similarity_threshold: float = Field(0.62, env="SIMILARITY_THRESHOLD")
    min_photos: int = Field(3, env="MIN_PHOTOS")

    # ===============================
    # Voice verification
    # ===============================
    voice_threshold: float = Field(0.68, env="VOICE_THRESHOLD")

    # ===============================
    # Liveness
    # ===============================
    liveness_required: bool = Field(False, env="LIVENESS_REQUIRED")

    # ===============================
    # Logging
    # ===============================
    log_level: str = Field("INFO", env="LOG_LEVEL")

    # ===============================
    # Model paths
    # ===============================
    retinaface_model: str = "models/retinaface.onnx"
    arcface_model: str = "models/arcface_r100_glint360k.onnx"

    model_config = SettingsConfigDict(
        env_file=ENV_PATH,
        env_file_encoding="utf-8",
        case_sensitive=False
        extra="ignore"
    )


# Global settings instance
settings = Settings()
