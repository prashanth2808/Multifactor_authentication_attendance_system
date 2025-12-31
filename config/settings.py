# config/settings.py
"""
Central configuration loader using pydantic-settings
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from pathlib import Path
import os

# Load .env from project root
ENV_PATH = Path(__file__).parent.parent / ".env"

class Settings(BaseSettings):
    """Application settings with validation"""
    
    # MongoDB
    mongodb_uri: str = Field(..., env="MONGODB_URI")
    db_name: str = Field("face_attendance", env="DB_NAME")
    
    # Recognition
    similarity_threshold: float = Field(0.62, env="SIMILARITY_THRESHOLD")
    min_photos: int = Field(3, env="MIN_PHOTOS")
    
    # Liveness
    liveness_required: bool = Field(False, env="LIVENESS_REQUIRED")
    
    # Logging
    log_level: str = Field("INFO", env="LOG_LEVEL")
    
    # Model paths (PURE models, NO BUFFALO)
    retinaface_model: str = "models/retinaface.onnx"
    arcface_model: str = "models/arcface_r100_glint360k.onnx"

    model_config = SettingsConfigDict(
        env_file=ENV_PATH,
        env_file_encoding="utf-8",
        case_sensitive=False
    )

# Global settings instance
settings = Settings()

# Validate on import
try:
    from rich.console import Console
    console = Console()
    console.print("[green]Settings loaded successfully[/green]")
except Exception as e:
    from rich.console import Console
    console = Console()
    console.print(f"[bold red]Settings validation failed: {e}[/bold red]")
    raise