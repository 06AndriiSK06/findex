from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    index_path: Path = Field(
        default=Path("data/index.bin"),
        description="Path to the serialized findex binary or pickle index.",
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    host: str = Field(default="0.0.0.0", description="Bind address for uvicorn.")
    port: int = Field(default=8000, ge=1, le=65535, description="Bind port for uvicorn.")
    cors_origins: list[str] = Field(
        default_factory=lambda: ["*"],
        description="Allowed CORS origins.",
    )
