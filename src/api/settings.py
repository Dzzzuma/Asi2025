"""Application settings loaded from environment variables / .env.

Sprint 4 requirement: configuration via .env / env vars, without committing secrets.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration.

    Values are read from environment variables and (optionally) a local `.env` file.
    The `.env` file must NOT be committed to the repository.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    MODEL_PATH: str | None = None
    WANDB_API_KEY: str | None = None
    DATABASE_URL: str = "sqlite:///local.db"


# Singleton settings instance
settings = Settings()
