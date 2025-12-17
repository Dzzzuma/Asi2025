"""Application settings loaded from environment variables / .env."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    MODEL_PATH: str | None = None
    WANDB_API_KEY: str | None = None
    DATABASE_URL: str = "sqlite:///local.db"


@lru_cache
def get_settings() -> Settings:
    return Settings()
