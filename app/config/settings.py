"""Application configuration settings using Pydantic Settings."""

from pathlib import Path
from typing import Any, Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Global configuration settings for the Autonomous AI Task Worker."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Core Application
    APP_NAME: str = "Autonomous AI Task Worker"
    APP_VERSION: str = "0.1.0"
    APP_ENV: str = "development"
    DEBUG: bool = False

    @field_validator("DEBUG", mode="before")
    @classmethod
    def parse_debug(cls, v: Any) -> bool:
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.strip().lower() in ("true", "1", "yes", "debug", "dev")
        return False
    PORT: int = 8000
    HOST: str = "0.0.0.0"
    LOG_LEVEL: str = "INFO"

    # Database
    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/task_worker",
        description="Async database connection string"
    )
    FALLBACK_SQLITE_URL: str = "sqlite+aiosqlite:///./data/task_worker.db"
    USE_SQLITE_FALLBACK: bool = True

    # OpenAI-Compatible LLM Settings
    OPENAI_API_KEY: Optional[str] = "mock-key-for-development"
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"
    OPENAI_MODEL: str = "gpt-4o-mini"
    LLM_TEMPERATURE: float = 0.0

    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    INBOX_DIR: Path = Field(default_factory=lambda: Path("./data/invoices"))
    ARTIFACTS_DIR: Path = Field(default_factory=lambda: Path("./data/artifacts"))

    # Browser automation
    BROWSER_HEADLESS: bool = True
    PLAYWRIGHT_BROWSERS_PATH: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent.parent / ".playwright-browsers"
    )

    # Finance Application Settings
    FINANCE_APP_USERNAME: str = "finance_admin"
    FINANCE_APP_PASSWORD: str = "password123"
    FINANCE_APP_SECRET_KEY: str = "finance-secret-key-autonomous-worker-2024"
    FINANCE_APP_COOKIE_NAME: str = "finance_session"

    # Agent Limits
    MAX_STEPS: int = 20
    DEFAULT_MAX_STEPS: int = 20
    MAX_CONSECUTIVE_FAILURES: int = 3


settings = Settings()

# Ensure Playwright installs and finds browsers on D: drive to prevent C: ENOSPC
import os
if "PLAYWRIGHT_BROWSERS_PATH" not in os.environ:
    os.environ["PLAYWRIGHT_BROWSERS_PATH"] = str(settings.PLAYWRIGHT_BROWSERS_PATH)
