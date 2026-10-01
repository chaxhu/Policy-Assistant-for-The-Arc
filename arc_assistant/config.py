"""Load and validate settings from the environment and the .env file."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import dotenv_values
from pydantic import BaseModel, Field, SecretStr, ValidationError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"
INDEX_DIR = PROJECT_ROOT / "index"
EVAL_DIR = PROJECT_ROOT / "eval"
TEST_SET_PATH = EVAL_DIR / "test_set.yaml"
RUNS_DIR = EVAL_DIR / "runs"
ENV_FILE = PROJECT_ROOT / ".env"

MISSING_KEY_MESSAGE = (
    "No OpenAI API key found. Copy .env.example to .env, paste your key after "
    "OPENAI_API_KEY=, save the file and restart the app."
)


class ConfigError(RuntimeError):
    """Raised when settings are missing or invalid. The message is safe to show to users."""


class Settings(BaseModel):
    """Validated application settings. The API key is a SecretStr so it never prints."""

    openai_api_key: SecretStr = SecretStr("")
    chat_model: str = "gpt-4o-mini"
    embed_model: str = "text-embedding-3-small"
    top_k: int = Field(default=5, ge=1, le=20)
    handoff_threshold: float = Field(default=0.35, ge=0.0, le=1.0)

    @property
    def has_api_key(self) -> bool:
        """True if an API key has been provided."""
        return bool(self.openai_api_key.get_secret_value().strip())


_ENV_NAMES = {
    "openai_api_key": "OPENAI_API_KEY",
    "chat_model": "OPENAI_CHAT_MODEL",
    "embed_model": "OPENAI_EMBED_MODEL",
    "top_k": "RETRIEVAL_TOP_K",
    "handoff_threshold": "HANDOFF_SCORE_THRESHOLD",
}


def load_settings(env_file: Path | None = ENV_FILE, require_key: bool = True) -> Settings:
    """Read settings from the .env file, with real environment variables taking priority.

    Raises ConfigError with a friendly message if the key is missing (when required)
    or a value cannot be parsed.
    """
    file_values = dotenv_values(env_file) if env_file and Path(env_file).exists() else {}
    raw: dict[str, str] = {}
    for field, env_name in _ENV_NAMES.items():
        value = os.environ.get(env_name) or file_values.get(env_name)
        if value not in (None, ""):
            raw[field] = value.strip()

    try:
        settings = Settings(**raw)
    except ValidationError as exc:
        bad = ", ".join(_ENV_NAMES[str(err["loc"][0])] for err in exc.errors())
        raise ConfigError(f"Invalid value in .env for: {bad}. Check .env.example.") from None

    if require_key and not settings.has_api_key:
        raise ConfigError(MISSING_KEY_MESSAGE)
    return settings
