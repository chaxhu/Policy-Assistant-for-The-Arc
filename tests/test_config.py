from pathlib import Path

import pytest

from arc_assistant.config import ConfigError, load_settings

ENV_NAMES = [
    "OPENAI_API_KEY",
    "OPENAI_CHAT_MODEL",
    "OPENAI_EMBED_MODEL",
    "RETRIEVAL_TOP_K",
    "HANDOFF_SCORE_THRESHOLD",
]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


def write_env(tmp_path: Path, text: str) -> Path:
    path = tmp_path / ".env"
    path.write_text(text, encoding="utf-8")
    return path


def test_missing_key_gives_clear_message(tmp_path: Path) -> None:
    env = write_env(tmp_path, "OPENAI_API_KEY=\n")
    with pytest.raises(ConfigError) as excinfo:
        load_settings(env)
    message = str(excinfo.value)
    assert "OPENAI_API_KEY" in message
    assert ".env" in message


def test_missing_env_file_is_treated_as_missing_key(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="No OpenAI API key"):
        load_settings(tmp_path / "does-not-exist.env")


def test_key_not_required_for_offline_pages(tmp_path: Path) -> None:
    settings = load_settings(write_env(tmp_path, ""), require_key=False)
    assert not settings.has_api_key
    assert settings.top_k == 5
    assert settings.handoff_threshold == 0.35


def test_values_are_read_and_key_is_never_printed(tmp_path: Path) -> None:
    env = write_env(
        tmp_path,
        "OPENAI_API_KEY=sk-test-secret\nRETRIEVAL_TOP_K=7\nHANDOFF_SCORE_THRESHOLD=0.4\n",
    )
    settings = load_settings(env)
    assert settings.top_k == 7
    assert settings.handoff_threshold == 0.4
    assert "sk-test-secret" not in repr(settings)
    assert "sk-test-secret" not in str(settings.model_dump())


def test_environment_variable_overrides_file(tmp_path: Path, monkeypatch) -> None:
    env = write_env(tmp_path, "OPENAI_API_KEY=from-file\nOPENAI_CHAT_MODEL=gpt-4o-mini\n")
    monkeypatch.setenv("OPENAI_CHAT_MODEL", "gpt-4o")
    assert load_settings(env).chat_model == "gpt-4o"


def test_invalid_number_names_the_setting(tmp_path: Path) -> None:
    env = write_env(tmp_path, "OPENAI_API_KEY=x\nRETRIEVAL_TOP_K=lots\n")
    with pytest.raises(ConfigError, match="RETRIEVAL_TOP_K"):
        load_settings(env)
