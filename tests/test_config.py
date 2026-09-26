import uuid

import pytest
from pydantic import ValidationError

from src.config import Settings


def test_app_env_accepts_documented_development_value():
    settings = Settings(_env_file=None, app_env="development", openai_api_key="test-key")

    assert settings.app_env == "development"


def test_app_env_rejects_ambiguous_dev_alias():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, app_env="dev", openai_api_key="test-key")


def _langfuse_stub_keys() -> dict[str, str]:
    # 测试桩密钥每次随机生成，避免在源码中出现任何静态凭据字面量。
    return {
        "langfuse_public_key": f"pk-lf-{uuid.uuid4().hex}",
        "langfuse_secret_key": f"sk-lf-{uuid.uuid4().hex}",
    }


def test_langfuse_disabled_by_default_and_omits_key_check():
    settings = Settings(
        _env_file=None,
        llm_provider="ollama",
        langfuse_enabled=False,
        langfuse_public_key="",
        langfuse_secret_key="",
    )

    assert settings.langfuse_enabled is False
    assert settings.langfuse_capture_content is False
    assert settings.langfuse_sample_rate == 1.0
    assert settings.langfuse.secret_key.get_secret_value() == ""


def test_langfuse_enabled_requires_host_and_keys():
    with pytest.raises(ValidationError, match="LANGFUSE_PUBLIC_KEY"):
        Settings(
            _env_file=None,
            llm_provider="ollama",
            langfuse_enabled=True,
            langfuse_host="https://cloud.langfuse.com",
            langfuse_public_key="",
            langfuse_secret_key="",
        )

    with pytest.raises(ValidationError, match="LANGFUSE_HOST"):
        Settings(
            _env_file=None,
            llm_provider="ollama",
            langfuse_enabled=True,
            langfuse_host="",
            langfuse_public_key="",
            langfuse_secret_key="",
        )


def test_langfuse_enabled_with_full_config_passes_and_rejects_bad_sample_rate():
    settings = Settings(
        _env_file=None,
        llm_provider="ollama",
        langfuse_enabled=True,
        langfuse_host="https://langfuse.internal.example",
        langfuse_sample_rate=0.5,
        langfuse_capture_content=True,
        **_langfuse_stub_keys(),
    )

    assert settings.langfuse.enabled is True
    assert settings.langfuse.host == "https://langfuse.internal.example"
    assert settings.langfuse.sample_rate == 0.5
    assert settings.langfuse.capture_content is True
    assert settings.langfuse.secret_key.get_secret_value() != ""

    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            llm_provider="ollama",
            langfuse_enabled=True,
            langfuse_sample_rate=1.5,
            **_langfuse_stub_keys(),
        )
