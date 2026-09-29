import uuid
from typing import Any

import pytest
from pydantic import ValidationError

from src.config import LLMConfig, Settings


def _settings(**kwargs: Any) -> Settings:
    """基于 basedpyright 会从模型字段合成 Settings.__init__（不含 pydantic-settings
    的 _env_file 等下划线参数），经 **kwargs 收口；运行时仍走 BaseSettings.__init__。"""
    return Settings(**kwargs)


def test_app_env_accepts_documented_development_value():
    settings = _settings(_env_file=None, app_env="development", openai_api_key="test-key")

    assert settings.app_env == "development"


def test_app_env_rejects_ambiguous_dev_alias():
    with pytest.raises(ValidationError):
        _settings(_env_file=None, app_env="dev", openai_api_key="test-key")


def _langfuse_stub_keys() -> dict[str, str]:
    # 测试桩密钥每次随机生成，避免在源码中出现任何静态凭据字面量。
    return {
        "langfuse_public_key": f"pk-lf-{uuid.uuid4().hex}",
        "langfuse_secret_key": f"sk-lf-{uuid.uuid4().hex}",
    }


def test_langfuse_disabled_by_default_and_omits_key_check():
    settings = _settings(
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
        _settings(
            _env_file=None,
            llm_provider="ollama",
            langfuse_enabled=True,
            langfuse_host="https://cloud.langfuse.com",
            langfuse_public_key="",
            langfuse_secret_key="",
        )

    with pytest.raises(ValidationError, match="LANGFUSE_HOST"):
        _settings(
            _env_file=None,
            llm_provider="ollama",
            langfuse_enabled=True,
            langfuse_host="",
            langfuse_public_key="",
            langfuse_secret_key="",
        )


def test_langfuse_enabled_with_full_config_passes_and_rejects_bad_sample_rate():
    settings = _settings(
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
        _settings(
            _env_file=None,
            llm_provider="ollama",
            langfuse_enabled=True,
            langfuse_sample_rate=1.5,
            **_langfuse_stub_keys(),
        )


def test_llm_config_has_explicit_retry_and_token_budget_defaults():
    """ISSUE-14：LLM 客户端默认显式 max_retries=1 与 token 预算。

    langchain-openai 默认 max_retries=2 会把单 invoke 最坏耗时放大到
    3 × timeout；理解/计划合并调用另设更小的输出预算。
    """
    cfg = LLMConfig(provider="openai", base_url="http://x", model="m", temperature=0.0)

    assert cfg.max_retries == 1
    assert cfg.max_tokens == 4096
    assert cfg.plan_max_tokens == 1024
    # ISSUE-20：分层选型默认未配置（空串回落主模型），按 benchmark 结果在 .env 配置
    assert cfg.plan_model == ""
