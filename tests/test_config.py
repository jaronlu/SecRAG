import pytest
from pydantic import ValidationError

from src.config import Settings


def test_app_env_accepts_documented_development_value():
    settings = Settings(_env_file=None, app_env="development", openai_api_key="test-key")

    assert settings.app_env == "development"


def test_app_env_rejects_ambiguous_dev_alias():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, app_env="dev", openai_api_key="test-key")
