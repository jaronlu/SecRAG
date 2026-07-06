import pytest
from fastapi import HTTPException
from httpx import ConnectError

from src.api.auth import AuthenticatedUser
from src.api.main import assistant_qa
from src.schemas.constants import ROLE_TECHNICAL
from src.schemas.request_response import AssistantQARequest


class _FailingAgentApp:
    def __init__(self, exc: Exception):
        self.exc = exc

    def invoke(self, state, config=None):
        raise self.exc


@pytest.mark.asyncio
async def test_assistant_qa_returns_503_for_provider_connection_error(monkeypatch):
    monkeypatch.setattr(
        "src.api.main._get_agent_app",
        lambda: _FailingAgentApp(ConnectError("connection failed")),
    )

    with pytest.raises(HTTPException) as exc:
        await assistant_qa(
            AssistantQARequest(query="查询"),
            AuthenticatedUser("user_tech", ROLE_TECHNICAL, "tech"),
        )

    assert exc.value.status_code == 503


@pytest.mark.asyncio
async def test_assistant_qa_returns_500_for_internal_error(monkeypatch):
    monkeypatch.setattr(
        "src.api.main._get_agent_app",
        lambda: _FailingAgentApp(ValueError("bad state")),
    )

    with pytest.raises(HTTPException) as exc:
        await assistant_qa(
            AssistantQARequest(query="查询"),
            AuthenticatedUser("user_tech", ROLE_TECHNICAL, "tech"),
        )

    assert exc.value.status_code == 500
