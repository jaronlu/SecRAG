import pytest
from fastapi import HTTPException
from httpx import ConnectError

from src.api.auth import AuthenticatedUser
from src.api.main import app, assistant_qa
from src.api.ui import render_ui_html
from src.schemas.constants import AGENT_RECURSION_LIMIT, ROLE_TECHNICAL
from src.schemas.request_response import AssistantQARequest
from src.utils.conversation import SQLiteConversationStore


class _FailingAgentApp:
    def __init__(self, exc: Exception):
        self.exc = exc

    def invoke(self, state, config=None):
        raise self.exc


class _SuccessfulAgentApp:
    def __init__(self):
        self.config = None

    def invoke(self, state, config=None):
        self.config = config
        return {
            **state,
            "final_answer": "ok",
            "citations": [],
            "confidence": "high",
            "compliance": {"passed": True},
        }


@pytest.fixture(autouse=True)
def _temp_conversation_store(monkeypatch, tmp_path):
    store = SQLiteConversationStore(tmp_path / "conversations.db")
    monkeypatch.setattr("src.api.main._get_conversation_store", lambda: store)


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


def test_legacy_basic_qa_route_is_not_registered():
    assert "/v1/qa" not in {getattr(route, "path", None) for route in app.routes}


def test_assistant_response_and_ui_do_not_expose_audit_trail():
    assert "audit_trail" not in app.openapi()["components"]["schemas"]["AssistantQAResponse"][
        "properties"
    ]
    assert "auditText" not in render_ui_html()


@pytest.mark.asyncio
async def test_assistant_qa_uses_graph_recursion_limit_for_multi_hop_flow(monkeypatch):
    agent = _SuccessfulAgentApp()
    monkeypatch.setattr("src.api.main._get_agent_app", lambda: agent)

    response = await assistant_qa(
        AssistantQARequest(query="查询"),
        AuthenticatedUser("user_tech", ROLE_TECHNICAL, "tech"),
    )

    assert response.answer == "ok"
    assert "audit_trail" not in response.model_dump()
    assert agent.config["recursion_limit"] == AGENT_RECURSION_LIMIT
    assert AGENT_RECURSION_LIMIT >= 40
