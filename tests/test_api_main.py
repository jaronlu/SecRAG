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


class _CacheableAgentApp:
    """返回可入缓存的终态：答案足够长且验证/合规均通过。"""

    def invoke(self, state, config=None):
        return {
            **state,
            "final_answer": "货币基金主要投资于短期货币工具，风险等级为低。",
            "citations": [{"source": "a.pdf"}],
            "confidence": "high",
            "compliance": {"passed": True, "flags": [], "risk_disclosure": "【风险提示】投资须谨慎。"},
            "verification": {"passed": True, "issues": [], "confidence": "high"},
        }


class _StubSemanticCache:
    """命中/存储行为可编程的缓存替身。"""

    def __init__(self, hit=None):
        self.hit = hit
        self.store_calls = []

    def lookup(self, query, role=""):
        return self.hit

    def store(
        self,
        query,
        answer,
        citations=None,
        confidence="",
        role="",
        compliance=None,
        verification=None,
    ):
        self.store_calls.append({
            "query": query,
            "answer": answer,
            "citations": citations,
            "confidence": confidence,
            "role": role,
            "compliance": compliance,
            "verification": verification,
        })
        return True


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


@pytest.mark.asyncio
async def test_assistant_qa_cache_hit_returns_stored_compliance_snapshot(monkeypatch):
    """P1-1: 命中路径返回 store 时保存的合规快照，而不是硬编码 {"passed": True}。"""
    stored_compliance = {
        "passed": True,
        "flags": [],
        "risk_disclosure": "【风险提示】投资须谨慎。",
        "suitability_warning": "",
    }
    cache = _StubSemanticCache(hit={
        "query": "货币基金风险等级",
        "answer": "货币基金风险等级为低。",
        "citations": [{"source": "a.pdf"}],
        "confidence": "high",
        "similarity": 0.95,
        "hit_count": 1,
        "compliance": stored_compliance,
        "verification": {"passed": True, "issues": [], "confidence": "high"},
    })
    monkeypatch.setattr("src.api.main.get_semantic_cache", lambda: cache)

    response = await assistant_qa(
        AssistantQARequest(query="货币基金风险等级"),
        AuthenticatedUser("user_tech", ROLE_TECHNICAL, "tech"),
    )

    assert response["compliance"] == stored_compliance
    assert response["cached"] is True


@pytest.mark.asyncio
async def test_assistant_qa_stores_compliance_snapshot_on_cache_miss(monkeypatch):
    """P1-1: 入缓存时把终态 compliance/verification 快照一起写入。"""
    cache = _StubSemanticCache()
    monkeypatch.setattr("src.api.main.get_semantic_cache", lambda: cache)
    monkeypatch.setattr("src.api.main._get_agent_app", lambda: _CacheableAgentApp())

    await assistant_qa(
        AssistantQARequest(query="货币基金的风险等级是什么"),
        AuthenticatedUser("user_tech", ROLE_TECHNICAL, "tech"),
    )

    assert len(cache.store_calls) == 1
    assert cache.store_calls[0]["compliance"]["passed"] is True
    assert cache.store_calls[0]["verification"]["passed"] is True
