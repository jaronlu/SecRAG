"""HTTP 契约测试：通过 TestClient 发真实 ASGI 请求验证路由优先级。

issues.md 一.2：SPA 通配路由必须注册在业务路由之后，否则会截走
GET /health、GET /metrics 等接口。直接调用 health_check() 无法发现该类问题，
必须走完整 HTTP 栈。
"""

import json

import pytest
from fastapi.testclient import TestClient

from src.api.auth import AuthenticatedUser, authenticate_user
from src.api.main import app
from src.schemas.constants import (
    API_ROUTE_ASSISTANT_QA,
    API_ROUTE_ASSISTANT_QA_STREAM,
    ROLE_TECHNICAL,
    STATE_CITATIONS,
    STATE_CONFIDENCE,
    STATE_FINAL_ANSWER,
    STATE_INTENT,
    STATE_RETRIEVAL_PLAN,
    STATE_TERMINAL,
)
from src.utils.audit import SQLiteAuditStore
from src.utils.conversation import SQLiteConversationStore


@pytest.fixture()
def client():
    return TestClient(app)


def test_health_endpoint_reachable(client):
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"


def test_metrics_endpoint_reachable(client):
    res = client.get("/metrics")
    assert res.status_code == 200
    assert "text/plain" in res.headers["content-type"]


def test_unknown_api_path_returns_404_not_index_html(client):
    res = client.get("/v1/nonexistent")
    assert res.status_code == 404


def test_favicon_returns_204(client):
    res = client.get("/favicon.ico")
    assert res.status_code == 204


def test_root_serves_html(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]


# ══════════════════════════════════════════════════════════════════════
# QA 主端点 HTTP 契约（P2-1）：真实路由 + 依赖注入 + 事件字节可消费
# ══════════════════════════════════════════════════════════════════════


class _DisabledCache:
    """默认关闭的缓存替身：lookup 永不命中。"""

    def lookup(self, query, role=""):
        return None

    def store(self, query, answer, citations=None, confidence="", role=""):
        return True


class _SuccessfulAgentApp:
    def invoke(self, state, config=None):
        return {
            **state,
            STATE_FINAL_ANSWER: "货币基金风险等级为低。",
            STATE_CITATIONS: [{"source": "a.pdf"}],
            STATE_CONFIDENCE: "high",
        }


class _SlowAgentApp:
    def invoke(self, state, config=None):
        import time

        time.sleep(0.5)
        return {**state, STATE_FINAL_ANSWER: "太慢", STATE_CITATIONS: [], STATE_CONFIDENCE: "low"}


class _StreamingAgentApp:
    """按 stream_mode="updates" 产出节点更新的事件源替身。"""

    async def astream(self, initial_state, config=None, stream_mode="updates"):
        yield {"query_understand": {STATE_INTENT: "FAQ"}}
        yield {"planner": {STATE_RETRIEVAL_PLAN: []}}
        yield {
            "compose": {
                STATE_FINAL_ANSWER: "货币基金风险等级为低。",
                STATE_TERMINAL: True,
                STATE_CITATIONS: [{"source": "a.pdf"}],
                STATE_CONFIDENCE: "high",
            }
        }


@pytest.fixture()
def qa_client(monkeypatch, tmp_path):
    """带鉴权覆盖与隔离存储的 QA 客户端：走真实路由与依赖注入。"""
    store = SQLiteConversationStore(tmp_path / "conversations.db")
    monkeypatch.setattr("src.api.main._get_conversation_store", lambda: store)
    monkeypatch.setattr("src.api.main.get_semantic_cache", lambda: _DisabledCache())
    monkeypatch.setattr(
        "src.api.main._get_cache_hit_audit_store",
        lambda: SQLiteAuditStore(tmp_path / "audit.db"),
    )
    app.dependency_overrides[authenticate_user] = lambda: AuthenticatedUser(
        "user_tech", ROLE_TECHNICAL, "tech"
    )
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_qa_endpoint_rate_limit_returns_429(qa_client, monkeypatch):
    monkeypatch.setattr("src.api.main.check_rate_limit", lambda key: (False, 0))

    res = qa_client.post(API_ROUTE_ASSISTANT_QA, json={"query": "货币基金风险"})

    assert res.status_code == 429
    assert res.headers.get("retry-after") == "60"


def test_qa_endpoint_timeout_returns_504(qa_client, monkeypatch):
    monkeypatch.setattr("src.api.main._get_agent_app", lambda: _SlowAgentApp())
    monkeypatch.setattr("src.api.main.config", type("C", (), {"api_request_timeout_seconds": 0.05}))

    res = qa_client.post(API_ROUTE_ASSISTANT_QA, json={"query": "货币基金风险"})

    assert res.status_code == 504


def test_qa_endpoint_cache_hit_returns_stored_compliance(qa_client, monkeypatch):
    stored_compliance = {
        "passed": True,
        "flags": [],
        "risk_disclosure": "",
        "suitability_warning": "",
    }

    class _HitCache:
        def lookup(self, query, role=""):
            return {
                "query": query,
                "answer": "货币基金风险等级为低。",
                "citations": [{"source": "a.pdf"}],
                "confidence": "high",
                "similarity": 0.95,
                "hit_count": 1,
                "compliance": stored_compliance,
                "verification": {"passed": True},
            }

        def store(self, query, answer, citations=None, confidence="", role=""):
            return True

    monkeypatch.setattr("src.api.main.get_semantic_cache", lambda: _HitCache())

    res = qa_client.post(API_ROUTE_ASSISTANT_QA, json={"query": "货币基金风险等级"})

    assert res.status_code == 200
    body = res.json()
    assert body["compliance"] == stored_compliance
    # response_model 契约：缓存内部字段不得泄露到 HTTP 响应
    assert "cached" not in body
    assert "cache_similarity" not in body


def test_qa_stream_emits_terminal_event_protocol(qa_client, monkeypatch):
    monkeypatch.setattr("src.api.main._get_agent_app", lambda: _StreamingAgentApp())

    events = []
    with qa_client.stream(
        "POST", API_ROUTE_ASSISTANT_QA_STREAM, json={"query": "货币基金风险"}
    ) as res:
        assert res.status_code == 200
        assert res.headers["content-type"].startswith("text/event-stream")
        current_event = None
        for line in res.iter_lines():
            if line.startswith("event: "):
                current_event = line[len("event: "):]
            elif line.startswith("data: ") and current_event:
                events.append((current_event, json.loads(line[len("data: "):])))
                current_event = None

    names = [name for name, _ in events]
    assert "answer" in names
    assert names[-1] == "done"

    progress_nodes = [data.get("node") for name, data in events if name == "progress"]
    assert {"query_understand", "planner", "compose"} <= set(progress_nodes)
    for name, data in events:
        assert data["type"] == name

    answer = next(data for name, data in events if name == "answer")
    assert answer["answer"] == "货币基金风险等级为低。"
    assert answer["citations"] == [{"source": "a.pdf"}]
    assert answer["confidence"] == "high"
    assert answer["thread_id"]
    assert answer["turn_id"]
