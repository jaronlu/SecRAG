"""HTTP 契约测试：通过 TestClient 发真实 ASGI 请求验证路由优先级。

issues.md 一.2：SPA 通配路由必须注册在业务路由之后，否则会截走
GET /health、GET /metrics 等接口。直接调用 health_check() 无法发现该类问题，
必须走完整 HTTP 栈。
"""

import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from src.api.auth import AuthenticatedUser, authenticate_user
from src.api.main import app
from src.schemas.constants import (
    API_ROUTE_ASSISTANT_QA,
    API_ROUTE_ASSISTANT_QA_STREAM,
    API_ROUTE_ASSISTANT_THREADS,
    ROLE_TECHNICAL,
    STATE_CITATIONS,
    STATE_COMPLIANCE,
    STATE_CONFIDENCE,
    STATE_FINAL_ANSWER,
    STATE_INTENT,
    STATE_RETRIEVAL_PLAN,
    STATE_TERMINAL,
    STATE_VERIFICATION,
)
from src.config import config
from src.product_scope import get_product_profile
from src.schemas.constants import ROLE_OPERATIONS
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
    # ENV-2：部署时效校验——/health 必须携带部署版本，发布流程据此核对
    # 部署代码 == HEAD；测试仓库内 git 可用，应解析出真实提交
    version = body["version"]
    assert version["commit"] != "unknown"
    assert isinstance(version["dirty"], bool)


# ══════════════════════════════════════════════════════════════════════
# /v1/assistant/me：角色由 Bearer token 服务端派生，请求体不参与身份
# ══════════════════════════════════════════════════════════════════════


def test_current_user_returns_server_derived_role(client):
    response = client.get(
        "/v1/assistant/me",
        headers={"Authorization": "Bearer demo-ops"},
    )

    assert response.status_code == 200
    assert response.json()["role"] == ROLE_OPERATIONS
    assert response.json()["department"] == "ops"
    assert response.json()["product_profile"] == config.product_profile


def test_current_user_exposes_no_identity_beyond_role_department_profile(client):
    """对外只暴露 role/department/product_profile；user_id、token、权限列表不得出现。"""
    response = client.get(
        "/v1/assistant/me",
        headers={"Authorization": "Bearer demo-ops"},
    )

    assert response.status_code == 200
    assert set(response.json()) == {"role", "department", "product_profile"}


def test_current_user_requires_bearer_token(client):
    response = client.get("/v1/assistant/me")

    assert response.status_code == 401


def test_current_user_rejects_unknown_token(client):
    response = client.get(
        "/v1/assistant/me",
        headers={"Authorization": "Bearer not-a-known-token"},
    )

    assert response.status_code == 401


def test_current_user_product_profile_matches_active_profile(client):
    """profile 字段与配置解析结果一致，非法配置在启动期已 fail closed。"""
    response = client.get(
        "/v1/assistant/me",
        headers={"Authorization": "Bearer demo-ops"},
    )

    assert response.status_code == 200
    assert response.json()["product_profile"] == get_product_profile()


# ══════════════════════════════════════════════════════════════════════
# 管理路由边界：非 technical 角色 403（服务端独立校验，前端隐藏入口不算权限）
# ════════════════════════════════════════════════════════════════════


def test_admin_document_routes_reject_non_technical_roles(client):
    """文档管理/缓存运维接口对非 technical 角色返回 403（角色校验先于业务逻辑）。"""
    for token in ("demo-advisor", "demo-ops"):
        headers = {"Authorization": f"Bearer {token}"}
        assert client.get("/v1/admin/documents", headers=headers).status_code == 403
        assert client.get("/v1/admin/documents/stats", headers=headers).status_code == 403
        assert client.get("/v1/admin/cache/stats", headers=headers).status_code == 403
        # 危险操作同样先被角色校验拦截
        assert client.post("/v1/admin/cache/clear", headers=headers).status_code == 403


def test_admin_document_routes_allow_technical_role(client, monkeypatch):
    """technical 角色可以访问文档管理接口（业务依赖打桩，不触碰真实知识库）。"""

    class _FakeKBManager:
        def list_documents(self, doc_type=None, limit=100, offset=0):
            return {"total": 0, "documents": []}

        def get_stats(self):
            return {"total_chunks": 0, "total_documents": 0}

        def get_document_chunks(self, source, limit=50, offset=0):
            return {"source": source, "total_chunks": 0, "chunks": []}

        def search_documents(self, query, top_k=5):
            return []

    class _FakeCache:
        def get_stats(self):
            return {"total_entries": 0}

    monkeypatch.setattr("src.utils.knowledge_base.get_kb_manager", lambda: _FakeKBManager())
    monkeypatch.setattr("src.api.admin.get_semantic_cache", lambda: _FakeCache())

    headers = {"Authorization": "Bearer demo-tech"}
    assert client.get("/v1/admin/documents", headers=headers).status_code == 200
    assert client.get("/v1/admin/documents/stats", headers=headers).status_code == 200
    assert (
        client.get("/v1/admin/documents/chunks", headers=headers, params={"source": "a.pdf"}).status_code
        == 200
    )
    assert (
        client.get("/v1/admin/documents/search", headers=headers, params={"query": "货币基金"}).status_code
        == 200
    )
    assert client.get("/v1/admin/cache/stats", headers=headers).status_code == 200


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

    def lookup(self, query, role="", **kwargs):
        return None

    def store(self, query, answer, citations=None, confidence="", role="", **kwargs):
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
    """按 stream_mode=["updates","messages"] + subgraphs=True 契约产出的事件源替身。

    ISSUE-9 前 stream_mode="updates" 的产出是裸 dict；ISSUE-9 起 API 层以
    subgraphs=True + 双模式拉流，每项为 (namespace, mode, data) 三元组。
    """

    async def astream(
        self, initial_state, config=None, stream_mode="updates", subgraphs=False
    ):
        assert stream_mode == ["updates", "messages"]
        assert subgraphs is True
        yield ((), "updates", {"query_understand": {STATE_INTENT: "FAQ"}})
        yield ((), "updates", {"planner": {STATE_RETRIEVAL_PLAN: []}})
        yield (
            (),
            "updates",
            {
                "compose": {
                    STATE_FINAL_ANSWER: "货币基金风险等级为低。",
                    STATE_TERMINAL: True,
                    STATE_CITATIONS: [{"source": "a.pdf"}],
                    STATE_CONFIDENCE: "high",
                }
            },
        )


class _MessageChunk:
    """AIMessageChunk 形状替身：只携带流式文本内容。"""

    def __init__(self, content: str):
        self.content = content


class _TokenStreamingAgentApp:
    """模拟 reason 子图内 LLM token 流 + 外层节点 updates 的混合事件流。

    契约与 langgraph ``astream(subgraphs=True, stream_mode=["updates","messages"])``
    实测一致：子图内 item 的 namespace 非空、外层为空元组；
    messages 项为 (message_chunk, metadata)，metadata.langgraph_node 标明来源节点。
    """

    async def astream(
        self, initial_state, config=None, stream_mode="updates", subgraphs=False
    ):
        yield ((), "updates", {"query_understand": {STATE_INTENT: "FAQ"}})
        # 子图内部节点更新：namespace 非空，不得外发为 progress
        yield (("reason:abc",), "updates", {"call_reason_model": {}})
        # reason 子图 LLM token：只允许 call_reason_model 来源外发
        yield (
            ("reason:abc",),
            "messages",
            (_MessageChunk("货币基金"), {"langgraph_node": "call_reason_model"}),
        )
        yield (
            ("reason:abc",),
            "messages",
            (_MessageChunk("风险等级为低。"), {"langgraph_node": "call_reason_model"}),
        )
        # 其他节点的 LLM token（query_understand/planner 的 JSON 输出）不得外发
        yield (
            (),
            "messages",
            (_MessageChunk("SHOULD_NOT_LEAK"), {"langgraph_node": "query_understand"}),
        )
        yield (
            (),
            "updates",
            {
                "compose": {
                    STATE_FINAL_ANSWER: "货币基金风险等级为低。",
                    STATE_TERMINAL: True,
                    STATE_CITATIONS: [{"source": "a.pdf"}],
                    STATE_CONFIDENCE: "high",
                }
            },
        )


@pytest.fixture()
def qa_client(monkeypatch, tmp_path):
    """带鉴权覆盖与隔离存储的 QA 客户端：走真实路由与依赖注入。"""
    store = SQLiteConversationStore(tmp_path / "conversations.db")
    monkeypatch.setattr("src.api.qa._get_conversation_store", lambda: store)
    monkeypatch.setattr("src.api.qa.get_semantic_cache", lambda: _DisabledCache())
    monkeypatch.setattr(
        "src.api.qa._get_cache_hit_audit_store",
        lambda: SQLiteAuditStore(tmp_path / "audit.db"),
    )
    app.dependency_overrides[authenticate_user] = lambda: AuthenticatedUser(
        "user_tech", ROLE_TECHNICAL, "tech"
    )
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_qa_endpoint_rate_limit_returns_429(qa_client, monkeypatch):
    monkeypatch.setattr("src.api.qa.check_rate_limit", lambda key: (False, 0))

    res = qa_client.post(API_ROUTE_ASSISTANT_QA, json={"query": "货币基金风险"})

    assert res.status_code == 429
    assert res.headers.get("retry-after") == "60"


def test_qa_endpoint_timeout_returns_504(qa_client, monkeypatch):
    monkeypatch.setattr("src.api.qa._get_agent_app", lambda: _SlowAgentApp())
    monkeypatch.setattr("src.api.qa.config", type("C", (), {"api_request_timeout_seconds": 0.05}))

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
        def lookup(self, query, role="", **kwargs):
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

        def store(self, query, answer, citations=None, confidence="", role="", **kwargs):
            return True

    monkeypatch.setattr("src.api.qa.get_semantic_cache", lambda: _HitCache())

    res = qa_client.post(API_ROUTE_ASSISTANT_QA, json={"query": "货币基金风险等级"})

    assert res.status_code == 200
    body = res.json()
    assert body["compliance"] == stored_compliance
    # response_model 契约：缓存内部字段不得泄露到 HTTP 响应
    assert "cached" not in body
    assert "cache_similarity" not in body


def test_qa_stream_emits_terminal_event_protocol(qa_client, monkeypatch):
    monkeypatch.setattr("src.api.qa._get_agent_app", lambda: _StreamingAgentApp())

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
    # SSE answer 事件必须携带 answer_status；替身终态未带验证/合规结果，
    # 缺信息按 fail closed 收敛为 blocked，不得伪装成 answered
    assert answer["answer_status"] == "blocked"


def _parse_sse_events(res) -> list[tuple[str, dict]]:
    events = []
    current_event = None
    for line in res.iter_lines():
        if line.startswith("event: "):
            current_event = line[len("event: "):]
        elif line.startswith("data: ") and current_event:
            events.append((current_event, json.loads(line[len("data: "):])))
            current_event = None
    return events


def test_qa_stream_emits_answer_delta_for_reason_tokens(qa_client, monkeypatch):
    """ISSUE-9：reason 节点 LLM token 以 answer_delta 事件先行流出，
    既有 progress/answer/done 事件保持兼容；非 reason 节点的 token 不外发。"""
    monkeypatch.setattr("src.api.qa._get_agent_app", lambda: _TokenStreamingAgentApp())

    with qa_client.stream(
        "POST", API_ROUTE_ASSISTANT_QA_STREAM, json={"query": "货币基金风险"}
    ) as res:
        assert res.status_code == 200
        events = _parse_sse_events(res)

    names = [name for name, _ in events]
    assert names[-1] == "done"
    assert "answer" in names

    deltas = [data.get("delta") or "" for name, data in events if name == "answer_delta"]
    assert deltas == ["货币基金", "风险等级为低。"]
    # answer_delta 全部先于 answer 终态事件
    assert names.index("answer_delta") < names.index("answer")
    for name, data in events:
        assert data["type"] == name

    # 既有 progress 契约不受影响：外层节点仍发 progress，子图内部节点不发
    progress_nodes = [data.get("node") for name, data in events if name == "progress"]
    assert "query_understand" in progress_nodes
    assert "call_reason_model" not in progress_nodes

    # 终态 answer 事件承载完整文本（前端以终态为准覆盖流式内容）
    answer = next(data for name, data in events if name == "answer")
    assert answer["answer"] == "货币基金风险等级为低。"
    # 非 reason 节点的 token 不得出现在任何 delta 中
    assert "SHOULD_NOT_LEAK" not in "".join(deltas)


# ══════════════════════════════════════════════════════════════════════
# answer_status 对外契约：同步响应与 SSE answer 事件都携带明确终态，
# 无证据/无权限/澄清/合规拦截不得伪装成成功答案
# ══════════════════════════════════════════════════════════════════════


class _VerifiedAgentApp:
    """图终态带通过验证与合规的完整 state。"""

    def invoke(self, state, config=None):
        return {
            **state,
            STATE_FINAL_ANSWER: "货币基金风险等级为低。",
            STATE_CITATIONS: [{"source": "a.pdf"}],
            STATE_CONFIDENCE: "high",
            STATE_VERIFICATION: {"passed": True, "issues": [], "confidence": "high"},
            STATE_COMPLIANCE: {"passed": True, "flags": [], "risk_disclosure": ""},
        }


class _SnapshotStreamingAgentApp:
    """产出 verify/compliance_check 前置更新 + compose 终态，覆盖快照解析。"""

    async def astream(self, initial_state, config=None, stream_mode="updates", subgraphs=False):
        yield ((), "updates", {"query_understand": {STATE_INTENT: "FAQ"}})
        yield ((), "updates", {
            "verify": {STATE_VERIFICATION: {"passed": True, "issues": [], "confidence": "high"}},
        })
        yield ((), "updates", {
            "compliance_check": {STATE_COMPLIANCE: {"passed": True, "flags": []}},
        })
        yield ((), "updates", {
            "compose": {
                STATE_FINAL_ANSWER: "货币基金风险等级为低。",
                STATE_TERMINAL: True,
                STATE_CITATIONS: [{"source": "a.pdf"}],
                STATE_CONFIDENCE: "high",
            },
        })


class _DeniedStreamingAgentApp:
    """权限拒绝短路终态：verification.issues 携带 permission_denied。"""

    async def astream(self, initial_state, config=None, stream_mode="updates", subgraphs=False):
        yield ((), "updates", {
            "permission_denied_response": {
                STATE_FINAL_ANSWER: "当前角色无权限访问完成该请求所需的数据源。",
                STATE_TERMINAL: True,
                STATE_CITATIONS: [],
                STATE_CONFIDENCE: "low",
                STATE_VERIFICATION: {
                    "passed": False,
                    "issues": ["permission_denied"],
                    "confidence": "low",
                },
            },
        })


def _collect_answer_event(qa_client, app, monkeypatch) -> dict:
    monkeypatch.setattr("src.api.qa._get_agent_app", lambda: app())
    with qa_client.stream(
        "POST", API_ROUTE_ASSISTANT_QA_STREAM, json={"query": "货币基金风险"}
    ) as res:
        assert res.status_code == 200
        events = _parse_sse_events(res)
    return next(data for name, data in events if name == "answer")


def test_qa_response_carries_answer_status(qa_client, monkeypatch):
    """同步响应携带 answer_status；验证与合规都通过时为 answered。"""
    monkeypatch.setattr("src.api.qa._get_agent_app", lambda: _VerifiedAgentApp())

    res = qa_client.post(API_ROUTE_ASSISTANT_QA, json={"query": "货币基金风险"})

    assert res.status_code == 200
    body = res.json()
    assert body["answer_status"] == "answered"


def test_qa_response_sanitizes_citation_source_path(qa_client, monkeypatch):
    """ISSUE-34：对外 payload 的 citation.source 只含文件名，不泄露服务端绝对路径。"""
    from src.api.qa import _qa_response_from_outcome
    from src.schemas.typed_dicts import AnswerOutcome

    outcome = AnswerOutcome(
        answer="货币基金风险等级为低。",
        citations=[
            {
                "citation_id": "cite_001",
                "source": "/Users/ryan/workspace/data/raw/faq/货币基金FAQ.pdf",
                "doc_type": "faq",
                "quote": "货币基金的风险等级为低风险。",
            }
        ],
        confidence="high",
        compliance={"passed": True, "flags": []},
        answer_status="answered",
    )

    response = _qa_response_from_outcome(outcome, "thread-1", "turn-1")

    citation = response.citations[0]
    assert citation["source"] == "货币基金FAQ.pdf"
    # 原文摘句与编号原样透传，供前端就地核对引用
    assert citation["quote"] == "货币基金的风险等级为低风险。"
    assert citation["citation_id"] == "cite_001"


def test_qa_stream_answer_status_uses_accumulated_state(qa_client, monkeypatch):
    """SSE answer_status 由终态输出 + 前置 verify/compliance 更新合并解析。"""
    answer = _collect_answer_event(qa_client, _SnapshotStreamingAgentApp, monkeypatch)

    assert answer["answer_status"] == "answered"


def test_qa_stream_permission_denied_maps_to_permission_denied(qa_client, monkeypatch):
    answer = _collect_answer_event(qa_client, _DeniedStreamingAgentApp, monkeypatch)

    assert answer["answer_status"] == "permission_denied"
    assert answer["citations"] == []


def test_qa_stream_records_time_to_first_token_once(qa_client, monkeypatch):
    """ISSUE-23：TTFT 在首个 answer_delta 记录一次，供设计线 P95 ≤5s 评估。"""
    import src.api.qa as api_qa
    from src.utils.metrics import MetricsRegistry

    registry = MetricsRegistry()
    monkeypatch.setattr(api_qa, "get_metrics", lambda: registry)
    monkeypatch.setattr("src.api.qa._get_agent_app", lambda: _TokenStreamingAgentApp())

    with qa_client.stream(
        "POST", API_ROUTE_ASSISTANT_QA_STREAM, json={"query": "货币基金风险"}
    ) as res:
        assert res.status_code == 200
        events = _parse_sse_events(res)

    assert any(name == "answer_delta" for name, _ in events)
    assert registry.time_to_first_token.get_count() == 1
    assert registry.get_summary()["ttft_p95_seconds"] >= 0.0


# ══════════════════════════════════════════════════════════════════════
# ISSUE-18：async 路由内的同步 SQLite/Chroma 调用必须移出事件循环
# ══════════════════════════════════════════════════════════════════════


def _running_loop_probe() -> bool:
    """在事件循环线程内返回 True；asyncio.to_thread 工作线程内抛 RuntimeError。"""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


class _LoopRecordingStore:
    """记录每次调用是否阻塞事件循环的会话存储替身。"""

    def __init__(self):
        self.flags: list[bool] = []

    def _probe(self) -> None:
        self.flags.append(_running_loop_probe())

    def ensure_thread_for_qa(self, **kwargs):
        self._probe()
        return {"thread_id": "thread-x", "turn_count": 0}

    def list_threads(self, user_id=None, limit=None):
        self._probe()
        return []

    def create_thread(self, **kwargs):
        self._probe()
        return {"thread_id": "thread-x", "title": "t", "created_at": "now"}

    def list_messages(self, thread_id=None, user_id=None):
        self._probe()
        return []

    def soft_delete_thread(self, thread_id=None, user_id=None):
        self._probe()


def test_conversation_routes_offload_sync_io(qa_client, monkeypatch):
    """ISSUE-18：会话 CRUD 不得在事件循环内执行同步 SQLite 调用。"""
    store = _LoopRecordingStore()
    monkeypatch.setattr("src.api.conversation._get_conversation_store", lambda: store)

    assert qa_client.get(API_ROUTE_ASSISTANT_THREADS).status_code == 200
    assert qa_client.post(API_ROUTE_ASSISTANT_THREADS, json={"title": "t"}).status_code == 200
    assert qa_client.get(f"{API_ROUTE_ASSISTANT_THREADS}/thread-x/messages").status_code == 200
    assert qa_client.delete(f"{API_ROUTE_ASSISTANT_THREADS}/thread-x").status_code == 204

    assert store.flags == [False, False, False, False]


def test_qa_routes_offload_thread_ensure(qa_client, monkeypatch):
    store = _LoopRecordingStore()
    monkeypatch.setattr("src.api.qa._get_conversation_store", lambda: store)
    monkeypatch.setattr("src.api.qa._get_agent_app", lambda: _StreamingAgentApp())

    with qa_client.stream(
        "POST", API_ROUTE_ASSISTANT_QA_STREAM, json={"query": "货币基金风险"}
    ) as res:
        assert res.status_code == 200

    assert store.flags == [False]


def test_health_probe_offloads_chroma_count(client, monkeypatch):
    """ISSUE-18：health 的 Chroma count() 探针不得阻塞事件循环。"""
    from types import SimpleNamespace

    flags: list[bool] = []

    class _FakeEngine:
        def __init__(self):
            flags.append(_running_loop_probe())
            self.collection = SimpleNamespace(count=lambda: 42)

    monkeypatch.setattr(
        "src.retrieval.vector_retriever.ChromaVectorRetriever", _FakeEngine
    )

    res = client.get("/health")

    assert res.status_code == 200
    assert res.json()["chroma"] == {"status": "ok", "doc_count": 42}
    assert flags == [False]
