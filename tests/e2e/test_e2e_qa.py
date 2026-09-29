"""全链路测试案例：环节 D QA 问答含 SSE（TC-016~TC-023）。

TC-016/018 为 Graph 级 E2E：真实 Agent Graph + 真实验证器/合规器 +
隔离存储，仅 LLM 与向量检索为测试替身。
"""

from __future__ import annotations

import json
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient

from src.api.auth import AuthenticatedUser, authenticate_user
from src.api.main import app
from src.schemas.constants import (
    API_ROUTE_ASSISTANT_QA,
    API_ROUTE_ASSISTANT_QA_STREAM,
    AUDIT_REQUEST_ID,
    CONFIDENCE_LOW,
    CONFIDENCE_MEDIUM,
    ROLE_ADVISOR,
    STATE_CITATIONS,
    STATE_COMPLIANCE,
    STATE_CONFIDENCE,
    STATE_FINAL_ANSWER,
    STATE_REASON_ATTEMPTS,
    STATE_RETRIEVAL_RESULTS,
    STATE_THREAD_ID,
    STATE_VERIFICATION,
    STATE_AUDIT_TRAIL,
    STATE_USER_ID,
)


# ══════════════════════════════════════════════════════════════════════
# TC-016 QA 正常全链路（Graph 级 E2E）
# ══════════════════════════════════════════════════════════════════════


def test_tc016_qa_full_chain_happy_path(run_agent_graph, isolated_stores, fake_llm):
    """TC-016：查询 → 检索 → 推理 → 验证 → 合规 → 会话/审计落库 全链路成功。

    真实链路中 API 层先 ensure 会话再调图；这里同样先建会话再执行。
    """
    from tests.e2e.conftest import build_state

    thread = isolated_stores.conversation.create_thread(
        user_id="user_advisor", user_role=ROLE_ADVISOR, client_id=None, title="TC-016"
    )
    state = run_agent_graph(build_state(thread_id=thread["thread_id"]))

    # 终态答案：结构化 Markdown + [来源1] 引用
    assert state[STATE_FINAL_ANSWER].startswith("## 结论")
    assert "[来源1]" in state[STATE_FINAL_ANSWER]
    assert state[STATE_CITATIONS], "成功链路必须产出引用"
    assert state[STATE_CITATIONS][0]["source"].endswith("xx_money_fund_2024.html")
    assert state[STATE_CONFIDENCE] in (CONFIDENCE_MEDIUM, "high")

    # 验证与合规均通过
    assert state[STATE_VERIFICATION]["passed"] is True
    assert state[STATE_COMPLIANCE]["passed"] is True

    # 会话落库：同一用户可见本轮问答
    messages = isolated_stores.conversation.list_messages(
        thread_id=state[STATE_THREAD_ID], user_id=state[STATE_USER_ID]
    )
    roles = [m["role"] for m in messages]
    assert "assistant" in roles
    assistant_content = " ".join(
        m["content"] for m in messages if m["role"] == "assistant"
    )
    assert "风险等级" in assistant_content

    # 审计落库：按 request_id 可取完整审计条目
    request_id = state[STATE_AUDIT_TRAIL][AUDIT_REQUEST_ID]
    trail = isolated_stores.audit.get_by_request_id(request_id)
    assert trail is not None
    assert trail["query"]["original"] == "XX货币市场基金的风险等级是什么？"
    assert trail["compliance"]["passed"] is True
    assert trail["reasoning"]["execution_path"][-1] == "audit_log"


# ══════════════════════════════════════════════════════════════════════
# TC-017 SSE 流式事件协议
# ══════════════════════════════════════════════════════════════════════


class _StreamingAgentApp:
    """按 stream_mode=["updates","messages"] + subgraphs=True 契约产出的 Agent 替身。"""

    async def astream(
        self, initial_state, config=None, stream_mode="updates", subgraphs=False
    ):
        yield ((), "updates", {"query_understand": {"intent": "产品咨询"}})
        yield ((), "updates", {"planner": {"retrieval_plan": []}})
        yield ((), "updates", {"retrieve": {"retrieval_results": [1]}})
        yield ((), "updates", {"grade_and_filter": {"retrieval_results": [1]}})
        yield ((), "updates", {"reason": {"final_answer": "x"}})
        yield (
            (),
            "updates",
            {
                "compose": {
                    "final_answer": "货币基金风险等级为低。",
                    "terminal": True,
                    "citations": [{"source": "a.pdf"}],
                    "confidence": "high",
                }
            },
        )


@pytest.fixture()
def sse_client(monkeypatch, tmp_path):
    from src.utils.audit import SQLiteAuditStore
    from src.utils.conversation import SQLiteConversationStore

    store = SQLiteConversationStore(tmp_path / "conversations.db")
    monkeypatch.setattr("src.api.main._get_conversation_store", lambda: store)
    monkeypatch.setattr(
        "src.api.main._get_cache_hit_audit_store",
        lambda: SQLiteAuditStore(tmp_path / "audit.db"),
    )
    monkeypatch.setattr("src.api.main._get_agent_app", lambda: _StreamingAgentApp())
    app.dependency_overrides[authenticate_user] = lambda: AuthenticatedUser(
        "user_advisor", ROLE_ADVISOR, "wealth"
    )
    yield TestClient(app)
    app.dependency_overrides.clear()


def _parse_sse(lines: list[str]) -> list[tuple[str, dict[str, Any]]]:
    events = []
    current: str | None = None
    for line in lines:
        if line.startswith("event: "):
            current = line[len("event: "):]
        elif line.startswith("data: ") and current:
            events.append((current, json.loads(line[len("data: "):])))
            current = None
    return events


def test_tc017_sse_event_protocol(sse_client):
    """TC-017：SSE 事件序列 progress…→answer→done；data.type 与 event 名一致。"""
    with sse_client.stream(
        "POST", API_ROUTE_ASSISTANT_QA_STREAM, json={"query": "货币基金风险"}
    ) as res:
        assert res.status_code == 200
        assert res.headers["content-type"].startswith("text/event-stream")
        events = _parse_sse(list(res.iter_lines()))

    names = [name for name, _ in events]
    assert names[0] == "progress"
    assert names[-1] == "done"
    assert "answer" in names

    progress_nodes = [data["node"] for name, data in events if name == "progress"]
    assert {"query_understand", "planner", "retrieve", "grade_and_filter", "compose"} <= set(
        progress_nodes
    )
    for name, data in events:
        assert data["type"] == name, "data.type 必须与 event 名一致"

    answer = next(data for name, data in events if name == "answer")
    assert answer["answer"] == "货币基金风险等级为低。"
    assert answer["citations"] == [{"source": "a.pdf"}]
    assert answer["confidence"] == "high"
    assert answer["thread_id"] and answer["turn_id"]


def test_sse_terminal_answer_follows_final_answer_not_node_name(sse_client, monkeypatch):
    """终态以节点声明的 terminal 标记为准，不按节点名白名单推断；
    新增的拒答/澄清路径按契约产出 final_answer + terminal 即自动发送 answer 事件。"""

    class _NewTerminalPathAgent:
        async def astream(
            self, initial_state, config=None, stream_mode="updates", subgraphs=False
        ):
            yield (
                (),
                "updates",
                {
                    "custom_refusal_path": {
                        "final_answer": "权限不足，无法回答该问题。",
                        "terminal": True,
                    }
                },
            )

    monkeypatch.setattr("src.api.main._get_agent_app", lambda: _NewTerminalPathAgent())

    with sse_client.stream(
        "POST", API_ROUTE_ASSISTANT_QA_STREAM, json={"query": "货币基金风险"}
    ) as res:
        assert res.status_code == 200
        events = _parse_sse(list(res.iter_lines()))

    names = [name for name, _ in events]
    assert "answer" in names
    assert names[-1] == "done"
    answer = next(data for name, data in events if name == "answer")
    assert answer["answer"] == "权限不足，无法回答该问题。"


# ══════════════════════════════════════════════════════════════════════
# TC-018 验证失败重试与安全兜底
# ══════════════════════════════════════════════════════════════════════


def test_tc018_verification_failure_retries_then_safe_fallback(
    run_agent_graph, isolated_stores, fake_llm
):
    """TC-018：编造数字的答案验证不通过 → 重推 1 次 → 仍失败 → 安全提示兜底。"""
    from src.schemas.constants import MAX_REASON_ATTEMPTS

    from tests.e2e.conftest import build_state

    thread = isolated_stores.conversation.create_thread(
        user_id="user_advisor", user_role=ROLE_ADVISOR, client_id=None, title="TC-018"
    )
    fake_llm.reason_content = "## 结论\n\n该基金年化收益率为3.9%，表现优异[来源1]。"

    state = run_agent_graph(build_state(thread_id=thread["thread_id"]))

    # 重试受 MAX_REASON_ATTEMPTS 限制，且确实发生了重推
    assert state[STATE_REASON_ATTEMPTS] == MAX_REASON_ATTEMPTS
    assert len([kind for kind, _ in fake_llm.calls if kind == "reason"]) == MAX_REASON_ATTEMPTS

    # 不可靠答案不得返回给用户：替换为安全提示 + 清空引用 + 低置信
    assert "未通过来源或数字验证" in state[STATE_FINAL_ANSWER]
    assert "3.9%" not in state[STATE_FINAL_ANSWER]
    assert state[STATE_CITATIONS] == []
    assert state[STATE_CONFIDENCE] == CONFIDENCE_LOW


# ══════════════════════════════════════════════════════════════════════
# TC-019/020/021/022 共享：QA API 客户端（真实路由 + 注入 Agent/存储）
# ══════════════════════════════════════════════════════════════════════


class _DisabledCache:
    """语义缓存替身：永不命中，store 静默成功。"""

    def lookup(self, query, role="", **kwargs):
        return None

    def store(self, query, answer, citations=None, confidence="", role="",
              compliance=None, verification=None, **kwargs):
        return True


@pytest.fixture()
def qa_api(monkeypatch, tmp_path):
    """QA API 上下文：holder["agent"] 注入 Agent 替身；holder["config"] 覆盖配置。"""
    from src.utils.audit import SQLiteAuditStore
    from src.utils.conversation import SQLiteConversationStore

    store = SQLiteConversationStore(tmp_path / "conversations.db")
    monkeypatch.setattr("src.api.main._get_conversation_store", lambda: store)
    monkeypatch.setattr(
        "src.api.main._get_cache_hit_audit_store",
        lambda: SQLiteAuditStore(tmp_path / "audit.db"),
    )
    monkeypatch.setattr("src.api.main.get_semantic_cache", lambda: _DisabledCache())

    holder: dict[str, Any] = {"agent": None}

    def _agent():
        assert holder["agent"] is not None, "测试未注入 agent 替身"
        return holder["agent"]

    monkeypatch.setattr("src.api.main._get_agent_app", _agent)

    app.dependency_overrides[authenticate_user] = lambda: AuthenticatedUser(
        "user_advisor", ROLE_ADVISOR, "wealth"
    )
    holder["client"] = TestClient(app)
    holder["conversation"] = store
    yield holder
    app.dependency_overrides.clear()


class _SleepyAgentApp:
    def invoke(self, state, config=None):
        import time

        time.sleep(0.5)
        return {**state, STATE_FINAL_ANSWER: "太慢", STATE_CITATIONS: [], STATE_CONFIDENCE: "low"}


class _BrokenProviderAgentApp:
    def invoke(self, state, config=None):
        import httpx
        from openai import APIConnectionError

        raise APIConnectionError(request=httpx.Request("POST", "http://llm.invalid/v1"))


# ══════════════════════════════════════════════════════════════════════
# TC-019 请求处理超时
# ══════════════════════════════════════════════════════════════════════


def test_tc019_qa_timeout_returns_504(qa_api, monkeypatch):
    """TC-019：Agent 超过 api_request_timeout_seconds → 504，不返回部分答案。"""
    qa_api["agent"] = _SleepyAgentApp()
    monkeypatch.setattr(
        "src.api.main.config", type("C", (), {"api_request_timeout_seconds": 0.05})
    )

    res = qa_api["client"].post(API_ROUTE_ASSISTANT_QA, json={"query": "货币基金风险"})

    assert res.status_code == 504
    assert "超时" in res.json()["detail"]


# ══════════════════════════════════════════════════════════════════════
# TC-020 LLM Provider 不可用
# ══════════════════════════════════════════════════════════════════════


def test_tc020_llm_provider_unavailable_returns_503(qa_api):
    """TC-020：Agent 抛 APIConnectionError → 503 + 排查指引，而非 500。"""
    qa_api["agent"] = _BrokenProviderAgentApp()

    res = qa_api["client"].post(API_ROUTE_ASSISTANT_QA, json={"query": "货币基金风险"})

    assert res.status_code == 503
    detail = res.json()["detail"]
    assert "LLM provider unavailable" in detail
    assert "OPENAI_API_BASE" in detail or "Ollama" in detail


# ══════════════════════════════════════════════════════════════════════
# TC-021 限流
# ══════════════════════════════════════════════════════════════════════


def test_tc021_rate_limit_blocks_qa_and_sse(qa_api, monkeypatch):
    """TC-021：限流触发时 QA 返回 429 + Retry-After；SSE 以 error 事件 429 返回。"""
    qa_api["agent"] = _StreamingAgentApp()
    monkeypatch.setattr("src.api.main.check_rate_limit", lambda key: (False, 0))

    res = qa_api["client"].post(API_ROUTE_ASSISTANT_QA, json={"query": "货币基金风险"})
    assert res.status_code == 429
    assert res.headers.get("retry-after") == "60"

    with qa_api["client"].stream(
        "POST", API_ROUTE_ASSISTANT_QA_STREAM, json={"query": "货币基金风险"}
    ) as sse:
        assert sse.status_code == 429
        events = _parse_sse(list(sse.iter_lines()))
    assert events, "限流的 SSE 也必须返回事件"
    name, data = events[0]
    assert name == "error"
    assert data["type"] == "error"
    assert "频繁" in data["detail"]


# ══════════════════════════════════════════════════════════════════════
# TC-022 会话异常
# ══════════════════════════════════════════════════════════════════════


def test_tc022_thread_not_found_and_cross_user_denied(qa_api):
    """TC-022：不存在的 thread_id → 404；他人 thread → 404，不泄露内容。"""
    qa_api["agent"] = _StreamingAgentApp()
    client = qa_api["client"]

    res = client.post(
        API_ROUTE_ASSISTANT_QA, json={"query": "货币基金风险", "thread_id": "no-such-thread"}
    )
    assert res.status_code == 404
    assert res.json()["detail"] == "会话不存在或不可访问"

    thread = qa_api["conversation"].create_thread(
        user_id="user_other", user_role=ROLE_ADVISOR, client_id=None, title="别人的会话"
    )
    res = client.post(
        API_ROUTE_ASSISTANT_QA,
        json={"query": "货币基金风险", "thread_id": thread["thread_id"]},
    )
    assert res.status_code == 404
    assert "风险" not in res.text, "跨用户访问不得泄露他人会话内容"


def test_tc022_context_mismatch_returns_409(qa_api):
    """TC-022：同一 thread 的角色/客户上下文变化 → 409 显式冲突。"""
    qa_api["agent"] = _StreamingAgentApp()
    thread = qa_api["conversation"].create_thread(
        user_id="user_advisor", user_role=ROLE_ADVISOR, client_id="client-A", title="TC-022"
    )

    res = qa_api["client"].post(
        API_ROUTE_ASSISTANT_QA,
        json={"query": "货币基金风险", "thread_id": thread["thread_id"], "client_id": "client-B"},
    )
    assert res.status_code == 409
    assert "上下文" in res.json()["detail"] or "变化" in res.json()["detail"]


# ══════════════════════════════════════════════════════════════════════
# TC-023 工具白名单与超时熔断
# ══════════════════════════════════════════════════════════════════════


@pytest.fixture()
def _reset_circuit_breaker():
    from src.agents import nodes as agent_nodes

    agent_nodes._tool_circuit_breaker.clear()
    yield
    agent_nodes._tool_circuit_breaker.clear()


def _tool_request(state: dict, name: str) -> Any:
    from langgraph.prebuilt.tool_node import ToolCallRequest

    return ToolCallRequest(
        state=state,
        tool_call={"name": name, "args": {}, "id": f"call_{name}"},
        tool=None,
        runtime=cast(Any, None),
    )


def test_tc023_unauthorized_tool_is_rejected_without_execution(_reset_circuit_breaker):
    """TC-023：advisor 调用 faq_search（角色无 FAQ 源）→ 拒绝且工具不执行。"""
    from langchain_core.messages import ToolMessage

    from src.agents.nodes import authorize_reason_tool_call
    from src.schemas.constants import (
        ROLE_ADVISOR,
        SOURCE_PRODUCT,
        STATE_RETRIEVAL_PLAN,
        STATE_USER_ROLE,
    )

    state = {
        STATE_USER_ROLE: ROLE_ADVISOR,
        STATE_RETRIEVAL_PLAN: [{"source": SOURCE_PRODUCT, "query": "q", "top_k": 3}],
        STATE_RETRIEVAL_RESULTS: [],
    }
    executed: list[Any] = []

    def _record(request: Any) -> ToolMessage:
        executed.append(request)
        return ToolMessage(content="", name="faq_search", tool_call_id="")

    result = authorize_reason_tool_call(_tool_request(state, "faq_search"), _record)

    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    assert "无权调用" in result.content
    assert executed == [], "越权工具不得执行"


def test_tc023_tool_timeout_trips_circuit_breaker(
    _reset_circuit_breaker, monkeypatch
):
    """TC-023：工具执行超时 → 错误 ToolMessage + 熔断，冷却期内直接拒绝。"""
    import time
    from langchain_core.messages import ToolMessage

    from src.agents import nodes as agent_nodes
    from src.agents.nodes import authorize_reason_tool_call
    from src.schemas.constants import (
        ROLE_ADVISOR,
        SOURCE_PRODUCT,
        STATE_RETRIEVAL_PLAN,
        STATE_USER_ROLE,
    )

    monkeypatch.setattr(agent_nodes, "TOOL_TIMEOUT_SECONDS", 0.05)
    state = {
        STATE_USER_ROLE: ROLE_ADVISOR,
        STATE_RETRIEVAL_PLAN: [{"source": SOURCE_PRODUCT, "query": "q", "top_k": 3}],
        STATE_RETRIEVAL_RESULTS: [],
    }

    def slow_execute(request):
        time.sleep(0.3)
        return ToolMessage(content="ok", name="calculator", tool_call_id="x")

    first = authorize_reason_tool_call(_tool_request(state, "calculator"), slow_execute)
    assert isinstance(first, ToolMessage) and first.status == "error"
    assert "超时" in first.content

    # 熔断期内第二次调用直接拒绝，不再执行工具
    calls: list[Any] = []

    def _record(request: Any) -> ToolMessage:
        calls.append(request)
        return ToolMessage(content="", name="calculator", tool_call_id="")

    second = authorize_reason_tool_call(_tool_request(state, "calculator"), _record)
    assert isinstance(second, ToolMessage) and second.status == "error"
    assert "熔断" in second.content
    assert calls == []
