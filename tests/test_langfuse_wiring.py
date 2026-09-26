"""Langfuse 接线测试：API 根 trace/callbacks 注入与节点 span。

覆盖 todo/TODO.md 验收项：
- API 入口创建根 trace，request_id 与 SQLite 审计共用，metadata 无用户身份；
- callbacks 经 RunnableConfig 注入 LangGraph（未启用时零感知）；
- _traced_node / reason 子图节点按白名单 metadata 建 span，查询/回答/
  引用原文不进 payload；
- 缓存命中路径带 cache_hit 标记，错误路径 trace 收尾为 error。

替身沿用 tests/test_langfuse_adapter.py 的 fake client 模式；不连真实服务。
"""

from __future__ import annotations

import json
import uuid
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import HTTPException
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.agents import nodes as agent_nodes
from src.agents.graph import _node_span_metadata, _traced_node
from src.api.auth import AuthenticatedUser, build_assistant_initial_state
from src.api.main import assistant_qa
from src.config import LangfuseConfig, config
from src.schemas.constants import (
    AUDIT_REQUEST_ID,
    ROLE_TECHNICAL,
    STATE_AUDIT_TRAIL,
    STATE_INTERMEDIATE_STEPS,
    STATE_MESSAGES,
    STATE_REASON_ATTEMPTS,
    STATE_RETRIEVAL_ATTEMPTS,
    STATE_RETRIEVAL_FILTERED_CHUNKS,
    STATE_RETRIEVAL_PLAN,
    STATE_RETRIEVAL_RESULTS,
    STATE_REQUEST_DEADLINE,
    STATE_VERIFICATION,
)
from src.schemas.request_response import AssistantQARequest
from src.utils.langfuse_adapter import (
    ROOT_TRACE_NAME,
    LangfuseAdapter,
    reset_current_trace,
    set_current_trace,
    start_node_span,
)
from src.utils.metrics import MetricsRegistry


# ───────────────────────── 替身（与 adapter 测试同型） ─────────────────────────


class FakeLangfuseSpan:
    def __init__(self) -> None:
        self.trace_id = uuid.uuid4().hex
        self.id = uuid.uuid4().hex[:16]
        self.updates: list[dict[str, Any]] = []
        self.ended = False

    def update(self, **kwargs: Any) -> None:
        self.updates.append(kwargs)

    def end(self, **kwargs: Any) -> None:
        self.ended = True


class FakeLangfuseClient:
    def __init__(self) -> None:
        self.started: list[tuple[FakeLangfuseSpan, dict[str, Any]]] = []

    def start_observation(self, **kwargs: Any) -> FakeLangfuseSpan:
        span = FakeLangfuseSpan()
        self.started.append((span, kwargs))
        return span

    def flush(self) -> None:
        pass


class FakeCallbackHandler:
    """替身：记录绑定到根 trace 的 trace_context。"""

    def __init__(self, *, trace_context: dict[str, str] | None = None) -> None:
        self.trace_context = trace_context


class _RecordingAgentApp:
    """记录收到的 state 与 RunnableConfig 的 agent 替身。"""

    def __init__(self) -> None:
        self.config: dict[str, Any] | None = None
        self.state: dict[str, Any] | None = None

    def invoke(self, state: dict[str, Any], config: Any = None) -> dict[str, Any]:
        self.config = config
        self.state = state
        return {
            **state,
            "final_answer": "ok",
            "citations": [],
            "confidence": "high",
            "compliance": {"passed": True},
        }


class _FailingAgentApp:
    def invoke(self, state: dict[str, Any], config: Any = None) -> dict[str, Any]:
        raise ValueError("bad state")


class _BindToolsFakeModel:
    """call_reason_model 用的模型替身：带 usage_metadata 的固定回答。"""

    def bind_tools(self, tools: Any, **kwargs: Any) -> "_BindToolsFakeModel":
        return self

    def invoke(self, messages: Any, config: Any = None) -> AIMessage:
        return AIMessage(
            content="## 结论\n\nok",
            usage_metadata={"input_tokens": 11, "output_tokens": 7, "total_tokens": 18},
        )


def _enabled_cfg(**overrides: Any) -> LangfuseConfig:
    defaults: dict[str, Any] = {
        "enabled": True,
        "host": "https://cloud.langfuse.com",
        "public_key": f"pk-lf-{uuid.uuid4().hex}",
        "secret_key": f"sk-lf-{uuid.uuid4().hex}",
        "sample_rate": 1.0,
        "capture_content": False,
    }
    defaults.update(overrides)
    return LangfuseConfig(**defaults)


def _wire_adapter(client: FakeLangfuseClient | None = None) -> LangfuseAdapter:
    return LangfuseAdapter(
        cfg=_enabled_cfg(),
        app_env="development",
        client=client if client is not None else FakeLangfuseClient(),
        metrics=MetricsRegistry(),
    )


@pytest.fixture(autouse=True)
def _temp_conversation_store(monkeypatch, tmp_path):
    from src.utils.conversation import SQLiteConversationStore

    store = SQLiteConversationStore(tmp_path / "conversations.db")
    monkeypatch.setattr("src.api.main._get_conversation_store", lambda: store)


@pytest.fixture()
def langfuse_env(monkeypatch):
    """启用 adapter（注入 fake client）+ callback handler 替身。"""
    client = FakeLangfuseClient()
    adapter = _wire_adapter(client)
    monkeypatch.setattr("src.api.main.get_langfuse", lambda: adapter)
    monkeypatch.setattr(
        "langfuse.langchain.CallbackHandler", FakeCallbackHandler, raising=True
    )
    return SimpleNamespace(adapter=adapter, client=client)


def _qa_request(query: str = "货币基金的风险等级是什么") -> AssistantQARequest:
    return AssistantQARequest(query=query)


# ───────────────────────── API 入口根 trace ─────────────────────────


@pytest.mark.asyncio
async def test_qa_creates_root_trace_and_injects_bound_callback(langfuse_env, monkeypatch):
    query = "货币基金的风险等级是什么"
    agent = _RecordingAgentApp()
    monkeypatch.setattr("src.api.main._get_agent_app", lambda: agent)

    await assistant_qa(
        _qa_request(query),
        AuthenticatedUser("user_lf_stream", ROLE_TECHNICAL, "tech"),
    )

    root_span, root_kwargs = langfuse_env.client.started[0]
    assert root_kwargs["name"] == ROOT_TRACE_NAME
    metadata = root_kwargs["metadata"]
    # request_id/thread_id 关联，且不携带用户身份信息
    assert metadata["request_id"]
    assert metadata["thread_id"]
    assert "user_id" not in metadata
    assert "department" not in metadata
    # 原始问题不进 payload
    assert query not in json.dumps(metadata, ensure_ascii=False)

    # 根 trace 与 SQLite 审计（state audit trail）共用同一 request_id
    assert agent.state is not None
    assert agent.state[STATE_AUDIT_TRAIL][AUDIT_REQUEST_ID] == metadata["request_id"]

    # callbacks 注入 RunnableConfig，并绑定到根 trace
    callbacks = agent.config["callbacks"]
    assert len(callbacks) == 1
    assert isinstance(callbacks[0], FakeCallbackHandler)
    assert callbacks[0].trace_context == {
        "trace_id": root_span.trace_id,
        "parent_span_id": root_span.id,
    }

    # 请求收尾：根 span 结束且状态 ok
    assert root_span.ended is True
    assert root_span.updates[-1]["metadata"]["status"] == "ok"


@pytest.mark.asyncio
async def test_qa_cache_hit_marks_cache_hit_on_trace(langfuse_env, monkeypatch):
    class _StubCache:
        def lookup(self, query: str, role: str = "") -> dict[str, Any]:
            return {
                "answer": "货币基金风险等级为低。",
                "citations": [],
                "confidence": "high",
                "similarity": 0.95,
                "compliance": {"passed": True, "risk_disclosure": ""},
                "verification": {"passed": True, "confidence": "high"},
            }

        def store(self, **kwargs: Any) -> bool:
            return True

    monkeypatch.setattr("src.api.main.get_semantic_cache", lambda: _StubCache())
    agent = _RecordingAgentApp()
    monkeypatch.setattr("src.api.main._get_agent_app", lambda: agent)

    await assistant_qa(
        _qa_request("货币基金风险等级"),
        AuthenticatedUser("user_lf_cache", ROLE_TECHNICAL, "tech"),
    )

    # 命中路径不执行图，但 trace 照常创建并带 cache_hit 标记
    assert agent.config is None
    root_span, _ = langfuse_env.client.started[0]
    assert root_span.ended is True
    updated = [u.get("metadata") or {} for u in root_span.updates]
    assert any(m.get("cache_hit") is True for m in updated)


@pytest.mark.asyncio
async def test_qa_error_finishes_trace_as_error(langfuse_env, monkeypatch):
    monkeypatch.setattr("src.api.main._get_agent_app", lambda: _FailingAgentApp())

    with pytest.raises(HTTPException) as exc:
        await assistant_qa(
            _qa_request("查询"),
            AuthenticatedUser("user_lf_error", ROLE_TECHNICAL, "tech"),
        )

    assert exc.value.status_code == 500
    root_span, _ = langfuse_env.client.started[0]
    assert root_span.ended is True
    final_metadata = root_span.updates[-1]["metadata"]
    assert final_metadata["status"] == "error"
    assert final_metadata["error_type"] == "ValueError"


@pytest.mark.asyncio
async def test_qa_without_langfuse_has_no_callbacks(monkeypatch):
    adapter = LangfuseAdapter(
        cfg=LangfuseConfig(enabled=False, host="", public_key="", secret_key=""),
        app_env="development",
        client=FakeLangfuseClient(),
        metrics=MetricsRegistry(),
    )
    monkeypatch.setattr("src.api.main.get_langfuse", lambda: adapter)
    agent = _RecordingAgentApp()
    monkeypatch.setattr("src.api.main._get_agent_app", lambda: agent)

    await assistant_qa(
        _qa_request("查询"),
        AuthenticatedUser("user_lf_off", ROLE_TECHNICAL, "tech"),
    )

    # Langfuse 关闭：RunnableConfig 无 callbacks 键，问答行为不变
    assert "callbacks" not in agent.config
    assert agent.config["recursion_limit"] > 0
    assert adapter.enabled is False


# ───────────────────────── 节点 span（_traced_node） ─────────────────────────


def _trace_with_client():
    client = FakeLangfuseClient()
    adapter = _wire_adapter(client)
    trace = adapter.start_request_trace("req-node", "thread-node")
    return trace, client


def test_traced_node_creates_span_with_whitelisted_metadata():
    trace, client = _trace_with_client()
    token = set_current_trace(trace)
    try:
        node = _traced_node("verify", lambda state: {STATE_VERIFICATION: {"passed": True}})
        result = node({})
    finally:
        reset_current_trace(token)

    assert result[STATE_VERIFICATION]["passed"] is True
    assert len(client.started) == 2  # 根 trace + verify span
    span, kwargs = client.started[1]
    assert kwargs["name"] == "verify"
    assert kwargs["metadata"] == {"node_name": "verify"}
    assert span.ended is True
    final_metadata = span.updates[-1]["metadata"]
    assert final_metadata["node_name"] == "verify"
    assert final_metadata["verification_status"] == "passed"
    assert final_metadata["duration_ms"] >= 0
    assert final_metadata["status"] == "ok"


def test_traced_node_without_trace_is_noop():
    client = FakeLangfuseClient()
    _wire_adapter(client)  # adapter 存在但 contextvar 无活跃 trace

    node = _traced_node("retrieve", lambda state: {})
    result = node({})

    assert result[STATE_INTERMEDIATE_STEPS][0]["step"] == "retrieve"
    assert client.started == []


def test_traced_node_error_span_and_reraise():
    trace, client = _trace_with_client()
    token = set_current_trace(trace)

    def _boom(state: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("node failed")

    try:
        node = _traced_node("compose", _boom)
        with pytest.raises(RuntimeError):
            node({})
    finally:
        reset_current_trace(token)

    span, _ = client.started[1]
    final_metadata = span.updates[-1]["metadata"]
    assert final_metadata["status"] == "error"
    assert final_metadata["error_type"] == "RuntimeError"
    assert span.ended is True


def test_retrieve_and_planner_span_metadata():
    state = {
        STATE_RETRIEVAL_RESULTS: [{"a": 1}, {"b": 2}],
        STATE_RETRIEVAL_PLAN: [],
    }
    result = {
        STATE_RETRIEVAL_RESULTS: [{"a": 1}, {"b": 2}, {"c": 3}, {"d": 4}, {"e": 5}],
        STATE_RETRIEVAL_ATTEMPTS: 2,
    }
    retrieve_meta = _node_span_metadata("retrieve", state, result)
    assert retrieve_meta["retrieval_count"] == 3
    assert retrieve_meta["retry_count"] == 2
    assert retrieve_meta["node_name"] == "retrieve"

    planner_meta = _node_span_metadata("planner", {}, {})
    assert planner_meta["model_name"] == config.llm.model

    grade_meta = _node_span_metadata("grade_and_filter", {}, {STATE_RETRIEVAL_FILTERED_CHUNKS: 4})
    assert grade_meta["retrieval_count"] == 4


# ───────────────────────── reason 子图 span ─────────────────────────


@pytest.fixture()
def bound_fake_llm(monkeypatch):
    monkeypatch.setattr(agent_nodes, "llm", _BindToolsFakeModel())
    agent_nodes._get_bound_reason_model.cache_clear()
    yield
    agent_nodes._get_bound_reason_model.cache_clear()


def _reason_state(**overrides: Any) -> dict[str, Any]:
    state = build_assistant_initial_state(
        AssistantQARequest(query="问题"),
        AuthenticatedUser("user_lf_node", ROLE_TECHNICAL, "tech"),
    )
    state[STATE_MESSAGES] = [HumanMessage(content="问题")]
    state[STATE_REASON_ATTEMPTS] = 1
    state.pop(STATE_REQUEST_DEADLINE, None)
    state.update(overrides)
    return state


def test_call_reason_model_span_carries_model_and_usage(bound_fake_llm):
    trace, client = _trace_with_client()
    token = set_current_trace(trace)
    try:
        result = agent_nodes.call_reason_model(_reason_state())
    finally:
        reset_current_trace(token)

    assert isinstance(result[STATE_MESSAGES][-1], AIMessage)
    span, kwargs = client.started[1]
    assert kwargs["name"] == "call_reason_model"
    assert kwargs["metadata"]["model_name"] == config.llm.model
    assert kwargs["metadata"]["retry_count"] == 1
    final_metadata = span.updates[-1]["metadata"]
    assert final_metadata["prompt_tokens"] == 11
    assert final_metadata["completion_tokens"] == 7
    assert final_metadata["total_tokens"] == 18
    assert final_metadata["duration_ms"] >= 0
    # 提示词与回答原文不进 payload
    assert "问题" not in json.dumps(kwargs["metadata"], ensure_ascii=False)


def test_tool_span_created_after_authorization():
    from src.agents.tools import calculator

    tool_name = calculator.name
    trace, client = _trace_with_client()
    token = set_current_trace(trace)
    try:
        request = SimpleNamespace(
            state=_reason_state(),
            tool_call={"name": tool_name, "args": {"expression": "1+1"}, "id": "call-1"},
        )

        def _execute(req: Any) -> ToolMessage:
            return ToolMessage(content="2", name=tool_name, tool_call_id="call-1", status="success")

        result = agent_nodes.authorize_reason_tool_call(request, _execute)
    finally:
        reset_current_trace(token)

    assert result.content == "2"
    span, kwargs = client.started[1]
    assert kwargs["name"] == tool_name
    assert kwargs["metadata"] == {"node_name": tool_name}
    # 工具原始参数不进 payload
    assert "expression" not in json.dumps(kwargs["metadata"], ensure_ascii=False)
    final_metadata = span.updates[-1]["metadata"]
    assert final_metadata["status"] == "ok"
    assert final_metadata["duration_ms"] >= 0


def test_unauthorized_tool_creates_no_span():
    from src.agents.tools import calculator

    trace, client = _trace_with_client()
    token = set_current_trace(trace)
    try:
        request = SimpleNamespace(
            state=_reason_state(),
            tool_call={"name": f"not_{calculator.name}", "args": {}, "id": "call-2"},
        )
        result = agent_nodes.authorize_reason_tool_call(
            request, lambda req: pytest.fail("未授权工具不应执行")
        )
    finally:
        reset_current_trace(token)

    assert result.status == "error"
    # 权限校验失败的调用不产生任何工具 span（client.started[0] 是根 trace 自身）
    assert len(client.started) == 1


def test_start_node_span_without_adapter_still_noop():
    # 无任何 adapter/trace 时（Langfuse 完全未接线），节点 span 帮助函数零开销且不抛
    handle = start_node_span("verify", metadata={"node_name": "verify"})
    handle.finish(status="ok", metadata={"duration_ms": 1.0})
