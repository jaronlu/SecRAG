"""Langfuse 验收测试（todo/TODO.md 测试与验收）——graph 级传播与 payload 卫生。

与 tests/e2e/test_e2e_langfuse_wiring.py 互补，覆盖验收点：
- callbacks 从 RunnableConfig 传播到 LangGraph 节点内 LLM 调用**和工具调用**
  （用假 handler 捕获 on_llm_start/on_tool_start；替身继承 BaseChatModel，
  走真实 ensure_config 传播路径，避免普通类替身 config 恒 None 的假阴性）；
- payload 卫生：金丝雀标记的原始问题/完整回答/chunk 文本/工具参数与输出/
  SQL/客户 ID/持仓在全链路运行后不出现在任何上送 payload 中，且 metadata
  键集合不超过白名单；
- Langfuse 关闭/未采样时 graph 业务结果与基线运行完全一致（no-op 断言）。

替身遵循 tests/e2e/conftest.py 模式：LLM/检索一律替身，存储落 tmp_path，
不连真实 Langfuse 服务。
"""

from __future__ import annotations

import json
import uuid
from typing import Any, ClassVar

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables.config import RunnableConfig, var_child_runnable_config

from src.agents import nodes as agent_nodes
from src.agents.graph import build_agent_with_checkpoint
from src.config import LangfuseConfig
from src.schemas.constants import ROLE_ADVISOR
from src.utils.langfuse_adapter import (
    ROOT_TRACE_NAME,
    LangfuseAdapter,
    LangfuseTraceMetadata,
    reset_current_trace,
    set_current_trace,
)
from src.utils.metrics import MetricsRegistry
from tests.e2e.conftest import (
    FUND_REPORT_CHUNK,
    PLANNER_MARKER,
    QUERY_UNDERSTAND_MARKER,
    RR_CONTENT,
    build_state,
    make_fund_result,
)

# 金丝雀标记（全字母，避免数字串触发业务 PII/数字验证路径）：
# 只要出现在任何上送 payload 中即为隐私违规
CANARY_QUESTION = "CANARY-QUESTION-MARKER"
CANARY_ANSWER = "CANARY-ANSWER-MARKER"
CANARY_CHUNK = "CANARY-CHUNK-MARKER"
# 不在本链路中出现的数据类型：断言其不因任何路径混入 payload
CANARY_SQL = "CANARY-SQL-STATEMENT"
CANARY_CUSTOMER_ID = "CANARY-CUST-001"
CANARY_POSITIONS = "CANARY-POSITIONS-DETAIL"

BASE_QUERY = "XX货币市场基金的风险等级是什么？"
QUERY_WITH_CANARY = f"{BASE_QUERY}（金丝雀{CANARY_QUESTION}）"
REASON_CONTENT = "## 结论\n\n本基金风险等级为R1（低风险），适合保守型投资者[来源1]。"
FINAL_CONTENT_WITH_CANARY = f"{REASON_CONTENT}金丝雀{CANARY_ANSWER}"


class ToolCallingChatModel(BaseChatModel):
    """先发起 calculator 工具调用、再给最终回答的 BaseChatModel 替身。

    记录 ensure_config 注入的 RunnableConfig（LLM 传播断言）与全部 prompt
    （金丝雀确实进入过系统内部，payload 断言才不是空证）。
    """

    seen_configs: ClassVar[list[RunnableConfig | None]] = []
    seen_prompts: ClassVar[list[str]] = []
    seen_outputs: ClassVar[list[str]] = []
    # ToolMessage 分支返回的最终回答，由用例注入（如金丝雀回答）
    final_content: ClassVar[str] = REASON_CONTENT

    def _generate(
        self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any
    ) -> ChatResult:
        ToolCallingChatModel.seen_configs.append(var_child_runnable_config.get())
        prompt = (
            messages[-1].content if isinstance(messages[-1].content, str) else str(messages[-1].content)
        )
        # 记录全部消息（含 system 中的检索证据），供金丝雀前提断言使用
        ToolCallingChatModel.seen_prompts.append(
            "\n".join(
                m.content if isinstance(m.content, str) else str(m.content) for m in messages
            )
        )
        if QUERY_UNDERSTAND_MARKER in prompt:
            content = json.dumps({
                "intent": "产品咨询",
                "query_type": "product_inquiry",
                "entities": {
                    "product_name": "XX货币市场基金",
                    "product_type": "fund",
                    "stock_code": "",
                    "regulation_name": "",
                    "client_segment": "",
                    "time_range": {"start": "", "end": ""},
                },
                "rewritten_query": "XX货币市场基金 风险等级",
                "ambiguity": [],
            }, ensure_ascii=False)
        elif PLANNER_MARKER in prompt:
            content = json.dumps([
                {"source": "product_search", "query": "XX货币市场基金 风险等级", "top_k": 3}
            ], ensure_ascii=False)
        elif isinstance(messages[-1], ToolMessage):
            # 工具结果已回填：给出最终回答
            content = ToolCallingChatModel.final_content
        else:
            # 首次推理：发起一次 calculator 工具调用
            content = ""
            return ChatResult(generations=[ChatGeneration(message=AIMessage(
                content=content,
                tool_calls=[{
                    "name": "calculator",
                    "args": {"expression": "1+1"},
                    "id": "call_accept_1",
                    "type": "tool_call",
                }],
            ))])
        ToolCallingChatModel.seen_outputs.append(content)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

    @property
    def _llm_type(self) -> str:
        return "tool-calling-chat"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ToolCallingChatModel":
        return self


class RecordingCallbackHandler(BaseCallbackHandler):
    """假 handler：捕获随 RunnableConfig 传播到的 LLM 与工具回调事件。

    工具名在 ``serialized["name"]``（langchain-core 回调不传 run_name）。
    """

    def __init__(self) -> None:
        self.llm_starts = 0
        self.tool_starts: list[str] = []
        self.tool_ends = 0

    def on_llm_start(self, *args: Any, **kwargs: Any) -> None:
        self.llm_starts += 1

    def on_tool_start(self, serialized: Any = None, input_str: Any = None, **kwargs: Any) -> None:
        name = (serialized or {}).get("name") or kwargs.get("run_name") or "unknown"
        self.tool_starts.append(str(name))

    def on_tool_end(self, *args: Any, **kwargs: Any) -> None:
        self.tool_ends += 1


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


def _wired_adapter(client: FakeLangfuseClient, **cfg_overrides: Any) -> LangfuseAdapter:
    defaults: dict[str, Any] = {
        "enabled": True,
        "host": "https://cloud.langfuse.com",
        "public_key": f"pk-lf-{uuid.uuid4().hex}",
        "secret_key": f"sk-lf-{uuid.uuid4().hex}",
        "sample_rate": 1.0,
        "capture_content": False,
    }
    defaults.update(cfg_overrides)
    return LangfuseAdapter(
        cfg=LangfuseConfig(**defaults),
        app_env="development",
        client=client,
        metrics=MetricsRegistry(),
    )


@pytest.fixture()
def tool_calling_llm(monkeypatch: pytest.MonkeyPatch):
    ToolCallingChatModel.seen_configs = []
    ToolCallingChatModel.seen_prompts = []
    ToolCallingChatModel.seen_outputs = []
    ToolCallingChatModel.final_content = REASON_CONTENT
    fake = ToolCallingChatModel()
    monkeypatch.setattr(agent_nodes, "llm", fake)
    agent_nodes._get_bound_reason_model.cache_clear()
    yield fake
    agent_nodes._get_bound_reason_model.cache_clear()


def _run_graph_with_callbacks(
    isolated_stores: Any,
    query: str,
    callbacks: list[BaseCallbackHandler] | None,
) -> tuple[dict[str, Any], Any]:
    """真实链路顺序：先 ensure 会话，再以 API 同款 config 执行图。"""
    thread = isolated_stores.conversation.create_thread(
        user_id="user_advisor", user_role=ROLE_ADVISOR, client_id=None, title="lf-accept"
    )
    state = build_state(thread_id=thread["thread_id"], query=query)
    config = RunnableConfig(configurable={"thread_id": state["thread_id"]})
    if callbacks:
        config["callbacks"] = callbacks
    graph = build_agent_with_checkpoint()
    return graph.invoke(state, config), state


def _serialized_payloads(client: FakeLangfuseClient) -> str:
    chunks = [
        json.dumps(kwargs, ensure_ascii=False, default=str) for _, kwargs in client.started
    ]
    chunks.extend(
        json.dumps(update, ensure_ascii=False, default=str)
        for span, _ in client.started
        for update in span.updates
    )
    return "\n".join(chunks)


# ─────────── 验收点 2：callback 传播到 LangGraph、LLM 与工具调用 ───────────


def test_callbacks_reach_llm_and_tool_calls_and_node_spans_attach_to_root(
    tool_calling_llm, isolated_stores, fake_retriever_factory
):
    client = FakeLangfuseClient()
    adapter = _wired_adapter(client)
    trace = adapter.start_request_trace("req-accept", "thread-accept")
    marker = RecordingCallbackHandler()
    token = set_current_trace(trace)
    try:
        result, _ = _run_graph_with_callbacks(isolated_stores, BASE_QUERY, [marker])
    finally:
        reset_current_trace(token)
        trace.finish(status="ok")

    assert result["final_answer"]
    # 1) LLM 传播：每次模型调用都拿到带 marker 的 config（非假阴性替身）
    assert tool_calling_llm.seen_configs
    assert all(cfg is not None for cfg in tool_calling_llm.seen_configs)
    for cfg in tool_calling_llm.seen_configs:
        assert any(h is marker for h in cfg["callbacks"].handlers)
    # 2) 工具传播：calculator 工具执行触发 handler 的 on_tool_start/on_tool_end
    assert marker.llm_starts >= 3  # query_understand + planner + reason×2
    assert marker.tool_starts == ["calculator"]
    assert marker.tool_ends == 1

    # 3) 节点 span 与工具 span 都挂到根 trace 之下
    root_span, root_kwargs = client.started[0]
    assert root_kwargs["name"] == ROOT_TRACE_NAME
    node_names = {kwargs["name"] for _, kwargs in client.started[1:]}
    assert {
        "query_understand",
        "planner",
        "retrieve",
        "call_reason_model",
        "verify",
        "compose",
        "calculator",
    } <= node_names
    for _, kwargs in client.started[1:]:
        assert kwargs["trace_context"]["trace_id"] == root_span.trace_id
        assert kwargs["trace_context"]["parent_span_id"] == root_span.id


# ─────────── 验收点 3：金丝雀 payload 卫生（全链路） ───────────


def test_canary_data_never_reaches_langfuse_payload(
    tool_calling_llm, isolated_stores, fake_retriever_factory
):
    # 注入金丝雀：问题、chunk 文本、最终回答全部带标记
    ToolCallingChatModel.final_content = FINAL_CONTENT_WITH_CANARY
    canary_result = make_fund_result()
    canary_result[RR_CONTENT] = f"{FUND_REPORT_CHUNK}金丝雀{CANARY_CHUNK}"
    fake_retriever_factory["results"] = [canary_result]

    client = FakeLangfuseClient()
    adapter = _wired_adapter(client)
    trace = adapter.start_request_trace("req-canary", "thread-canary")
    marker = RecordingCallbackHandler()
    token = set_current_trace(trace)
    try:
        result, _ = _run_graph_with_callbacks(isolated_stores, QUERY_WITH_CANARY, [marker])
    finally:
        reset_current_trace(token)
        trace.finish(status="ok")

    # 前提：金丝雀确实流经系统（否则卫生断言是空证）——
    # 问题/chunk 进入过模型 prompt，带标记回答是模型产出（验证通过时即用户可见回答）
    assert any(CANARY_QUESTION in prompt for prompt in tool_calling_llm.seen_prompts)
    assert any(CANARY_CHUNK in prompt for prompt in tool_calling_llm.seen_prompts)
    assert any(CANARY_ANSWER in out for out in tool_calling_llm.seen_outputs)

    # 任何上送 payload（span 参数 + 后续 update）中都不得出现金丝雀：
    # 原始问题/完整回答/chunk 文本/工具参数/工具调用 id/SQL/客户 ID/持仓
    payloads = _serialized_payloads(client)
    for forbidden in (
        CANARY_QUESTION,
        CANARY_ANSWER,
        CANARY_CHUNK,
        CANARY_SQL,
        CANARY_CUSTOMER_ID,
        CANARY_POSITIONS,
        "1+1",  # 工具原始参数
        "expression",  # 工具参数键名
        "call_accept_1",  # 工具调用 id
    ):
        assert forbidden not in payloads, f"隐私违规：{forbidden} 进入 Langfuse payload"

    # 工具 span 存在但只带白名单 metadata（工具被观测、内容不外泄）；
    # 验证失败重试会再次调用工具，这里只断言每次都零内容
    tool_spans = [kwargs for _, kwargs in client.started if kwargs["name"] == "calculator"]
    assert tool_spans
    assert all(kwargs["metadata"] == {"node_name": "calculator"} for kwargs in tool_spans)

    # metadata 键集合不得超过白名单
    for _, kwargs in client.started:
        assert set(kwargs.get("metadata") or {}) <= set(LangfuseTraceMetadata.__annotations__)
        for update in _span_of(client, kwargs).updates:
            assert set(update.get("metadata") or {}) <= set(
                LangfuseTraceMetadata.__annotations__
            )


def _span_of(client: FakeLangfuseClient, kwargs: dict[str, Any]) -> FakeLangfuseSpan:
    return next(span for span, kw in client.started if kw is kwargs)


# ─────────── 验收点 1：关闭/未采样时业务结果与基线一致 ───────────


def _strip_volatile(value: Any) -> Any:
    """剔除快照中的易变字段（审计 timestamp），保证跨运行可比。"""
    if isinstance(value, dict):
        return {k: _strip_volatile(v) for k, v in value.items() if k != "timestamp"}
    if isinstance(value, list):
        return [_strip_volatile(v) for v in value]
    return value


def _business_snapshot(result: dict[str, Any]) -> tuple[Any, ...]:
    return (
        result.get("final_answer"),
        json.dumps(_strip_volatile(result.get("citations", [])), ensure_ascii=False, sort_keys=True),
        result.get("confidence"),
        result.get("compliance"),
    )


def test_disabled_and_sampled_out_runs_match_baseline_business_result(
    fake_llm, isolated_stores, fake_retriever_factory
):
    # 基线：完全无 Langfuse
    baseline, _ = _run_graph_with_callbacks(isolated_stores, BASE_QUERY, None)

    # 关闭：no-op trace 挂到 contextvar，业务零感知
    disabled_client = FakeLangfuseClient()
    disabled = _wired_adapter(disabled_client, enabled=False, host="", public_key="", secret_key="")
    trace_off = disabled.start_request_trace("req-off", "thread-off")
    token = set_current_trace(trace_off)
    try:
        result_off, _ = _run_graph_with_callbacks(isolated_stores, BASE_QUERY, None)
    finally:
        reset_current_trace(token)
        trace_off.finish(status="ok")
    assert disabled.enabled is False
    assert disabled_client.started == []

    # 启用但采样率 0：不产生任何 span，业务仍正常
    sampled_client = FakeLangfuseClient()
    sampled = _wired_adapter(sampled_client, sample_rate=0.0)
    trace_out = sampled.start_request_trace("req-out", "thread-out")
    token = set_current_trace(trace_out)
    try:
        result_out, _ = _run_graph_with_callbacks(isolated_stores, BASE_QUERY, None)
    finally:
        reset_current_trace(token)
        trace_out.finish(status="ok")
    assert trace_out.is_sampled is False
    assert sampled_client.started == []
    assert sampled._metrics.langfuse_dropped_total.get(labels={"reason": "sampled_out"}) == 1.0

    # 三次运行业务结果逐项一致
    assert _business_snapshot(result_off) == _business_snapshot(baseline)
    assert _business_snapshot(result_out) == _business_snapshot(baseline)
    assert baseline["final_answer"]
