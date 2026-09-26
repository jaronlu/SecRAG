"""Langfuse 传播 e2e：graph 级 RunnableConfig.callbacks 到达节点内 LLM 调用，
节点 span 挂到根 trace 且 metadata 只含白名单标量。

评审修正意见落点：普通类替身的 ``config`` 形参恒为 None——contextvar 只在
langchain Runnable 内部经 ``ensure_config`` 生效，传播断言必须用继承
``BaseChatModel`` 的替身（真实现 ChatOpenAI/ChatOllama 即如此），否则假阴性。
"""

from __future__ import annotations

import json
import uuid
from typing import Any, ClassVar

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables.config import var_child_runnable_config

from src.agents import nodes as agent_nodes
from src.agents.graph import build_agent_with_checkpoint
from src.config import LangfuseConfig
from src.schemas.constants import ROLE_ADVISOR
from src.utils.langfuse_adapter import (
    ROOT_TRACE_NAME,
    LangfuseAdapter,
    reset_current_trace,
    set_current_trace,
)
from src.utils.metrics import MetricsRegistry

# 与 tests/e2e/conftest.py 的替身回答契约一致（tests 目录非包，无法导入）
QUERY_UNDERSTAND_MARKER = "请分析以下行业业务查询"
PLANNER_MARKER = "生成检索计划"
REASON_CONTENT = "## 结论\n\n本基金风险等级为R1（低风险），适合保守型投资者[来源1]。"

QUERY = "XX货币市场基金的风险等级是什么？"


class RecordingChatModel(BaseChatModel):
    """记录 ensure_config 注入的 RunnableConfig 的 BaseChatModel 替身。"""

    seen_configs: ClassVar[list[dict[str, Any] | None]] = []

    def _generate(
        self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any
    ) -> ChatResult:
        RecordingChatModel.seen_configs.append(var_child_runnable_config.get())
        prompt = (
            messages[-1].content if isinstance(messages[-1].content, str) else str(messages[-1].content)
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
        else:
            content = REASON_CONTENT
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

    @property
    def _llm_type(self) -> str:
        return "recording-chat"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "RecordingChatModel":
        return self


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


class MarkerCallbackHandler(BaseCallbackHandler):
    """标记 handler：断言其随 graph callbacks 传播到节点内模型调用。"""


@pytest.fixture()
def recording_llm(monkeypatch):
    RecordingChatModel.seen_configs = []
    fake = RecordingChatModel()
    monkeypatch.setattr(agent_nodes, "llm", fake)
    agent_nodes._get_bound_reason_model.cache_clear()
    yield fake
    agent_nodes._get_bound_reason_model.cache_clear()


def test_graph_callbacks_reach_node_llm_and_node_spans_attach_to_trace(
    recording_llm, isolated_stores, fake_retriever_factory
):
    """callbacks 经 graph.invoke 传播到节点内 llm.invoke；节点 span 挂根 trace。"""
    from src.utils.langfuse_adapter import start_node_span  # noqa: F401 传播依赖 nodes 导入

    client = FakeLangfuseClient()
    adapter = LangfuseAdapter(
        cfg=LangfuseConfig(
            enabled=True,
            host="https://cloud.langfuse.com",
            public_key=f"pk-lf-{uuid.uuid4().hex}",
            secret_key=f"sk-lf-{uuid.uuid4().hex}",
            sample_rate=1.0,
            capture_content=False,
        ),
        app_env="development",
        client=client,
        metrics=MetricsRegistry(),
    )
    trace = adapter.start_request_trace("req-e2e", "thread-e2e")
    marker = MarkerCallbackHandler()
    token = set_current_trace(trace)
    try:
        # 真实链路中 API 层先 ensure 会话再调图；这里同样先建会话再执行
        from tests.e2e.conftest import build_state

        thread = isolated_stores.conversation.create_thread(
            user_id="user_advisor", user_role=ROLE_ADVISOR, client_id=None, title="lf-wiring"
        )
        state = build_state(thread_id=thread["thread_id"], query=QUERY)
        graph = build_agent_with_checkpoint()
        result = graph.invoke(
            state,
            {
                "configurable": {"thread_id": state["thread_id"]},
                "callbacks": [marker],
            },
        )
    finally:
        reset_current_trace(token)
        # 生产中由 API 端点 finally 收尾；测试内等价模拟
        trace.finish(status="ok")

    assert result["final_answer"]
    # 1) graph 级 callbacks 经 ensure_config 到达节点内的模型调用（非假阴性：
    #    替身继承 BaseChatModel，走真实 ensure_config 传播路径）
    assert recording_llm.seen_configs, "模型调用未记录 config"
    assert all(cfg is not None for cfg in recording_llm.seen_configs)
    for cfg in recording_llm.seen_configs:
        callbacks = cfg.get("callbacks")
        assert callbacks is not None
        assert any(h is marker for h in callbacks.handlers)

    # 2) 根 trace 创建，节点 span 挂到同一 trace_id 之下
    root_span, root_kwargs = client.started[0]
    assert root_kwargs["name"] == ROOT_TRACE_NAME
    node_spans = [(span, kwargs) for span, kwargs in client.started[1:]]
    node_names = {kwargs["name"] for _, kwargs in node_spans}
    assert {
        "query_understand",
        "planner",
        "retrieve",
        "grade_and_filter",
        "call_reason_model",
        "extract_citations",
        "verify",
        "compliance_check",
        "compose",
    } <= node_names
    for _, kwargs in node_spans:
        assert kwargs["trace_context"]["trace_id"] == root_span.trace_id
        assert kwargs["trace_context"]["parent_span_id"] == root_span.id

    # 3) metadata 只含白名单标量：查询/回答/引用原文不进 payload
    for span, kwargs in client.started:
        metadata = kwargs.get("metadata") or {}
        serialized = json.dumps(metadata, ensure_ascii=False)
        assert QUERY not in serialized
        assert "XX货币市场基金2024年年度报告" not in serialized
        assert REASON_CONTENT not in serialized
    retrieve_meta = _final_metadata_for(node_spans, "retrieve")
    assert retrieve_meta["retrieval_count"] >= 1
    verify_meta = _final_metadata_for(node_spans, "verify")
    assert verify_meta["verification_status"] in ("passed", "failed")
    compose_meta = _final_metadata_for(node_spans, "compose")
    assert compose_meta["verification_status"] in ("passed", "failed")
    assert compose_meta["compliance_status"] in ("passed", "blocked")
    # 请求收尾：根 span 结束
    assert root_span.ended is True


def _span_for(node_spans: list[tuple[FakeLangfuseSpan, dict[str, Any]]], name: str) -> FakeLangfuseSpan:
    return next(span for span, kwargs in node_spans if kwargs["name"] == name)


def _final_metadata_for(
    node_spans: list[tuple[FakeLangfuseSpan, dict[str, Any]]], name: str
) -> dict[str, Any]:
    return _span_for(node_spans, name).updates[-1]["metadata"]
