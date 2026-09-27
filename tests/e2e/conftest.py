"""全链路测试战役共享 fixture（docs/test-plans/e2e-test-cases.md）。

原则：LLM / 向量检索 / 外部网络一律替换为测试替身；SQLite 存储、ChromaDB、
注册表落到 tmp_path，保证用例可重复运行、相互独立。
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, cast

import pytest
from langchain_core.messages import AIMessage, BaseMessage

from src.agents import nodes as agent_nodes
from src.agents.graph import build_agent_with_checkpoint
from src.agents.state import AssistantState
from src.api.auth import AuthenticatedUser, build_assistant_initial_state
from src.retrieval import result_cache
from src.schemas.constants import (
    META_CHUNK_ID,
    META_DOC_TYPE,
    META_PERMISSION_LEVEL,
    META_SOURCE,
    META_TITLE,
    PERMISSION_PUBLIC,
    RR_CONTENT,
    RR_DENIED,
    RR_METADATA,
    RR_SCORE,
    ROLE_ADVISOR,
)

FUND_REPORT_SOURCE = "data/raw/reports/xx_money_fund_2024.html"
FUND_REPORT_TITLE = "XX货币市场基金2024年年度报告（摘要）"
FUND_REPORT_CHUNK = (
    "根据基金合同与2024年年度报告，本基金风险等级为R1（低风险），"
    "适合保守型投资者。基金主要投资于货币市场工具，不投资股票或可转换债券。"
)

QUERY_UNDERSTAND_MARKER = "请分析以下行业业务查询"
PLANNER_MARKER = "生成检索计划"


def make_fund_result(**overrides: Any) -> dict[str, Any]:
    """一条可复用的 public 财报检索结果。"""
    result: dict[str, Any] = {
        RR_CONTENT: FUND_REPORT_CHUNK,
        RR_METADATA: {
            META_SOURCE: FUND_REPORT_SOURCE,
            META_TITLE: FUND_REPORT_TITLE,
            META_DOC_TYPE: "research_report",
            META_CHUNK_ID: "chunk-001",
            META_PERMISSION_LEVEL: PERMISSION_PUBLIC,
        },
        RR_SCORE: 0.92,
        RR_DENIED: False,
    }
    result[RR_METADATA].update(overrides.pop("metadata", {}))
    result.update(overrides)
    return result


class FakeChatModel:
    """按 prompt 关键词路由的假 LLM。

    query_understand / planner 的 prompt 走 JSON 契约；其余调用视为
    ReAct 推理，返回无工具调用的 AIMessage。
    """

    def __init__(self) -> None:
        self.query_understand_response: dict[str, Any] = {
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
        }
        self.plan_response: list[dict[str, Any]] = [
            {"source": "product_search", "query": "XX货币市场基金 风险等级", "top_k": 3}
        ]
        self.reason_content = "## 结论\n\n本基金风险等级为R1（低风险），适合保守型投资者[来源1]。"
        self.calls: list[tuple[str, list[BaseMessage]]] = []

    def invoke(self, messages: list[BaseMessage], config: Any = None) -> AIMessage:
        prompt = messages[-1].content if isinstance(messages[-1].content, str) else str(messages[-1].content)
        if QUERY_UNDERSTAND_MARKER in prompt:
            self.calls.append(("query_understand", messages))
            return AIMessage(content=json.dumps(self.query_understand_response, ensure_ascii=False))
        if PLANNER_MARKER in prompt:
            self.calls.append(("planner", messages))
            return AIMessage(content=json.dumps(self.plan_response, ensure_ascii=False))
        self.calls.append(("reason", messages))
        return AIMessage(content=self.reason_content)

    def bind_tools(self, tools: Any, **kwargs: Any) -> "FakeChatModel":
        return self


@pytest.fixture()
def fake_llm(monkeypatch):
    """替换节点模块级 LLM；清理 _get_bound_reason_model 的 lru_cache 防止跨用例污染。"""
    fake = FakeChatModel()
    monkeypatch.setattr(agent_nodes, "llm", fake)
    agent_nodes._get_bound_reason_model.cache_clear()
    yield fake
    agent_nodes._get_bound_reason_model.cache_clear()


@pytest.fixture()
def isolated_stores(monkeypatch, tmp_path):
    """会话/审计/outbox 落到 tmp；检索结果 TTL 缓存用例间清理。"""
    from src.utils.audit import SQLiteAuditStore
    from src.utils.conversation import SQLiteConversationStore

    conversation = SQLiteConversationStore(tmp_path / "conversations.db")
    audit = SQLiteAuditStore(tmp_path / "audit.db")
    monkeypatch.setattr(agent_nodes, "_get_conversation_store", lambda: conversation)
    monkeypatch.setattr(agent_nodes, "_get_audit_store", lambda: audit)
    monkeypatch.setattr(
        agent_nodes, "AUDIT_OUTBOX_PATH", str(tmp_path / "audit_outbox.jsonl")
    )
    result_cache.invalidate_retrieval_caches()
    yield SimpleNamespace(
        conversation=conversation, audit=audit, tmp_path=tmp_path
    )
    result_cache.invalidate_retrieval_caches()


@pytest.fixture()
def fake_retriever_factory(monkeypatch):
    """替换 nodes.HybridRetriever；测试通过 holder["results"] 注入结果或异常。"""
    holder: dict[str, Any] = {"results": [make_fund_result()]}

    class FakeHybridRetriever:
        def __init__(self, user_role: str, data_permissions: list[str] | None = None):
            self.user_role = user_role
            self.data_permissions = data_permissions or []
            holder.setdefault("instances", []).append(self)

        def retrieve(self, plan: list[dict[str, Any]]) -> list[dict[str, Any]]:
            results = holder["results"]
            if callable(results):
                return cast(list[dict[str, Any]], results(plan, self))
            return list(results)

    monkeypatch.setattr(agent_nodes, "HybridRetriever", FakeHybridRetriever)
    holder["class"] = FakeHybridRetriever
    return holder


def build_state(
    query: str = "XX货币市场基金的风险等级是什么？", **user_kwargs: Any
) -> AssistantState:
    """构造 Agent Graph 初始 state（advisor 默认）。"""
    from src.schemas.request_response import AssistantQARequest

    user = AuthenticatedUser(
        user_id=user_kwargs.pop("user_id", "user_advisor"),
        role=user_kwargs.pop("role", ROLE_ADVISOR),
        department=user_kwargs.pop("department", "wealth"),
    )
    request = AssistantQARequest(query=query, **user_kwargs.pop("request_kwargs", {}))
    return build_assistant_initial_state(request, user, **user_kwargs)


@pytest.fixture()
def run_agent_graph(fake_llm, isolated_stores, fake_retriever_factory):
    """返回可调用的图执行器：fresh checkpointer 保证用例独立。"""
    from src.schemas.constants import STATE_THREAD_ID

    def _run(state: AssistantState) -> dict[str, Any]:
        graph = build_agent_with_checkpoint()
        return cast(
            dict[str, Any],
            graph.invoke(
                state, {"configurable": {STATE_THREAD_ID: state["thread_id"]}}
            ),
        )

    return _run
