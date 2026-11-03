"""MVP 验收契约（plan.md Task 9）：tests/evaluation/mvp_knowledge_qa.json 的 8 类场景。

执行方式与 tests/e2e 同源：真实 Agent Graph + 真实验证器/合规器 + 隔离存储（tmp_path），
LLM 与检索为测试替身；不调用外部 provider，不修改 data/*.db。
answer_status 由 src/utils/answer_status.py 对真实图终态解析，验证五态对外契约。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from src.agents import nodes as agent_nodes
from src.agents.graph import build_agent_with_checkpoint
from src.api.auth import AuthenticatedUser, authenticate_user, build_assistant_initial_state
from src.api.main import app
from src.retrieval import result_cache
from src.schemas.constants import (
    ANSWER_STATUS_ANSWERED,
    ANSWER_STATUS_BLOCKED,
    ANSWER_STATUS_NEEDS_CLARIFICATION,
    ANSWER_STATUS_NO_EVIDENCE,
    ANSWER_STATUS_PERMISSION_DENIED,
    API_ROUTE_ASSISTANT_QA,
    AUDIT_REQUEST_ID,
    META_CHUNK_ID,
    META_DOC_TYPE,
    META_PERMISSION_LEVEL,
    META_RETRIEVAL_SOURCE,
    META_SOURCE,
    META_TITLE,
    PLAN_SOURCE,
    RR_CONTENT,
    RR_DENIED,
    RR_METADATA,
    RR_REASON,
    RR_SCORE,
    ROLE_OPERATIONS,
    SOURCE_FAQ,
    SOURCE_PRODUCT,
    STATE_AUDIT_TRAIL,
    STATE_CITATIONS,
    STATE_COMPLIANCE,
    STATE_RETRIEVAL_PLAN,
    STATE_THREAD_ID,
    STATE_VERIFICATION,
)
from src.utils.answer_status import answer_status_from_state

EVAL_PATH = Path(__file__).resolve().parents[1] / "tests" / "evaluation" / "mvp_knowledge_qa.json"

# ── demo 语料中的稳定事实（虚构演示数据，见 data/raw/demo_knowledge_base）──

PRODUCT_SOURCE = "data/raw/demo_knowledge_base/samples/product/sample_wealth_product_risk_disclosure.html"
PRODUCT_TITLE = "示例理财产品风险揭示书摘要"
PRODUCT_CHUNK = (
    "产品名称：示例稳健增利理财产品（虚构）。产品类型：固定收益类理财产品，非保本浮动收益型。"
    "本产品风险等级评定为 R2（中低风险）。本产品仅适合风险承受能力评级为 C2（稳健型）及以上的业务者购买。"
)

FAQ_SOURCE = "data/raw/demo_knowledge_base/samples/faq/sample_internal_system_operation_faq.html"
FAQ_TITLE = "示例内部系统操作常见问题（FAQ）"
FAQ_CHUNK = (
    "Q：如何重置内部系统登录密码？A：登录内部门户后，点击右上角账户设置 → 安全中心 → 重置密码，"
    "输入原密码及短信验证码即可完成重置。忘记原密码时，需联系技术支持部门通过工号验证后重置。"
)

CONFIDENTIAL_SOURCE = "data/raw/demo_knowledge_base/samples/confidential/sample_confidential_restructuring_draft.html"
CONFIDENTIAL_TITLE = "未公开研究草稿：重大资产重组事项跟踪（内部）"

# query_understand 合并调用的 prompt 特征串（与 e2e conftest 同一契约）
QUERY_UNDERSTAND_MARKER = "一次返回查询理解与检索计划"


def _make_result(
    *,
    content: str,
    source: str,
    title: str,
    doc_type: str,
    retrieval_source: str,
    chunk_id: str,
    permission_level: str = "internal",
    allowed_roles: list[str] | None = None,
) -> dict[str, Any]:
    return {
        RR_CONTENT: content,
        RR_METADATA: {
            META_SOURCE: source,
            META_TITLE: title,
            META_DOC_TYPE: doc_type,
            META_RETRIEVAL_SOURCE: retrieval_source,
            META_CHUNK_ID: chunk_id,
            META_PERMISSION_LEVEL: permission_level,
            **({"allowed_roles": allowed_roles} if allowed_roles is not None else {}),
        },
        RR_SCORE: 0.9,
        RR_DENIED: False,
    }


def _denied_result(**kwargs: Any) -> dict[str, Any]:
    result = _make_result(**kwargs)
    result[RR_DENIED] = True
    result[RR_REASON] = "permission_denied"
    return result


def _make_product_results(count: int = 3) -> list[dict[str, Any]]:
    return [
        _make_result(
            content=PRODUCT_CHUNK,
            source=PRODUCT_SOURCE,
            title=PRODUCT_TITLE,
            doc_type="product",
            retrieval_source=SOURCE_PRODUCT,
            chunk_id=f"chunk-product-{i}",
            permission_level="public",
        )
        for i in range(count)
    ]


def _make_faq_results(count: int = 3) -> list[dict[str, Any]]:
    return [
        _make_result(
            content=FAQ_CHUNK,
            source=FAQ_SOURCE,
            title=FAQ_TITLE,
            doc_type="faq",
            retrieval_source=SOURCE_FAQ,
            chunk_id=f"chunk-faq-{i}",
            permission_level="internal",
        )
        for i in range(count)
    ]


# ── 测试替身 fixtures（与 tests/e2e/conftest.py 同一契约的最小子集）──


class _RoutedFakeChatModel:
    """按 query_understand 特征串路由的假 LLM；reason 返回可配置内容。"""

    def __init__(self) -> None:
        self.understand: dict[str, Any] = {
            "intent": "产品咨询",
            "query_type": "product_inquiry",
            "entities": {
                "product_name": "示例稳健增利理财产品",
                "product_type": "fund",
                "stock_code": "",
                "regulation_name": "",
                "client_segment": "",
                "time_range": {"start": "", "end": ""},
            },
            "rewritten_query": "示例稳健增利理财产品 风险等级",
            "ambiguity": [],
        }
        self.plan: list[dict[str, Any]] = [
            {"source": SOURCE_PRODUCT, "query": "示例稳健增利理财产品 风险等级", "top_k": 3}
        ]
        self.reason_content = (
            "## 结论\n\n本产品风险等级评定为 R2（中低风险），仅适合 C2（稳健型）及以上"
            "风险承受能力的业务者购买[来源1]。"
        )
        self.calls: list[str] = []

    def invoke(self, messages: list[Any], config: Any = None, **kwargs: Any) -> Any:
        from langchain_core.messages import AIMessage

        prompt = messages[-1].content if isinstance(messages[-1].content, str) else str(messages[-1].content)
        if QUERY_UNDERSTAND_MARKER in prompt:
            self.calls.append("query_understand")
            return AIMessage(content=json.dumps({**self.understand, "retrieval_plan": self.plan}, ensure_ascii=False))
        self.calls.append("reason")
        return AIMessage(content=self.reason_content)

    def bind_tools(self, tools: Any, **kwargs: Any) -> "_RoutedFakeChatModel":
        return self


@pytest.fixture()
def fake_llm(monkeypatch):
    fake = _RoutedFakeChatModel()
    monkeypatch.setattr(agent_nodes, "llm", fake)
    agent_nodes._get_bound_reason_model.cache_clear()  # pyright: ignore[reportFunctionMemberAccess]
    yield fake
    agent_nodes._get_bound_reason_model.cache_clear()  # pyright: ignore[reportFunctionMemberAccess]


@pytest.fixture()
def isolated_stores(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from src.utils.audit import SQLiteAuditStore
    from src.utils.conversation import SQLiteConversationStore

    conversation = SQLiteConversationStore(tmp_path / "conversations.db")
    audit = SQLiteAuditStore(tmp_path / "audit.db")
    monkeypatch.setattr(agent_nodes, "_get_conversation_store", lambda: conversation)
    monkeypatch.setattr(agent_nodes, "_get_audit_store", lambda: audit)
    monkeypatch.setattr(agent_nodes, "AUDIT_OUTBOX_PATH", str(tmp_path / "audit_outbox.jsonl"))
    result_cache.invalidate_retrieval_caches()
    yield SimpleNamespace(conversation=conversation, audit=audit)
    result_cache.invalidate_retrieval_caches()


@pytest.fixture()
def retriever_results():
    """检索结果注入口：测试直接改 holder['results']（或赋可调用）。"""
    return {"results": _make_product_results()}


@pytest.fixture()
def fake_retriever(monkeypatch, retriever_results):
    holder = retriever_results

    class FakeHybridRetriever:
        def __init__(self, user_role: str, data_permissions: list[str] | None = None):
            self.user_role = user_role
            self.data_permissions = data_permissions or []

        def retrieve(self, plan: list[dict[str, Any]]) -> list[dict[str, Any]]:
            results = holder["results"]
            if callable(results):
                return list(results(plan, self))
            return list(results)

    monkeypatch.setattr(agent_nodes, "HybridRetriever", FakeHybridRetriever)
    return holder


def _run_graph(state: dict[str, Any]) -> dict[str, Any]:
    graph = build_agent_with_checkpoint()
    return dict(
        graph.invoke(state, {"configurable": {"thread_id": state[STATE_THREAD_ID]}})
    )


def _operations_state(query: str, thread_id: str, user_id: str = "user_ops") -> dict[str, Any]:
    user = AuthenticatedUser(user_id=user_id, role=ROLE_OPERATIONS, department="ops")
    from src.schemas.request_response import AssistantQARequest

    request = AssistantQARequest(query=query, thread_id=thread_id)
    return dict(build_assistant_initial_state(request, user, thread_id=thread_id))


def _thread(stores: Any, user_id: str = "user_ops") -> str:
    return stores.conversation.create_thread(
        user_id=user_id, user_role=ROLE_OPERATIONS, client_id=None, title="MVP"
    )["thread_id"]


def _load_scenarios() -> dict[str, dict[str, Any]]:
    return {
        scenario["id"]: scenario
        for scenario in json.loads(EVAL_PATH.read_text(encoding="utf-8"))["scenarios"]
    }


# ── 验收集自身契约 ──


def test_mvp_acceptance_set_declares_eight_scenarios():
    scenarios = json.loads(EVAL_PATH.read_text(encoding="utf-8"))["scenarios"]
    assert len(scenarios) == 8
    ids = {s["id"] for s in scenarios}
    assert ids == {
        "S1_answered_product_risk_level",
        "S2_answered_faq_operation_flow",
        "S3_no_evidence_market_quote",
        "S4_needs_clarification_ambiguous_term",
        "S5_permission_denied_restricted_doc",
        "S6_blocked_investment_advice",
        "S7_followup_same_thread_no_cross_user_leak",
        "S8_service_unavailable_never_answered",
    }


def test_mvp_acceptance_status_values_are_the_product_contract_five():
    allowed = {
        ANSWER_STATUS_ANSWERED,
        ANSWER_STATUS_NO_EVIDENCE,
        ANSWER_STATUS_PERMISSION_DENIED,
        ANSWER_STATUS_NEEDS_CLARIFICATION,
        ANSWER_STATUS_BLOCKED,
    }
    for scenario in json.loads(EVAL_PATH.read_text(encoding="utf-8"))["scenarios"]:
        for turn in scenario.get("turns", [scenario]):
            assert turn["expected_status"] in allowed | {"transport_error"}


# ── 场景 S1/S2：answered + 计划源约束 + 引用存在 ──


@pytest.mark.parametrize(
    ("query", "plan_source", "doc_type", "results_factory", "reason_content"),
    [
        (
            "示例稳健增利理财产品的风险等级是多少？",
            SOURCE_PRODUCT,
            "product",
            _make_product_results,
            "## 结论\n\n本产品风险等级评定为 R2（中低风险），仅适合 C2（稳健型）及以上"
            "风险承受能力的业务者购买[来源1]。",
        ),
        (
            "内部系统登录密码忘记后如何重置？",
            SOURCE_FAQ,
            "faq",
            _make_faq_results,
            "## 结论\n\n忘记原密码时，需联系技术支持部门通过工号验证后重置[来源1]。",
        ),
    ],
)
def test_answered_scenarios_return_citations_from_curated_sources(
    fake_llm, isolated_stores, fake_retriever, query, plan_source, doc_type, results_factory, reason_content
):
    scenario = _load_scenarios()
    expected = scenario[
        "S1_answered_product_risk_level"
        if doc_type == "product"
        else "S2_answered_faq_operation_flow"
    ]
    assert expected["expected_status"] == "answered"
    assert plan_source in expected["required_source_types"]

    fake_llm.plan = [{"source": plan_source, "query": query, "top_k": 3}]
    fake_llm.understand["rewritten_query"] = query
    fake_llm.reason_content = reason_content
    # 理解意图与查询语义一致，避免意图补源引入场景外的检索源
    if plan_source == SOURCE_FAQ:
        fake_llm.understand["intent"] = "FAQ"
        fake_llm.understand["query_type"] = "faq_inquiry"
    else:
        fake_llm.understand["intent"] = "产品咨询"
        fake_llm.understand["query_type"] = "product_inquiry"
    fake_retriever["results"] = results_factory()

    state = _run_graph(_operations_state(query, _thread(isolated_stores)))

    assert answer_status_from_state(state) == "answered"
    assert state[STATE_VERIFICATION]["passed"] is True
    assert state[STATE_COMPLIANCE]["passed"] is True
    # 计划必须覆盖场景声明的检索源，且不超出 profile 白名单
    planned_sources = {step[PLAN_SOURCE] for step in state[STATE_RETRIEVAL_PLAN]}
    assert plan_source in planned_sources
    assert planned_sources <= {SOURCE_PRODUCT, SOURCE_FAQ}
    # 引用存在且来源类型匹配
    citations = state[STATE_CITATIONS]
    assert citations, "answered 场景必须带引用"
    citation_meta = citations[0].get("metadata", {})
    assert citation_meta.get("doc_type") == doc_type
    assert citation_meta.get("retrieval_source") == plan_source
    # 审计仍服务端持久化
    request_id = state[STATE_AUDIT_TRAIL][AUDIT_REQUEST_ID]
    assert isolated_stores.audit.get_by_request_id(request_id) is not None


# ── 场景 S3：知识库无实时行情 → no_evidence，无引用 ──


def test_no_evidence_scenario_rejects_without_citations(
    fake_llm, isolated_stores, fake_retriever
):
    scenario = _load_scenarios()["S3_no_evidence_market_quote"]
    assert scenario["expected_status"] == "no_evidence"

    fake_retriever["results"] = []  # 检索耗尽仍 0 召回

    state = _run_graph(_operations_state(scenario["query"], _thread(isolated_stores)))

    assert answer_status_from_state(state) == "no_evidence"
    assert state[STATE_CITATIONS] == []
    assert state[STATE_VERIFICATION]["passed"] is False
    assert "no_retrieval_results" in state[STATE_VERIFICATION]["issues"]
    # 未通过验证的终态不得被缓存为成功答案（缓存在 API 层执行，这里验证状态来源）
    assert "未找到" in state["final_answer"]


# ── 场景 S4：歧义简称 → needs_clarification ──


def test_ambiguous_scenario_terminates_in_clarification(
    fake_llm, isolated_stores, fake_retriever
):
    scenario = _load_scenarios()["S4_needs_clarification_ambiguous_term"]
    assert scenario["expected_status"] == "needs_clarification"

    fake_llm.understand["ambiguity"] = ["产品简称不明确：请确认您指的是哪一款产品或哪个指标（如市盈率 PE）。"]
    fake_llm.understand["intent"] = "产品咨询"
    # 澄清发生在检索之前
    fake_retriever["results"] = _make_product_results()

    state = _run_graph(_operations_state(scenario["query"], _thread(isolated_stores)))

    assert answer_status_from_state(state) == "needs_clarification"
    assert state[STATE_CITATIONS] == []
    assert fake_llm.calls.count("reason") == 0, "澄清终态不得进入推理"
    assert "为了给您更准确的回答" in state["final_answer"]


def test_clarification_does_not_leak_into_followup_turn_on_persistent_graph(
    fake_llm, isolated_stores, fake_retriever
):
    """生产 API 使用进程级单例图 + thread 级 checkpointer：同线程澄清后的下一轮
    正常提问不得残留 clarification_needed（final review F1）。"""
    fake_retriever["results"] = _make_product_results()
    graph = build_agent_with_checkpoint()
    thread_id = _thread(isolated_stores)

    def _run(state: dict[str, Any]) -> dict[str, Any]:
        return dict(graph.invoke(state, {"configurable": {"thread_id": state[STATE_THREAD_ID]}}))

    fake_llm.understand["ambiguity"] = ["产品简称不明确：请确认您指的是哪一款产品或哪个指标。"]
    turn1 = _run(_operations_state("这款产品的PE是多少？", thread_id))
    assert answer_status_from_state(turn1) == "needs_clarification"

    fake_llm.understand["ambiguity"] = []
    turn2 = _run(_operations_state("示例稳健增利理财产品的风险等级是多少？", thread_id))
    assert answer_status_from_state(turn2) == "answered"


# ── 场景 S5：受限文档 → permission_denied，无引用 ──


def test_restricted_document_scenario_denies_permission(
    fake_llm, isolated_stores, fake_retriever
):
    scenario = _load_scenarios()["S5_permission_denied_restricted_doc"]
    assert scenario["expected_status"] == "permission_denied"

    fake_llm.plan = [{"source": SOURCE_PRODUCT, "query": scenario["query"], "top_k": 3}]
    fake_retriever["results"] = [
        _denied_result(
            content="重大资产重组事项内部跟踪草稿（机密）。",
            source=CONFIDENTIAL_SOURCE,
            title=CONFIDENTIAL_TITLE,
            doc_type="research_report",
            retrieval_source=SOURCE_PRODUCT,
            chunk_id="chunk-conf-1",
            permission_level="confidential",
            allowed_roles=["advisor", "institutional_sales", "compliance", "technical"],
        )
    ]

    state = _run_graph(_operations_state(scenario["query"], _thread(isolated_stores)))

    assert answer_status_from_state(state) == "permission_denied"
    assert state[STATE_CITATIONS] == []
    assert "无权限" in state["final_answer"]
    assert state[STATE_COMPLIANCE]["passed"] is False


# ── 场景 S6：投资建议 → blocked，无引用 ──


def test_investment_advice_scenario_blocked(fake_llm, isolated_stores, fake_retriever):
    scenario = _load_scenarios()["S6_blocked_investment_advice"]
    assert scenario["expected_status"] == "blocked"

    fake_llm.reason_content = (
        "## 结论\n\n该产品当前表现稳健，建议买入该产品锁定收益[来源1]。"
    )
    fake_retriever["results"] = _make_product_results()

    state = _run_graph(_operations_state(scenario["query"], _thread(isolated_stores)))

    assert answer_status_from_state(state) == "blocked"
    # 拦截路径：答案替换为安全兜底文案，引用清空（验证失败兜底或合规拦截均可，
    # 两者按契约同属 blocked，不得伪装成成功答案）
    assert state[STATE_CITATIONS] == []
    verification_passed = state[STATE_VERIFICATION].get("passed", False)
    compliance_passed = state[STATE_COMPLIANCE].get("passed", False)
    assert not (verification_passed and compliance_passed)
    fallback_text = state["final_answer"]
    assert (
        "未通过来源或数字验证" in fallback_text
        or "未通过合规检查" in fallback_text
        or "停止输出" in fallback_text
    )


# ── 场景 S7：同会话两轮追问 + 跨用户隔离 ──


def test_followup_turns_keep_answer_status_and_isolate_users(
    fake_llm, isolated_stores, fake_retriever
):
    scenario = _load_scenarios()["S7_followup_same_thread_no_cross_user_leak"]
    assert [turn["expected_status"] for turn in scenario["turns"]] == ["answered", "answered"]
    fake_retriever["results"] = _make_product_results()

    thread_id = _thread(isolated_stores, "user_ops_a")

    turn1 = _run_graph(_operations_state(scenario["turns"][0]["query"], thread_id, user_id="user_ops_a"))
    turn2 = _run_graph(_operations_state(scenario["turns"][1]["query"], thread_id, user_id="user_ops_a"))

    assert answer_status_from_state(turn1) == "answered"
    assert answer_status_from_state(turn2) == "answered"

    # 同一用户两轮都在同一会话内可见
    messages_a = isolated_stores.conversation.list_messages(thread_id=thread_id, user_id="user_ops_a")
    assert len([m for m in messages_a if m["role"] == "assistant"]) >= 2

    # 跨用户：另一位用户在同一 thread_id 上读不到该会话（存储层 fail closed 直接拒绝）
    from src.utils.conversation import ConversationNotFoundError

    with pytest.raises(ConversationNotFoundError):
        isolated_stores.conversation.list_messages(thread_id=thread_id, user_id="user_ops_b")


# ── 场景 S8：服务不可用 → HTTP 错误，不得伪装成 answered ──


class _UnavailableAgentApp:
    def invoke(self, state, config=None):
        import httpx

        raise httpx.ConnectError("connection refused")


@pytest.fixture()
def api_client(monkeypatch, tmp_path):
    from src.utils.audit import SQLiteAuditStore
    from src.utils.conversation import SQLiteConversationStore

    conversation = SQLiteConversationStore(tmp_path / "conversations.db")
    monkeypatch.setattr("src.api.qa._get_conversation_store", lambda: conversation)
    monkeypatch.setattr("src.api.qa.get_semantic_cache", lambda: _DisabledCache())
    monkeypatch.setattr("src.api.qa._get_cache_hit_audit_store", lambda: SQLiteAuditStore(tmp_path / "audit.db"))
    app.dependency_overrides[authenticate_user] = lambda: AuthenticatedUser(
        "user_ops", ROLE_OPERATIONS, "ops"
    )
    yield TestClient(app)
    app.dependency_overrides.clear()


class _DisabledCache:
    def lookup(self, query, role="", **kwargs):
        return None

    def store(self, query, answer, citations=None, confidence="", role="", **kwargs):
        return True


def test_provider_unavailable_returns_http_error_not_answered(api_client, monkeypatch):
    scenario = _load_scenarios()["S8_service_unavailable_never_answered"]
    assert scenario["expected_status"] == "transport_error"
    monkeypatch.setattr("src.api.qa._get_agent_app", lambda: _UnavailableAgentApp())

    res = api_client.post(
        API_ROUTE_ASSISTANT_QA,
        json={"query": scenario["query"]},
    )

    assert res.status_code == 503
    assert "answer_status" not in res.json()
