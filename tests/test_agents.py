"""Agent Graph 节点和路由函数单元测试"""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from src.agents.graph import (
    _traced_node,
    build_agent_graph,
    build_reason_subgraph,
    is_compliant,
    should_reason_again,
    should_retry_retrieval,
)
from src.agents.nodes import (
    _get_bound_reason_model,
    _structure_answer,
    audit_log,
    compliance_check,
    compose,
    extract_citations,
    grade_and_filter,
    planner,
    prepare_reason,
    retrieve,
    verify,
)
from src.agents.state import AssistantState
from src.schemas.constants import (
    AUDIT_COMPLIANCE,
    AUDIT_QUERY,
    AUDIT_QUERY_ORIGINAL,
    AUDIT_REASONING,
    AUDIT_REQUEST_ID,
    AUDIT_RESPONSE,
    AUDIT_RESPONSE_CONFIDENCE,
    AUDIT_RETRIEVAL,
    AUDIT_RETRIEVAL_SOURCES,
    AUDIT_RETRIEVAL_TOTAL_CHUNKS,
    AUDIT_STARTED_PERF_COUNTER,
    AUDIT_TIMESTAMP,
    AUDIT_VERIFICATION,
    CONFIDENCE_HIGH,
    CONFIDENCE_LOW,
    CONFIDENCE_MEDIUM,
    MAX_TOOL_ITERATIONS,
    META_CHUNK_ID,
    META_ALLOWED_ROLES,
    META_PERMISSION_LEVEL,
    META_SOURCE,
    META_TITLE,
    PLAN_FILTERS,
    PLAN_QUERY,
    PLAN_SOURCE,
    PLAN_TOP_K,
    ROLE_ADVISOR,
    ROLE_COMPLIANCE,
    ROLE_INSTITUTIONAL_SALES,
    ROLE_OPERATIONS,
    ROLE_TECHNICAL,
    RR_CONTENT,
    RR_DENIED,
    RR_METADATA,
    RR_SCORE,
    SOURCE_FAQ,
    SOURCE_PRODUCT,
    SOURCE_REPORT,
    STATE_AUDIT_TRAIL,
    STATE_CITATIONS,
    STATE_CLIENT_ID,
    STATE_COMPLIANCE,
    STATE_CONFIDENCE,
    STATE_DEPARTMENT,
    STATE_ENTITIES,
    STATE_FINAL_ANSWER,
    STATE_INTENT,
    STATE_MESSAGES,
    STATE_INTERMEDIATE_STEPS,
    STATE_ORIGINAL_QUERY,
    STATE_QUERY_TYPE,
    STATE_REASON_ATTEMPTS,
    STATE_REASON_MESSAGE_START,
    STATE_REASON_STARTED_PERF_COUNTER,
    STATE_RETRIEVAL_ATTEMPTS,
    STATE_RETRIEVAL_FILTERED_CHUNKS,
    STATE_RETRIEVAL_PLAN,
    STATE_RETRIEVAL_RESULTS,
    STATE_RETRIEVAL_TOTAL_CHUNKS,
    STATE_REWRITTEN_QUERY,
    STATE_TOOL_CALLS,
    STATE_TOOL_ITERATIONS,
    STATE_TOOL_MESSAGE_CURSOR,
    STATE_DATA_PERMISSIONS,
    STATE_USER_ROLE,
    STATE_VERIFICATION,
)
from src.utils.audit import SQLiteAuditStore
from src.utils.verifier import CitationExtractor, SourceVerifier


def _result(content: str, score: float = 0.9, meta: dict[str, Any] | None = None) -> dict[str, Any]:
    """快捷构造检索结果 dict"""
    return {
        RR_CONTENT: content,
        RR_SCORE: score,
        RR_METADATA: meta or {META_TITLE: "测试文档", META_SOURCE: "test.pdf"},
    }


def _state(**overrides: Any) -> AssistantState:
    """快捷构造最小 AssistantState"""
    return cast(
        AssistantState,
        {
            STATE_RETRIEVAL_RESULTS: [],
            STATE_FINAL_ANSWER: "",
            STATE_USER_ROLE: ROLE_OPERATIONS,
            STATE_VERIFICATION: {},
            STATE_COMPLIANCE: {},
            STATE_MESSAGES: [],
            STATE_TOOL_CALLS: [],
            STATE_INTERMEDIATE_STEPS: [],
            STATE_REASON_ATTEMPTS: 0,
            STATE_TOOL_ITERATIONS: 0,
            STATE_REASON_MESSAGE_START: 0,
            STATE_TOOL_MESSAGE_CURSOR: 0,
            STATE_REASON_STARTED_PERF_COUNTER: 0.0,
            **overrides,
        },
    )


ADVICE_BUY = "推荐" + "买" + "入"
ADVICE_SELL = "建议" + "卖" + "出"
RESTRICTED_TEXT = "限制级测试内容"
SENSITIVE_FIXTURE_TEXT = "内" + "幕" + "信息"
HIGH_RISK_PRODUCT = "私" + "募" + "产品"


# ══════════════════════════════════════════════════════════════════════
# grade_and_filter
# ══════════════════════════════════════════════════════════════════════


class TestGradeAndFilter:
    def test_empty_results(self):
        state = _state()
        result = grade_and_filter(state)
        assert result == {}

    def test_sorts_by_score_desc(self):
        r1 = _result("a", score=0.7)
        r2 = _result("b", score=0.9)
        r3 = _result("c", score=0.65)
        state = _state(**{STATE_RETRIEVAL_RESULTS: [r1, r2, r3]})
        result = grade_and_filter(state)
        scores = [r[RR_SCORE] for r in result[STATE_RETRIEVAL_RESULTS]]
        assert scores == [0.9, 0.7, 0.65]

    def test_filters_low_relevance_results(self):
        state = _state(**{
            STATE_RETRIEVAL_RESULTS: [_result("weak", score=0.52), _result("strong", score=0.8)]
        })
        result = grade_and_filter(state)

        assert [r[RR_CONTENT] for r in result[STATE_RETRIEVAL_RESULTS]] == ["strong"]

    def test_truncates_to_grade_top_k(self):
        results = [_result(f"doc{i}", score=0.9 - i * 0.01) for i in range(20)]
        state = _state(**{STATE_RETRIEVAL_RESULTS: results})
        result = grade_and_filter(state)
        assert len(result[STATE_RETRIEVAL_RESULTS]) <= 10

    def test_deduplicates_same_source_chunk_across_retrieval_hops(self):
        duplicate = _result(
            "same",
            meta={META_SOURCE: "faq.html", META_CHUNK_ID: "chunk-1"},
        )
        state = _state(**{STATE_RETRIEVAL_RESULTS: [duplicate, duplicate.copy()]})

        result = grade_and_filter(state)

        assert len(result[STATE_RETRIEVAL_RESULTS]) == 1

    def test_records_filtered_chunk_count(self):
        state = _state(**{
            STATE_RETRIEVAL_RESULTS: [_result("weak", score=0.4), _result("strong", score=0.8)]
        })

        result = grade_and_filter(state)

        assert result[STATE_RETRIEVAL_FILTERED_CHUNKS] == 1

    def test_does_not_reemit_other_state_fields(self):
        state = _state(**{STATE_FINAL_ANSWER: "unchanged"})
        result = grade_and_filter(state)
        assert STATE_FINAL_ANSWER not in result


def test_planner_injects_stock_code_filter_for_report_search(monkeypatch):
    response = MagicMock()
    response.content = json.dumps([
        {PLAN_SOURCE: SOURCE_REPORT, PLAN_QUERY: "贵州茅台研报", PLAN_TOP_K: 5}
    ])

    class FakeLLM:
        def invoke(self, messages):
            return response

    monkeypatch.setattr("src.agents.nodes.llm", FakeLLM())
    state = _state(**{
        STATE_USER_ROLE: ROLE_ADVISOR,
        STATE_ORIGINAL_QUERY: "贵州茅台评级",
        STATE_REWRITTEN_QUERY: "贵州茅台600519研报评级",
        STATE_INTENT: "研报观点",
        STATE_QUERY_TYPE: "report_inquiry",
        STATE_ENTITIES: {"stock_code": "600519.SH"},
    })

    result = planner(state)

    assert result[STATE_RETRIEVAL_PLAN][0][PLAN_FILTERS] == {"stock_code": "600519"}


# ══════════════════════════════════════════════════════════════════════
# verify
# ══════════════════════════════════════════════════════════════════════


class TestVerify:
    def test_citation_quote_contains_structured_metadata_and_deduplicates_headers(self):
        extractor = CitationExtractor()
        metadata = {
            META_TITLE: "贵州茅台研报",
            META_SOURCE: "report.pdf",
            "institution": "诚通证券",
            "rating": "买入",
            "date": "2026-05-25",
            "stock_code": "600519",
        }

        citations = extractor.extract(
            [
                _result("2026 年 05 月 20 日贵州茅台", meta={**metadata, META_CHUNK_ID: "a"}),
                _result("2026 年 05 月 20 日贵州茅台", meta={**metadata, META_CHUNK_ID: "b"}),
            ],
            query="贵州茅台的评级和机构",
        )

        assert len(citations) == 1
        assert "机构=诚通证券" in citations[0]["quote"]
        assert "评级=买入" in citations[0]["quote"]
        assert "来源日期=2026-05-25" in citations[0]["quote"]

    def test_source_verifier_rejects_visible_citation_that_omits_structured_claim(self):
        verifier = SourceVerifier()
        result = verifier.verify(
            answer="研报评级为买入，机构为诚通证券。",
            citations=[{
                "source": "report.pdf",
                "chunk_id": "chunk-1",
                "quote": "贵州茅台研报",
                "metadata": {"rating": "买入", "institution": "诚通证券"},
            }],
            retrieval_results=[
                _result(
                    "贵州茅台研报",
                    meta={META_SOURCE: "report.pdf", META_CHUNK_ID: "chunk-1"},
                )
            ],
        )

        assert result["passed"] is False
        assert any("可见引用未包含" in issue for issue in result["issues"])

    def test_verification_accepts_whitelisted_metadata_as_grounding_evidence(self):
        retrieval_results = [
            _result(
                "贵州茅台研报",
                meta={
                    META_TITLE: "贵州茅台2025年年报及2026年一季报点评",
                    META_SOURCE: "report.pdf",
                    META_CHUNK_ID: "chunk-1",
                    "institution": "诚通证券",
                    "rating": "买入",
                    "date": "2026-05-25",
                    "stock_code": "600519",
                },
            )
        ]
        citations = CitationExtractor().extract(
            retrieval_results,
            query="贵州茅台研报评级和机构",
        )
        state = _state(**{
            STATE_FINAL_ANSWER: (
                "贵州茅台2025年年报及2026年一季报点评由诚通证券发布，"
                "评级为买入，来源日期为2026-05-25，股票代码600519。"
            ),
            STATE_RETRIEVAL_RESULTS: retrieval_results,
            STATE_CITATIONS: citations,
        })

        result = verify(state)

        assert result[STATE_VERIFICATION]["passed"] is True

    def test_extract_citations_runs_before_verification(self):
        state = _state(**{
            STATE_ORIGINAL_QUERY: "净利润",
            STATE_RETRIEVAL_RESULTS: [
                _result(
                    "净利润 747 亿元",
                    meta={
                        META_TITLE: "2024年报",
                        META_SOURCE: "report.pdf",
                        META_CHUNK_ID: "chunk_001",
                    },
                )
            ],
        })
        result = extract_citations(state)
        assert result[STATE_CITATIONS][0]["chunk_id"] == "chunk_001"

    def test_numbers_without_results(self):
        state = _state(**{
            STATE_FINAL_ANSWER: "净利润 747 亿元",
            STATE_RETRIEVAL_RESULTS: [],
        })
        result = verify(state)
        verification = result[STATE_VERIFICATION]
        assert verification.get("passed") is False
        assert any("数字" in i for i in verification.get("issues", []))

    def test_numbers_with_results_passes(self):
        state = _state(**{
            STATE_FINAL_ANSWER: "净利润 747 亿元",
            STATE_RETRIEVAL_RESULTS: [_result("净利润 747 亿元")],
        })
        result = verify(state)
        assert result[STATE_VERIFICATION].get("passed") is True

    def test_tool_only_answer_uses_successful_tool_output_as_evidence(self):
        state = _state(**{
            STATE_FINAL_ANSWER: "平安银行研报由示例机构发布",
            STATE_RETRIEVAL_RESULTS: [],
            STATE_TOOL_CALLS: [
                {
                    "tool": "sql_query_tool",
                    "output": "平安银行研报由示例机构发布",
                    "success": True,
                }
            ],
        })

        result = verify(state)

        assert result[STATE_VERIFICATION].get("passed") is True

    def test_failed_tool_output_is_not_grounding_evidence(self):
        state = _state(**{
            STATE_FINAL_ANSWER: "平安银行研报由示例机构发布",
            STATE_RETRIEVAL_RESULTS: [],
            STATE_TOOL_CALLS: [
                {
                    "tool": "sql_query_tool",
                    "output": "平安银行研报由示例机构发布",
                    "success": False,
                }
            ],
        })

        result = verify(state)

        assert result[STATE_VERIFICATION].get("passed") is False
        assert any("成功工具输出" in issue for issue in result[STATE_VERIFICATION]["issues"])

    def test_markdown_table_is_grounded_by_structured_tool_output(self):
        state = _state(**{
            STATE_FINAL_ANSWER: (
                "### 样本\n"
                "| stock_code | report_date | eps_2026 |\n"
                "|---|---|---|\n"
                "| 000001 | 2026-04-26 | 2.08 |"
            ),
            STATE_RETRIEVAL_RESULTS: [],
            STATE_TOOL_CALLS: [
                {
                    "tool": "sql_query_tool",
                    "output": (
                        '[{"stock_code":"000001","report_date":"2026-04-26",'
                        '"eps_2026":2.08}]'
                    ),
                    "success": True,
                }
            ],
        })

        result = verify(state)

        assert result[STATE_VERIFICATION].get("passed") is True

    def test_structured_tool_output_does_not_support_missing_value(self):
        state = _state(**{
            STATE_FINAL_ANSWER: "| stock_code | eps_2026 |\n|---|---|\n| 000001 | 9.99 |",
            STATE_RETRIEVAL_RESULTS: [],
            STATE_TOOL_CALLS: [
                {
                    "tool": "sql_query_tool",
                    "output": '[{"stock_code":"000001","eps_2026":2.08}]',
                    "success": True,
                }
            ],
        })

        result = verify(state)

        assert result[STATE_VERIFICATION].get("passed") is False

    def test_numbers_not_in_results_fails(self):
        state = _state(**{
            STATE_FINAL_ANSWER: "净利润 888 亿元",
            STATE_RETRIEVAL_RESULTS: [_result("净利润 747 亿元")],
        })
        result = verify(state)
        verification = result[STATE_VERIFICATION]
        assert verification.get("passed") is False
        assert any("888" in issue for issue in verification.get("issues", []))

    def test_investment_advice_blocked_for_advisor(self):
        state = _state(**{
            STATE_USER_ROLE: ROLE_ADVISOR,
            STATE_FINAL_ANSWER: f"{ADVICE_BUY}这只标的",
            STATE_RETRIEVAL_RESULTS: [_result(ADVICE_BUY)],
        })
        result = verify(state)
        assert any("业务建议" in i for i in result[STATE_VERIFICATION].get("issues", []))

    def test_investment_advice_blocked_for_sales(self):
        state = _state(**{
            STATE_USER_ROLE: ROLE_INSTITUTIONAL_SALES,
            STATE_FINAL_ANSWER: ADVICE_SELL,
            STATE_RETRIEVAL_RESULTS: [_result("卖" + "出建议")],
        })
        result = verify(state)
        assert any("业务建议" in i for i in result[STATE_VERIFICATION].get("issues", []))

    def test_attributed_research_rating_is_not_system_advice(self):
        answer = "东兴证券在2025-10-28发布的报告评级为买入"
        state = _state(**{
            STATE_USER_ROLE: ROLE_ADVISOR,
            STATE_FINAL_ANSWER: answer,
            STATE_RETRIEVAL_RESULTS: [_result(answer)],
        })

        result = verify(state)

        assert result[STATE_VERIFICATION].get("passed") is True

    def test_no_advice_check_for_compliance(self):
        """合规角色不触发业务建议检查"""
        state = _state(**{
            STATE_USER_ROLE: ROLE_COMPLIANCE,
            STATE_FINAL_ANSWER: ADVICE_SELL,
            STATE_RETRIEVAL_RESULTS: [_result("数据")],
        })
        result = verify(state)
        advice_issues = [i for i in result[STATE_VERIFICATION].get("issues", []) if "业务建议" in i]
        assert len(advice_issues) == 0

    def test_compliance_requires_article_number(self):
        state = _state(**{
            STATE_USER_ROLE: ROLE_COMPLIANCE,
            STATE_FINAL_ANSWER: "根据相关规定，股东行为需要披露",
            STATE_RETRIEVAL_RESULTS: [_result("法规内容")],
        })
        result = compliance_check(state)
        compliance = result[STATE_COMPLIANCE]
        assert compliance.get("passed") is False
        assert "citation_precision:missing_article" in compliance.get("flags", [])

    def test_compliance_with_article_number_passes(self):
        state = _state(**{
            STATE_USER_ROLE: ROLE_COMPLIANCE,
            STATE_FINAL_ANSWER: "根据第5条规定，股东行为需要披露",
            STATE_RETRIEVAL_RESULTS: [_result("法规内容")],
        })
        result = compliance_check(state)
        assert result[STATE_COMPLIANCE].get("passed") is True

    def test_confidence_low_on_issues(self):
        state = _state(**{
            STATE_FINAL_ANSWER: "净利润 888 亿元",
            STATE_RETRIEVAL_RESULTS: [],
        })
        result = verify(state)
        assert result[STATE_VERIFICATION].get("confidence") == CONFIDENCE_LOW

    def test_confidence_high_on_clean(self):
        state = _state(**{
            STATE_FINAL_ANSWER: "正常内容",
            STATE_RETRIEVAL_RESULTS: [_result("正常内容")],
        })
        result = verify(state)
        assert result[STATE_VERIFICATION].get("confidence") == CONFIDENCE_HIGH


# ══════════════════════════════════════════════════════════════════════
# compliance_check
# ══════════════════════════════════════════════════════════════════════


class TestComplianceCheck:
    def test_sensitive_keyword_blocked(self):
        state = _state(**{STATE_FINAL_ANSWER: f"这是{SENSITIVE_FIXTURE_TEXT}"})
        result = compliance_check(state)
        compliance = result[STATE_COMPLIANCE]
        assert compliance.get("passed") is False
        assert any("sensitive" in f for f in compliance.get("flags", []))

    def test_investment_advice_flagged(self):
        state = _state(**{STATE_FINAL_ANSWER: f"{ADVICE_BUY}这只标的"})
        result = compliance_check(state)
        assert any("advice" in f for f in result[STATE_COMPLIANCE].get("flags", []))

    def test_risk_disclosure_appended(self):
        state = _state(**{STATE_FINAL_ANSWER: "正常回答"})
        result = compliance_check(state)
        assert "风险提示" in result[STATE_COMPLIANCE].get("risk_disclosure", "")

    def test_clean_answer_passes(self):
        state = _state(**{STATE_FINAL_ANSWER: "该产品风险等级为R3，适合稳健型及以上业务者"})
        result = compliance_check(state)
        assert result[STATE_COMPLIANCE].get("passed") is True

    def test_suitability_warning_for_advisor_with_high_risk(self):
        state = _state(**{
            STATE_USER_ROLE: ROLE_ADVISOR,
            STATE_CLIENT_ID: "fixture_client_id",
            STATE_FINAL_ANSWER: f"该{HIGH_RISK_PRODUCT}预期收益较高",
        })
        result = compliance_check(state)
        assert "适当性" in result[STATE_COMPLIANCE].get("suitability_warning", "")

    def test_no_suitability_without_client_id(self):
        # _state 默认无 STATE_CLIENT_ID，state.get() 返回 None
        state = _state(**{
            STATE_USER_ROLE: ROLE_ADVISOR,
            STATE_FINAL_ANSWER: f"该{HIGH_RISK_PRODUCT}预期收益较高",
        })
        result = compliance_check(state)
        assert result[STATE_COMPLIANCE].get("suitability_warning") == ""


# ══════════════════════════════════════════════════════════════════════
# retrieve
# ══════════════════════════════════════════════════════════════════════


class TestRetrieve:
    def test_uses_hybrid_retriever_and_appends_results(self, monkeypatch):
        captured = {}

        class FakeHybridRetriever:
            def __init__(self, user_role: str, data_permissions: list[str]):
                captured["user_role"] = user_role
                captured["data_permissions"] = data_permissions

            def retrieve(self, plan):
                captured["plan"] = plan
                return [_result("新结果", meta={META_SOURCE: "product_search"})]

        monkeypatch.setattr("src.agents.nodes.HybridRetriever", FakeHybridRetriever)

        state = _state(**{
            STATE_USER_ROLE: ROLE_ADVISOR,
            STATE_RETRIEVAL_PLAN: [
                {
                    PLAN_SOURCE: SOURCE_PRODUCT,
                    PLAN_QUERY: "产品风险",
                    PLAN_TOP_K: 3,
                }
            ],
            STATE_RETRIEVAL_RESULTS: [_result("旧结果")],
            STATE_RETRIEVAL_ATTEMPTS: 1,
            STATE_REWRITTEN_QUERY: "重写查询",
        })

        result = retrieve(state)

        assert captured["user_role"] == ROLE_ADVISOR
        assert captured["plan"] == [
            {
                PLAN_SOURCE: SOURCE_PRODUCT,
                PLAN_QUERY: "产品风险",
                PLAN_TOP_K: 3,
                PLAN_FILTERS: None,
            }
        ]
        assert [r[RR_CONTENT] for r in result[STATE_RETRIEVAL_RESULTS]] == ["旧结果", "新结果"]
        assert result[STATE_RETRIEVAL_ATTEMPTS] == 2

    def test_falls_back_to_rewritten_query_and_default_top_k(self, monkeypatch):
        captured = {}

        class FakeHybridRetriever:
            def __init__(self, user_role: str, data_permissions: list[str]):
                captured["user_role"] = user_role
                captured["data_permissions"] = data_permissions

            def retrieve(self, plan):
                captured["plan"] = plan
                return []

        monkeypatch.setattr("src.agents.nodes.HybridRetriever", FakeHybridRetriever)

        state = _state(**{
            STATE_USER_ROLE: ROLE_OPERATIONS,
            STATE_RETRIEVAL_PLAN: [{PLAN_SOURCE: SOURCE_FAQ}],
            STATE_REWRITTEN_QUERY: "开户流程",
        })

        result = retrieve(state)

        assert captured["plan"] == [
            {
                PLAN_SOURCE: SOURCE_FAQ,
                PLAN_QUERY: "开户流程",
                PLAN_TOP_K: 5,
                PLAN_FILTERS: None,
            }
        ]
        assert result[STATE_RETRIEVAL_ATTEMPTS] == 1
        assert result[STATE_RETRIEVAL_RESULTS] == []


# ══════════════════════════════════════════════════════════════════════
# role-aware tools
# ══════════════════════════════════════════════════════════════════════


class TestRoleAwareTools:
    @pytest.fixture(autouse=True)
    def _clear_reason_model_cache(self):
        _get_bound_reason_model.cache_clear()
        yield
        _get_bound_reason_model.cache_clear()

    def test_technical_role_gets_all_design_allowed_retrieval_tools(self):
        from src.agents.tools import get_tools_for_role

        tool_names = {tool.name for tool in get_tools_for_role(ROLE_TECHNICAL)}

        assert SOURCE_FAQ in tool_names
        assert SOURCE_REPORT in tool_names
        assert "calculator" in tool_names

    def test_report_tool_filters_chunk_not_allowed_for_technical_role(self, monkeypatch):
        from langchain_core.messages import AIMessage
        from langgraph.graph import END, START, StateGraph
        from langgraph.prebuilt import ToolNode

        from src.agents.tools import report_search
        from src.retrieval.base import BaseRetriever

        class RestrictedReportRetriever(BaseRetriever):
            def retrieve(self, query: str, top_k: int = 5, filters=None):
                return [{
                    RR_CONTENT: "confidential report content",
                    RR_METADATA: {
                        META_SOURCE: "restricted-report.pdf",
                        META_PERMISSION_LEVEL: "confidential",
                        META_ALLOWED_ROLES: [ROLE_COMPLIANCE],
                    },
                    RR_SCORE: 0.9,
                }]

        monkeypatch.setattr(
            "src.retrieval.hybrid_retriever.ChromaVectorRetriever",
            RestrictedReportRetriever,
        )
        state = _state(**{
            STATE_USER_ROLE: ROLE_TECHNICAL,
            STATE_DATA_PERMISSIONS: ["public", "internal", "confidential"],
            STATE_MESSAGES: [
                AIMessage(
                    content="",
                    tool_calls=[{
                        "name": report_search.name,
                        "args": {"query": "confidential"},
                        "id": "report-call",
                        "type": "tool_call",
                    }],
                )
            ],
        })

        graph = StateGraph(AssistantState)
        graph.add_node("tools", ToolNode([report_search]))
        graph.add_edge(START, "tools")
        graph.add_edge("tools", END)
        result = graph.compile().invoke(state)
        tool_output = result[STATE_MESSAGES][-1].content
        payload = json.loads(tool_output)

        assert payload[0][RR_DENIED] is True
        assert "confidential report content" not in tool_output

    def test_reason_binds_role_filtered_tools(self, monkeypatch):
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        captured = {}

        class FakeLLM:
            def bind_tools(self, tools):
                captured["tool_names"] = {tool.name for tool in tools}
                return self

            def invoke(self, messages):
                captured["model_messages"] = messages
                return AIMessage(content="ok", id="response")

        monkeypatch.setattr("src.agents.nodes.llm", FakeLLM())

        prior_message = HumanMessage(content="previous", id="prior")
        state = _state(**{
            STATE_USER_ROLE: ROLE_TECHNICAL,
            STATE_ORIGINAL_QUERY: "查询",
            STATE_DEPARTMENT: "tech",
            STATE_MESSAGES: [prior_message],
            STATE_RETRIEVAL_RESULTS: [_result("context", score=0.8)],
        })

        result = build_reason_subgraph().invoke(state)

        assert SOURCE_FAQ in captured["tool_names"]
        assert SOURCE_REPORT in captured["tool_names"]
        model_messages = captured["model_messages"]
        assert isinstance(model_messages[0], SystemMessage)
        assert [message.content for message in model_messages[1:]] == ["查询"]
        assert [message.content for message in result[STATE_MESSAGES]] == [
            "previous",
            "查询",
            "ok",
        ]
        assert result[STATE_FINAL_ANSWER] == "## 结论\n\nok"
        assert result[STATE_REASON_ATTEMPTS] == 1
        assert result[STATE_INTERMEDIATE_STEPS][-1]["step"] == "reason"

    def test_reason_uses_metadata_context_and_excludes_already_retrieved_source(self, monkeypatch):
        from langchain_core.messages import AIMessage, SystemMessage

        captured = {}

        class FakeLLM:
            def bind_tools(self, tools):
                captured["tool_names"] = {tool.name for tool in tools}
                return self

            def invoke(self, messages):
                assert isinstance(messages[0], SystemMessage)
                captured["system_prompt"] = messages[0].content
                return AIMessage(content="评级为买入，机构为诚通证券")

        monkeypatch.setattr("src.agents.nodes.llm", FakeLLM())
        state = _state(**{
            STATE_USER_ROLE: ROLE_ADVISOR,
            STATE_ORIGINAL_QUERY: "贵州茅台的评级和机构",
            STATE_DEPARTMENT: "wealth",
            STATE_MESSAGES: [],
            STATE_RETRIEVAL_PLAN: [{PLAN_SOURCE: SOURCE_REPORT, PLAN_QUERY: "贵州茅台"}],
            STATE_RETRIEVAL_RESULTS: [
                _result(
                    "贵州茅台研报",
                    meta={
                        META_TITLE: "贵州茅台研报",
                        META_SOURCE: "report.pdf",
                        "institution": "诚通证券",
                        "rating": "买入",
                        "date": "2026-05-25",
                    },
                )
            ],
        })

        build_reason_subgraph().invoke(state)

        assert SOURCE_REPORT not in captured["tool_names"]
        assert "机构=诚通证券" in captured["system_prompt"]
        assert "评级=买入" in captured["system_prompt"]
        assert "来源日期=2026-05-25" in captured["system_prompt"]
        assert "只回答用户询问的字段" in captured["system_prompt"]
        assert "不得改写为报告日期或发布日期" in captured["system_prompt"]

    def test_reason_allows_tool_only_answer_when_retrieval_is_empty(self, monkeypatch):
        from langchain_core.messages import AIMessage, ToolMessage

        captured = {}

        class FakeLLM:
            def bind_tools(self, tools):
                captured["bind_count"] = captured.get("bind_count", 0) + 1
                captured["tool_names"] = {tool.name for tool in tools}
                return self

            def invoke(self, messages):
                captured.setdefault("prompts", []).append(messages[0].content)
                if isinstance(messages[-1], ToolMessage):
                    return AIMessage(content=f"计算结果为 {messages[-1].content}")
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "calculator",
                            "args": {"expression": "1+1"},
                            "id": "call-1",
                        }
                    ],
                )

        monkeypatch.setattr("src.agents.nodes.llm", FakeLLM())
        state = _state(**{
            STATE_USER_ROLE: ROLE_ADVISOR,
            STATE_ORIGINAL_QUERY: "计算 1+1",
            STATE_DEPARTMENT: "wealth",
            STATE_RETRIEVAL_RESULTS: [],
        })

        result = build_reason_subgraph().invoke(state)

        assert "没有可用文档检索结果" in captured["prompts"][0]
        assert "纯工具回答不得编造文档引用" in captured["prompts"][0]
        assert result[STATE_FINAL_ANSWER] == "## 结论\n\n计算结果为 2.0000"
        assert result[STATE_TOOL_CALLS] == [
            {
                "tool": "calculator",
                "output": "2.0000",
                "success": True,
            }
        ]
        assert captured["bind_count"] == 1

    def test_reason_rejects_tool_hidden_from_role(self, monkeypatch):
        from langchain_core.messages import AIMessage, ToolMessage

        class FakeLLM:
            def bind_tools(self, tools):
                assert SOURCE_FAQ not in {tool.name for tool in tools}
                return self

            def invoke(self, messages):
                if isinstance(messages[-1], ToolMessage):
                    return AIMessage(content="无法调用未授权工具")
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": SOURCE_FAQ,
                            "args": {"query": "内部问题"},
                            "id": "denied-1",
                        }
                    ],
                )

        monkeypatch.setattr("src.agents.nodes.llm", FakeLLM())
        result = build_reason_subgraph().invoke(_state(**{
            STATE_USER_ROLE: ROLE_ADVISOR,
            STATE_ORIGINAL_QUERY: "内部问题",
        }))

        assert result[STATE_TOOL_CALLS][0]["tool"] == SOURCE_FAQ
        assert result[STATE_TOOL_CALLS][0]["success"] is False
        assert "无权调用" in result[STATE_TOOL_CALLS][0]["output"]

    def test_reason_stops_after_tool_iteration_limit(self, monkeypatch):
        from langchain_core.messages import AIMessage

        class FakeLLM:
            calls = 0

            def bind_tools(self, tools):
                return self

            def invoke(self, messages):
                self.calls += 1
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "calculator",
                            "args": {"expression": "1+1"},
                            "id": f"call-{self.calls}",
                        }
                    ],
                )

        monkeypatch.setattr("src.agents.nodes.llm", FakeLLM())
        result = build_reason_subgraph().invoke(_state(**{
            STATE_USER_ROLE: ROLE_ADVISOR,
            STATE_ORIGINAL_QUERY: "持续计算",
        }))

        assert result[STATE_TOOL_ITERATIONS] == MAX_TOOL_ITERATIONS
        assert len(result[STATE_TOOL_CALLS]) == MAX_TOOL_ITERATIONS + 1
        assert result[STATE_TOOL_CALLS][-1] == {
            "tool": "calculator",
            "output": "工具调用次数达到上限，已停止执行。",
            "success": False,
        }
        assert result[STATE_TOOL_MESSAGE_CURSOR] == len(result[STATE_MESSAGES])
        assert "工具调用次数达到上限" in result[STATE_FINAL_ANSWER]
        assert result[STATE_INTERMEDIATE_STEPS][-1]["success"] is False

    def test_prepare_reason_includes_verification_feedback_on_retry(self):
        result = prepare_reason(_state(**{
            STATE_ORIGINAL_QUERY: "净利润是多少",
            STATE_REASON_ATTEMPTS: 1,
            STATE_VERIFICATION: {"passed": False, "issues": ["数字缺少来源"]},
        }))

        assert "原问题：净利润是多少" in result[STATE_MESSAGES][0].content
        assert "数字缺少来源" in result[STATE_MESSAGES][0].content


class TestStructuredAnswer:
    def test_structures_multi_section_answer_and_preserves_table(self):
        answer = (
            "Answer:\n以下是研报索引样本。\n\n"
            "| stock_code | stock_name |\n"
            "|---|---|\n"
            "| 000001 | 平安银行 |\n\n"
            "Citations:\n无引用\n\nAudit Trail:\n{}"
        )

        result = _structure_answer(answer)

        assert result.startswith("## 结论\n\n以下是研报索引样本。")
        assert "## 关键结果" in result
        assert "| 000001 | 平安银行 |" in result
        assert "Citations:" not in result
        assert "Audit Trail:" not in result

    def test_single_paragraph_does_not_invent_detail_section(self):
        result = _structure_answer("未找到可验证资料。")

        assert result == "## 结论\n\n未找到可验证资料。"
        assert "## 关键结果" not in result

    def test_existing_structured_answer_is_unchanged(self):
        answer = "## 结论\n\n已完成。\n\n## 关键结果\n\n1. 第一项"

        assert _structure_answer(answer) == answer


# ══════════════════════════════════════════════════════════════════════
# compose
# ══════════════════════════════════════════════════════════════════════


class TestCompose:
    def test_uses_verified_citations_without_recreating_them(self):
        citation = {
            "citation_id": "cite_001",
            "doc_title": "2024年报",
            "source": "report.pdf",
            "chunk_id": "chunk_001",
        }
        state = _state(**{
            STATE_FINAL_ANSWER: "示例公司净利润为747亿",
            STATE_CITATIONS: [citation],
            STATE_VERIFICATION: {"passed": True, "confidence": CONFIDENCE_HIGH},
            STATE_COMPLIANCE: {"passed": True},
            STATE_RETRIEVAL_RESULTS: [_result("a"), _result("b"), _result("c")],
        })
        result = compose(state)
        assert len(result[STATE_CITATIONS]) == 1
        assert result[STATE_CITATIONS][0] is citation

    def test_verification_failure_returns_fixed_safe_response(self):
        state = _state(**{
            STATE_FINAL_ANSWER: RESTRICTED_TEXT,
            STATE_CITATIONS: [{"source": "restricted.pdf"}],
            STATE_VERIFICATION: {"passed": False, "confidence": CONFIDENCE_LOW},
            STATE_COMPLIANCE: {"passed": True, "risk_disclosure": ""},
        })
        result = compose(state)
        assert RESTRICTED_TEXT not in result[STATE_FINAL_ANSWER]
        assert result[STATE_CITATIONS] == []

    def test_compliance_failure_returns_fixed_safe_response(self):
        state = _state(**{
            STATE_FINAL_ANSWER: RESTRICTED_TEXT,
            STATE_CITATIONS: [{"source": "restricted.pdf"}],
            STATE_VERIFICATION: {"passed": True, "confidence": CONFIDENCE_HIGH},
            STATE_COMPLIANCE: {"passed": False, "risk_disclosure": ""},
        })
        result = compose(state)
        assert RESTRICTED_TEXT not in result[STATE_FINAL_ANSWER]
        assert result[STATE_CITATIONS] == []

    def test_risk_disclosure_in_answer(self):
        state = _state(**{
            STATE_FINAL_ANSWER: "回答内容",
            STATE_COMPLIANCE: {"risk_disclosure": "【风险提示】测试", "suitability_warning": ""},
        })
        result = compose(state)
        assert "风险提示" in result[STATE_FINAL_ANSWER]

    def test_confidence_low_when_compliance_failed(self):
        state = _state(**{
            STATE_FINAL_ANSWER: "内容",
            STATE_COMPLIANCE: {"passed": False, "risk_disclosure": "", "suitability_warning": ""},
            STATE_VERIFICATION: {"confidence": CONFIDENCE_HIGH},
            STATE_RETRIEVAL_RESULTS: [_result("a"), _result("b"), _result("c")],
        })
        result = compose(state)
        assert result[STATE_CONFIDENCE] == CONFIDENCE_LOW

    def test_confidence_high_all_conditions(self):
        state = _state(**{
            STATE_FINAL_ANSWER: "内容",
            STATE_COMPLIANCE: {"passed": True, "risk_disclosure": "", "suitability_warning": ""},
            STATE_VERIFICATION: {"confidence": CONFIDENCE_HIGH},
            STATE_RETRIEVAL_RESULTS: [_result("a"), _result("b"), _result("c")],
        })
        result = compose(state)
        assert result[STATE_CONFIDENCE] == CONFIDENCE_HIGH

    def test_confidence_medium_with_few_results(self):
        state = _state(**{
            STATE_FINAL_ANSWER: "内容",
            STATE_COMPLIANCE: {"passed": True, "risk_disclosure": "", "suitability_warning": ""},
            STATE_VERIFICATION: {"confidence": CONFIDENCE_HIGH},
            STATE_RETRIEVAL_RESULTS: [_result("a")],
        })
        result = compose(state)
        assert result[STATE_CONFIDENCE] == CONFIDENCE_MEDIUM


# ══════════════════════════════════════════════════════════════════════
# audit_log
# ══════════════════════════════════════════════════════════════════════


class TestAuditLog:
    @pytest.fixture(autouse=True)
    def _use_temp_audit_store(self, monkeypatch, tmp_path):
        self.audit_store = SQLiteAuditStore(tmp_path / "audit.db")
        monkeypatch.setattr("src.agents.nodes._get_audit_store", lambda: self.audit_store)

    def test_creates_audit_entry(self):
        state = _state(**{
            STATE_FINAL_ANSWER: "回答",
            STATE_RETRIEVAL_RESULTS: [_result("内容")],
        })
        result = audit_log(state)
        entry = result[STATE_AUDIT_TRAIL]
        assert AUDIT_REQUEST_ID in entry
        assert AUDIT_TIMESTAMP in entry
        assert entry.get(AUDIT_QUERY, {}).get(AUDIT_QUERY_ORIGINAL) == ""
        assert entry.get(AUDIT_RETRIEVAL, {}).get(AUDIT_RETRIEVAL_TOTAL_CHUNKS) == 1

    def test_audit_includes_all_sections(self):
        state = _state()
        result = audit_log(state)
        entry = result[STATE_AUDIT_TRAIL]
        for section in (
            AUDIT_QUERY,
            AUDIT_RETRIEVAL,
            AUDIT_REASONING,
            AUDIT_VERIFICATION,
            AUDIT_COMPLIANCE,
            AUDIT_RESPONSE,
        ):
            assert section in entry, f"audit entry missing section: {section}"

    def test_audit_sources_are_unique_in_first_seen_order(self):
        state = _state(**{
            STATE_RETRIEVAL_RESULTS: [
                _result("a", meta={META_SOURCE: "faq.html"}),
                _result("b", meta={META_SOURCE: "faq.html"}),
                _result("c", meta={META_SOURCE: "product.html"}),
            ],
        })
        result = audit_log(state)

        audit_trail = result[STATE_AUDIT_TRAIL]
        assert audit_trail.get(AUDIT_RETRIEVAL, {}).get(AUDIT_RETRIEVAL_SOURCES) == [
            "faq.html",
            "product.html",
        ]

    def test_reuses_initialized_audit_metadata_and_calculates_duration(self):
        timestamp = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
        state = _state(**{
            STATE_AUDIT_TRAIL: {
                AUDIT_REQUEST_ID: "request-123",
                AUDIT_STARTED_PERF_COUNTER: time.perf_counter() - 1,
                AUDIT_TIMESTAMP: timestamp,
            },
        })

        result = audit_log(state)

        audit_trail = result[STATE_AUDIT_TRAIL]
        assert audit_trail.get(AUDIT_REQUEST_ID) == "request-123"
        assert audit_trail.get(AUDIT_TIMESTAMP) == timestamp
        assert AUDIT_STARTED_PERF_COUNTER not in audit_trail
        assert audit_trail.get("total_duration_ms", 0) > 0

    def test_uses_real_execution_path_node_duration_and_retrieval_counts(self):
        state = _state(**{
            STATE_RETRIEVAL_RESULTS: [_result("kept")],
            STATE_RETRIEVAL_TOTAL_CHUNKS: 7,
            STATE_RETRIEVAL_FILTERED_CHUNKS: 1,
            STATE_INTERMEDIATE_STEPS: [
                {"step": "retrieve", "duration_ms": 12.0, "success": True},
                {"step": "reason", "duration_ms": 34.5, "success": True},
            ],
        })

        result = audit_log(state)
        audit = result[STATE_AUDIT_TRAIL]

        assert audit[AUDIT_RETRIEVAL]["total_chunks"] == 7
        assert audit[AUDIT_RETRIEVAL]["filtered_chunks"] == 1
        assert audit[AUDIT_REASONING]["duration_ms"] == 34.5
        assert audit[AUDIT_REASONING]["iterations"] == 1
        assert audit[AUDIT_REASONING]["execution_path"] == ["retrieve", "reason", "audit_log"]

    def test_traced_node_records_name_duration_and_success(self):
        wrapped = _traced_node("sample", lambda state: {STATE_FINAL_ANSWER: "ok"})

        result = wrapped(_state(**{
            STATE_MESSAGES: [],
            STATE_INTERMEDIATE_STEPS: [
                {"step": "previous", "duration_ms": 1.0, "success": True}
            ],
        }))

        step = result[STATE_INTERMEDIATE_STEPS][-1]
        assert STATE_MESSAGES not in result
        assert result[STATE_INTERMEDIATE_STEPS][0]["step"] == "previous"
        assert step["step"] == "sample"
        assert step["duration_ms"] >= 0
        assert step["success"] is True

    def test_persists_audit_entry_for_lookup(self):
        timestamp = datetime.now(timezone.utc).isoformat()
        state = _state(**{
            STATE_AUDIT_TRAIL: {
                AUDIT_REQUEST_ID: "request-lookup",
                AUDIT_TIMESTAMP: timestamp,
            },
            STATE_FINAL_ANSWER: "回答",
            STATE_CONFIDENCE: CONFIDENCE_MEDIUM,
        })

        result = audit_log(state)

        persisted = self.audit_store.get_by_request_id("request-lookup")
        assert persisted is not None
        assert persisted.get(AUDIT_REQUEST_ID) == "request-lookup"
        assert persisted.get(AUDIT_TIMESTAMP) == timestamp
        audit_trail = result[STATE_AUDIT_TRAIL]
        assert persisted.get(AUDIT_RESPONSE, {}).get(AUDIT_RESPONSE_CONFIDENCE) == audit_trail.get(
            AUDIT_RESPONSE, {}
        ).get(AUDIT_RESPONSE_CONFIDENCE)


# ══════════════════════════════════════════════════════════════════════
# 条件路由函数
# ══════════════════════════════════════════════════════════════════════


class TestShouldRetryRetrieval:
    def test_empty_results(self):
        assert should_retry_retrieval(_state()) == "retrieve"

    def test_low_score(self):
        state = _state(**{STATE_RETRIEVAL_RESULTS: [_result("x", score=0.3)]})
        assert should_retry_retrieval(state) == "retrieve"

    def test_enough_medium_score_results_continue_without_replanning(self):
        state = _state(**{
            STATE_RETRIEVAL_RESULTS: [
                _result("a", score=0.74),
                _result("b", score=0.70),
                _result("c", score=0.65),
            ],
            STATE_RETRIEVAL_ATTEMPTS: 1,
        })

        assert should_retry_retrieval(state) == "continue"

    def test_high_score(self):
        state = _state(**{
            STATE_RETRIEVAL_RESULTS: [
                _result("x", score=0.9),
                _result("y", score=0.8),
                _result("z", score=0.76),
            ]
        })
        assert should_retry_retrieval(state) == "continue"

    def test_single_high_score_still_retries(self):
        state = _state(**{STATE_RETRIEVAL_RESULTS: [_result("x", score=0.9)]})
        assert should_retry_retrieval(state) == "retrieve"


class TestShouldReasonAgain:
    def test_not_passed(self):
        state = _state(**{STATE_VERIFICATION: {"passed": False}, STATE_REASON_ATTEMPTS: 1})
        assert should_reason_again(state) == "retry"

    def test_passed(self):
        state = _state(**{STATE_VERIFICATION: {"passed": True}})
        assert should_reason_again(state) == "continue"

    def test_not_passed_stops_after_max_reason_attempts(self):
        state = _state(**{STATE_VERIFICATION: {"passed": False}, STATE_REASON_ATTEMPTS: 2})
        assert should_reason_again(state) == "continue"


class TestIsCompliant:
    def test_passed(self):
        state = _state(**{STATE_COMPLIANCE: {"passed": True}})
        assert is_compliant(state) == "pass"

    def test_blocked(self):
        state = _state(**{STATE_COMPLIANCE: {"passed": False}})
        assert is_compliant(state) == "block"


# ══════════════════════════════════════════════════════════════════════
# build_agent_graph
# ══════════════════════════════════════════════════════════════════════


class TestBuildAgentGraph:
    def test_returns_state_graph(self):
        from langgraph.graph import StateGraph

        graph = build_agent_graph()
        assert isinstance(graph, StateGraph)

    def test_compile_returns_compiled_graph(self, monkeypatch):
        """编译后返回 CompiledStateGraph"""
        monkeypatch.setattr(
            "langgraph.graph.StateGraph.compile",
            lambda self, **_: MagicMock(name="compiled"),
        )
        graph = build_agent_graph()
        compiled = graph.compile()
        assert compiled is not None
