"""Agent Graph 节点和路由函数单元测试"""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import MagicMock

from src.agents.state import AssistantState
from src.agents.graph import (
    build_agent_graph,
    is_compliant,
    should_reason_again,
    should_retry_retrieval,
)
from src.agents.nodes import (
    audit_log,
    compliance_check,
    compose,
    grade_and_filter,
    retrieve,
    verify,
)
from src.schemas.constants import (
    CONFIDENCE_HIGH,
    CONFIDENCE_LOW,
    CONFIDENCE_MEDIUM,
    META_CHUNK_ID,
    META_PERMISSION_LEVEL,
    META_SOURCE,
    META_TITLE,
    PERMISSION_PUBLIC,
    ROLE_ADVISOR,
    ROLE_COMPLIANCE,
    ROLE_INSTITUTIONAL_SALES,
    ROLE_OPERATIONS,
    RR_CONTENT,
    RR_METADATA,
    RR_SCORE,
    STATE_CITATIONS,
    STATE_CLIENT_ID,
    STATE_COMPLIANCE,
    STATE_CONFIDENCE,
    STATE_DATA_PERMISSIONS,
    STATE_FINAL_ANSWER,
    STATE_RETRIEVAL_ATTEMPTS,
    STATE_RETRIEVAL_PLAN,
    STATE_RETRIEVAL_RESULTS,
    STATE_REWRITTEN_QUERY,
    STATE_USER_ROLE,
    STATE_VERIFICATION,
)


def _result(content: str, score: float = 0.9, meta: dict[str, Any] | None = None) -> dict[str, Any]:
    """快捷构造检索结果 dict"""
    return {
        RR_CONTENT: content,
        RR_SCORE: score,
        RR_METADATA: meta or {META_TITLE: "测试文档", META_SOURCE: "test.pdf"},
    }


def _state(**overrides: Any) -> AssistantState:
    """快捷构造最小 AssistantState"""
    return cast(AssistantState, {
        STATE_RETRIEVAL_RESULTS: [],
        STATE_FINAL_ANSWER: "",
        STATE_USER_ROLE: ROLE_OPERATIONS,
        STATE_VERIFICATION: {},
        STATE_COMPLIANCE: {},
        **overrides,
    })


# ══════════════════════════════════════════════════════════════════════
# grade_and_filter
# ══════════════════════════════════════════════════════════════════════


class TestGradeAndFilter:
    def test_empty_results(self):
        state = _state()
        result = grade_and_filter(state)
        assert result[STATE_RETRIEVAL_RESULTS] == []

    def test_sorts_by_score_desc(self):
        r1 = _result("a", score=0.5)
        r2 = _result("b", score=0.9)
        r3 = _result("c", score=0.3)
        state = _state(**{STATE_RETRIEVAL_RESULTS: [r1, r2, r3]})
        result = grade_and_filter(state)
        scores = [r[RR_SCORE] for r in result[STATE_RETRIEVAL_RESULTS]]
        assert scores == [0.9, 0.5, 0.3]

    def test_truncates_to_grade_top_k(self):
        results = [_result(f"doc{i}", score=0.9 - i * 0.01) for i in range(20)]
        state = _state(**{STATE_RETRIEVAL_RESULTS: results})
        result = grade_and_filter(state)
        assert len(result[STATE_RETRIEVAL_RESULTS]) <= 10

    def test_preserves_other_state_fields(self):
        state = _state(**{STATE_FINAL_ANSWER: "unchanged"})
        result = grade_and_filter(state)
        assert result[STATE_FINAL_ANSWER] == "unchanged"


# ══════════════════════════════════════════════════════════════════════
# verify
# ══════════════════════════════════════════════════════════════════════


class TestVerify:
    def test_numbers_without_results(self):
        state = _state(**{
            STATE_FINAL_ANSWER: "净利润 747 亿元",
            STATE_RETRIEVAL_RESULTS: [],
        })
        result = verify(state)
        assert result[STATE_VERIFICATION]["passed"] is False
        assert any("数字" in i for i in result[STATE_VERIFICATION]["issues"])

    def test_numbers_with_results_passes(self):
        state = _state(**{
            STATE_FINAL_ANSWER: "净利润 747 亿元",
            STATE_RETRIEVAL_RESULTS: [_result("净利润 747 亿元")],
        })
        result = verify(state)
        assert result[STATE_VERIFICATION]["passed"] is True

    def test_permission_denied(self):
        state = _state(**{
            STATE_FINAL_ANSWER: "正常回答",
            STATE_DATA_PERMISSIONS: [PERMISSION_PUBLIC],
            STATE_RETRIEVAL_RESULTS: [
                _result(
                    "机密内容",
                    meta={META_PERMISSION_LEVEL: "confidential", META_SOURCE: "secret.pdf"},
                ),
            ],
        })
        result = verify(state)
        assert any("权限不足" in i for i in result[STATE_VERIFICATION]["issues"])

    def test_investment_advice_blocked_for_advisor(self):
        state = _state(**{
            STATE_USER_ROLE: ROLE_ADVISOR,
            STATE_FINAL_ANSWER: "推荐买入这只股票",
            STATE_RETRIEVAL_RESULTS: [_result("推荐买入")],
        })
        result = verify(state)
        assert any("投资建议" in i for i in result[STATE_VERIFICATION]["issues"])

    def test_investment_advice_blocked_for_sales(self):
        state = _state(**{
            STATE_USER_ROLE: ROLE_INSTITUTIONAL_SALES,
            STATE_FINAL_ANSWER: "建议卖出",
            STATE_RETRIEVAL_RESULTS: [_result("卖出建议")],
        })
        result = verify(state)
        assert any("投资建议" in i for i in result[STATE_VERIFICATION]["issues"])

    def test_no_advice_check_for_compliance(self):
        """合规角色不触发投资建议检查"""
        state = _state(**{
            STATE_USER_ROLE: ROLE_COMPLIANCE,
            STATE_FINAL_ANSWER: "建议卖出",
            STATE_RETRIEVAL_RESULTS: [_result("数据")],
        })
        result = verify(state)
        advice_issues = [i for i in result[STATE_VERIFICATION]["issues"] if "投资建议" in i]
        assert len(advice_issues) == 0

    def test_compliance_requires_article_number(self):
        state = _state(**{
            STATE_USER_ROLE: ROLE_COMPLIANCE,
            STATE_FINAL_ANSWER: "根据相关规定，大股东减持需要披露",
            STATE_RETRIEVAL_RESULTS: [_result("法规内容")],
        })
        result = verify(state)
        assert any("条款" in i for i in result[STATE_VERIFICATION]["issues"])

    def test_compliance_with_article_number_passes(self):
        state = _state(**{
            STATE_USER_ROLE: ROLE_COMPLIANCE,
            STATE_FINAL_ANSWER: "根据第5条规定，大股东减持需要披露",
            STATE_RETRIEVAL_RESULTS: [_result("法规内容")],
        })
        result = verify(state)
        assert result[STATE_VERIFICATION]["passed"] is True

    def test_confidence_low_on_issues(self):
        state = _state(**{
            STATE_FINAL_ANSWER: "净利润 888 亿元",
            STATE_RETRIEVAL_RESULTS: [],
        })
        result = verify(state)
        assert result[STATE_VERIFICATION]["confidence"] == CONFIDENCE_LOW

    def test_confidence_high_on_clean(self):
        state = _state(**{
            STATE_FINAL_ANSWER: "正常内容",
            STATE_RETRIEVAL_RESULTS: [_result("正常内容")],
        })
        result = verify(state)
        assert result[STATE_VERIFICATION]["confidence"] == CONFIDENCE_HIGH


# ══════════════════════════════════════════════════════════════════════
# compliance_check
# ══════════════════════════════════════════════════════════════════════


class TestComplianceCheck:
    def test_sensitive_keyword_blocked(self):
        state = _state(**{STATE_FINAL_ANSWER: "这是内幕信息"})
        result = compliance_check(state)
        assert result[STATE_COMPLIANCE]["passed"] is False
        assert any("sensitive" in f for f in result[STATE_COMPLIANCE]["flags"])

    def test_investment_advice_flagged(self):
        state = _state(**{STATE_FINAL_ANSWER: "推荐买入这只股票"})
        result = compliance_check(state)
        assert any("advice" in f for f in result[STATE_COMPLIANCE]["flags"])

    def test_risk_disclosure_appended(self):
        state = _state(**{STATE_FINAL_ANSWER: "正常回答"})
        result = compliance_check(state)
        assert "风险提示" in result[STATE_COMPLIANCE]["risk_disclosure"]

    def test_clean_answer_passes(self):
        state = _state(**{STATE_FINAL_ANSWER: "该产品风险等级为R3，适合稳健型及以上投资者"})
        result = compliance_check(state)
        assert result[STATE_COMPLIANCE]["passed"] is True

    def test_suitability_warning_for_advisor_with_high_risk(self):
        state = _state(**{
            STATE_USER_ROLE: ROLE_ADVISOR,
            STATE_CLIENT_ID: "client_001",
            STATE_FINAL_ANSWER: "该私募基金预期收益较高",
        })
        result = compliance_check(state)
        assert "适当性" in result[STATE_COMPLIANCE]["suitability_warning"]

    def test_no_suitability_without_client_id(self):
        # _state 默认无 STATE_CLIENT_ID，state.get() 返回 None
        state = _state(**{
            STATE_USER_ROLE: ROLE_ADVISOR,
            STATE_FINAL_ANSWER: "该私募基金预期收益较高",
        })
        result = compliance_check(state)
        assert result[STATE_COMPLIANCE]["suitability_warning"] == ""


# ══════════════════════════════════════════════════════════════════════
# retrieve
# ══════════════════════════════════════════════════════════════════════


class TestRetrieve:
    def test_uses_hybrid_retriever_and_appends_results(self, monkeypatch):
        captured = {}

        class FakeHybridRetriever:
            def __init__(self, user_role: str):
                captured["user_role"] = user_role

            def retrieve(self, plan):
                captured["plan"] = plan
                return [_result("新结果", meta={META_SOURCE: "product_search"})]

        monkeypatch.setattr("src.agents.nodes.HybridRetriever", FakeHybridRetriever)

        state = _state(**{
            STATE_USER_ROLE: ROLE_ADVISOR,
            STATE_RETRIEVAL_PLAN: [{"source": "product_search", "query": "产品风险", "top_k": 3}],
            STATE_RETRIEVAL_RESULTS: [_result("旧结果")],
            STATE_RETRIEVAL_ATTEMPTS: 1,
            "rewritten_query": "重写查询",
        })

        result = retrieve(state)

        assert captured["user_role"] == ROLE_ADVISOR
        assert captured["plan"] == [{"source": "product_search", "query": "产品风险", "top_k": 3, "filters": None}]
        assert [r[RR_CONTENT] for r in result[STATE_RETRIEVAL_RESULTS]] == ["旧结果", "新结果"]
        assert result[STATE_RETRIEVAL_ATTEMPTS] == 2

    def test_falls_back_to_rewritten_query_and_default_top_k(self, monkeypatch):
        captured = {}

        class FakeHybridRetriever:
            def __init__(self, user_role: str):
                captured["user_role"] = user_role

            def retrieve(self, plan):
                captured["plan"] = plan
                return []

        monkeypatch.setattr("src.agents.nodes.HybridRetriever", FakeHybridRetriever)

        state = _state(**{
            STATE_USER_ROLE: ROLE_OPERATIONS,
            STATE_RETRIEVAL_PLAN: [{"source": "faq_search"}],
            STATE_REWRITTEN_QUERY: "开户流程",
        })

        result = retrieve(state)

        assert captured["plan"] == [{"source": "faq_search", "query": "开户流程", "top_k": 5, "filters": None}]
        assert result[STATE_RETRIEVAL_ATTEMPTS] == 1
        assert result[STATE_RETRIEVAL_RESULTS] == []


# ══════════════════════════════════════════════════════════════════════
# compose
# ══════════════════════════════════════════════════════════════════════


class TestCompose:
    def test_citations_built_from_results(self):
        state = _state(**{
            STATE_FINAL_ANSWER: "茅台净利润为747亿",
            STATE_RETRIEVAL_RESULTS: [
                _result(
                    "茅台净利润747亿",
                    meta={
                        META_TITLE: "2024年报",
                        META_SOURCE: "report.pdf",
                        META_CHUNK_ID: "chunk_001",
                    },
                ),
            ],
        })
        result = compose(state)
        assert len(result[STATE_CITATIONS]) == 1
        assert result[STATE_CITATIONS][0]["doc_title"] == "2024年报"
        assert result[STATE_CITATIONS][0]["source"] == "report.pdf"

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
    def test_creates_audit_entry(self):
        state = _state(**{
            STATE_FINAL_ANSWER: "回答",
            STATE_RETRIEVAL_RESULTS: [_result("内容")],
        })
        result = audit_log(state)
        entry = result["audit_trail"]
        assert "request_id" in entry
        assert "timestamp" in entry
        assert entry["query"]["original"] == ""
        assert entry["retrieval"]["total_chunks"] == 1

    def test_audit_includes_all_sections(self):
        state = _state()
        result = audit_log(state)
        entry = result["audit_trail"]
        for section in (
            "query",
            "retrieval",
            "reasoning",
            "verification",
            "compliance",
            "response",
        ):
            assert section in entry, f"audit entry missing section: {section}"


# ══════════════════════════════════════════════════════════════════════
# 条件路由函数
# ══════════════════════════════════════════════════════════════════════


class TestShouldRetryRetrieval:
    def test_empty_results(self):
        assert should_retry_retrieval(_state()) == "retrieve"

    def test_low_score(self):
        state = _state(**{STATE_RETRIEVAL_RESULTS: [_result("x", score=0.3)]})
        assert should_retry_retrieval(state) == "retrieve"

    def test_high_score(self):
        state = _state(**{STATE_RETRIEVAL_RESULTS: [_result("x", score=0.9)]})
        assert should_retry_retrieval(state) == "continue"


class TestShouldReasonAgain:
    def test_not_passed(self):
        state = _state(**{STATE_VERIFICATION: {"passed": False}})
        assert should_reason_again(state) == "retry"

    def test_passed(self):
        state = _state(**{STATE_VERIFICATION: {"passed": True}})
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
