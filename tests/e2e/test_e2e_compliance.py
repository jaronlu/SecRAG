"""全链路测试案例：环节 E 金融合规与安全（TC-024~TC-029）。"""

from __future__ import annotations

import pytest

from src.schemas.constants import (
    CONFIDENCE_LOW,
    ROLE_ADVISOR,
    ROLE_COMPLIANCE,
    STATE_CITATIONS,
    STATE_COMPLIANCE,
    STATE_CONFIDENCE,
    STATE_FINAL_ANSWER,
    STATE_REASON_ATTEMPTS,
    STATE_VERIFICATION,
)
from src.utils.compliance import ComplianceChecker, matches_investment_advice


# ══════════════════════════════════════════════════════════════════════
# TC-024 投资建议合规拦截
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("text,expected_flag", [
    ("建议买入该产品", "advice:建议买入"),
    ("推荐你买入这只基金", "advice:推荐买入"),
    ("可以考虑买入一些", "advice:建议买入"),
    ("建议卖出部分仓位", "advice:建议卖出"),
    ("我们给出目标价 12.5 元", "advice:目标价"),
    ("推 荐 你 买 入（空格绕过）", "advice:推荐买入"),
])
def test_tc024_investment_advice_patterns_blocked(text, expected_flag):
    """TC-024：直接表述、改写绕过、空格分隔的建议全部被模糊匹配拦截。"""
    result = ComplianceChecker().check(text, user_role=ROLE_ADVISOR)
    assert result["passed"] is False
    assert expected_flag in result["flags"]


# DEF-001（TC-024）：TP+数字的目标价表述漏检。
# matches_investment_advice 先去除全部空白（防空格绕过），"TP 12.5" 归一化为
# "TP12.5" 后 _TARGET_PRICE_REGEXES 的 \bTP\b 边界失效——空格防护与 TP 正则不兼容。
# strict=True：若将来修复该缺陷，此测试将 XPASS 并提醒移除标记。
@pytest.mark.xfail(
    reason="DEF-001：TP+数字目标价写法（TP 12.5/建议TP 15元/TP12.5）漏检",
    strict=True,
)
@pytest.mark.parametrize("text", ["TP 12.5 元", "建议TP 15元", "TP12.5"])
def test_tc024_tp_with_number_should_be_blocked(text):
    assert "advice:目标价" in ComplianceChecker().check(text, user_role=ROLE_ADVISOR)["flags"]


def test_tc024_compose_blocks_non_compliant_answer():
    """TC-024：验证通过但合规不通过时，compose 停止输出并清空引用。"""
    from src.agents.nodes import compose

    state = {
        STATE_FINAL_ANSWER: "该产品可以考虑买入。",
        STATE_CITATIONS: [{"source": "a.pdf", "chunk_id": "c1"}],
        STATE_VERIFICATION: {"passed": True, "issues": [], "confidence": "high"},
        STATE_COMPLIANCE: {
            "passed": False,
            "flags": ["advice:建议买入"],
            "risk_disclosure": "",
            "suitability_warning": "",
        },
        "retrieval_results": [],
        "reranker_status": "unavailable",
    }

    update = compose(state)

    answer = update[STATE_FINAL_ANSWER]
    assert "当前请求或生成内容未通过合规检查" in answer
    assert "买入" not in answer.split("【风险提示】")[0], "违规内容不得保留"
    assert update[STATE_CITATIONS] == []
    assert update[STATE_CONFIDENCE] == CONFIDENCE_LOW


def test_tc024_normal_discussion_is_not_blocked():
    """TC-024：中性表述（评级引用、风险描述）不触发建议拦截。"""
    text = "根据2024年年度报告，该基金风险等级为R1，主要投资于货币市场工具[来源1]。"
    result = ComplianceChecker().check(text, user_role=ROLE_ADVISOR)
    assert result["passed"] is True
    assert matches_investment_advice(text) == []


# ══════════════════════════════════════════════════════════════════════
# TC-025 敏感词拦截
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("text,flag", [
    ("这里掌握大量内幕信息", "sensitive:内幕信息"),
    ("以下是未公开的业绩数据", "sensitive:未公开"),
    ("我们预测业绩预测将超预期", "sensitive:业绩预测"),
])
def test_tc025_sensitive_keywords_flagged(text, flag):
    """TC-025：敏感词命中 → flags 带 sensitive: 前缀且不通过。"""
    result = ComplianceChecker().check(text, user_role=ROLE_ADVISOR)
    assert flag in result["flags"]
    assert result["passed"] is False


# ══════════════════════════════════════════════════════════════════════
# TC-026 合规角色条款引用精度
# ══════════════════════════════════════════════════════════════════════


def test_tc026_compliance_role_requires_article_reference():
    """TC-026：合规角色回答无条款引用 → citation_precision 缺失；含条款则通过。"""
    checker = ComplianceChecker()
    plain = "该基金投资范围包括货币市场工具。"
    result = checker.check(plain, user_role=ROLE_COMPLIANCE)
    assert "citation_precision:missing_article" in result["flags"]
    assert result["passed"] is False

    cited = "依据《公开募集开放式证券投资基金流动性风险管理规定》第五条，基金应保持流动性。"
    result = checker.check(cited, user_role=ROLE_COMPLIANCE)
    assert "citation_precision:missing_article" not in result["flags"]
    assert result["passed"] is True

    # 其他角色不要求条款引用
    result = checker.check(plain, user_role=ROLE_ADVISOR)
    assert result["passed"] is True


# ══════════════════════════════════════════════════════════════════════
# TC-027 适当性警告
# ══════════════════════════════════════════════════════════════════════


def test_tc027_suitability_warning_for_high_risk_product():
    """TC-027：advisor + 客户号 + 高风险产品 → 适当性提示，合规本身仍通过。"""
    checker = ComplianceChecker()
    result = checker.check(
        "这款私募产品采用摊余成本法估值。", user_role=ROLE_ADVISOR, client_id="C001"
    )
    assert any(f.startswith("suitability:") for f in result["flags"])
    assert result["suitability_warning"].startswith("\n\n【适当性提示】")
    assert result["passed"] is True, "适当性是提示不是拦截"

    # 无客户号时不提示
    result = checker.check(
        "这款私募产品采用摊余成本法估值。", user_role=ROLE_ADVISOR, client_id=None
    )
    assert result["suitability_warning"] == ""
    assert result["passed"] is True


def test_tc027_suitability_appended_to_final_answer():
    """TC-027：适当性警告附加到 compose 的最终答案尾部。"""
    from src.agents.nodes import compose

    warning = "\n\n【适当性提示】该产品风险等级较高，请确认客户风险承受能力是否匹配。"
    state = {
        STATE_FINAL_ANSWER: "## 结论\n\n该产品为私募产品。",
        STATE_CITATIONS: [{"source": "a.pdf"}],
        STATE_VERIFICATION: {"passed": True, "issues": [], "confidence": "high"},
        STATE_COMPLIANCE: {
            "passed": True,
            "flags": ["suitability:私募产品"],
            "risk_disclosure": "",
            "suitability_warning": warning,
        },
        "retrieval_results": [],
        "reranker_status": "unavailable",
    }

    update = compose(state)
    assert update[STATE_FINAL_ANSWER].endswith(warning)
    assert update[STATE_CITATIONS] == [{"source": "a.pdf"}], "合规通过时引用保留"
