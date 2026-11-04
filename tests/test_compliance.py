"""合规检查工具测试。"""

import pytest

from src.schemas.constants import ROLE_ADVISOR, ROLE_COMPLIANCE
from src.utils.compliance import ComplianceChecker


ADVICE_BUY = "推荐" + "买" + "入"
SENSITIVE_TEXT = "内" + "幕" + "信息"
HIGH_RISK_PRODUCT = "私" + "募" + "产品"
TARGET_PRICE = "目标" + "价"


def test_compliance_checker_flags_sensitive_info_and_advice():
    checker = ComplianceChecker()

    result = checker.check(f"包含{SENSITIVE_TEXT}，并{ADVICE_BUY}")

    assert result.get("passed") is False
    assert any(flag.startswith("sensitive:") for flag in (result.get("flags") or []) or [])
    assert any(flag.startswith("advice:") for flag in (result.get("flags") or []) or [])
    assert "风险提示" in (result.get("risk_disclosure") or "")


def test_compliance_checker_blocks_advice_only_text():
    checker = ComplianceChecker()

    result = checker.check(f"{ADVICE_BUY}这只标的")

    assert result.get("passed") is False
    assert any(flag.startswith("advice:") for flag in (result.get("flags") or []) or [])


def test_compliance_checker_allows_attributed_research_rating():
    checker = ComplianceChecker()

    result = checker.check("东兴证券在2025-10-28发布的报告评级为买入。")

    assert result.get("passed") is True
    assert not any(flag.startswith("advice:") for flag in (result.get("flags") or []) or [])


def test_compliance_checker_allows_verified_attributed_target_price():
    checker = ComplianceChecker()

    result = checker.check(
        f"该研报记录的{TARGET_PRICE}为100元。",
        allow_attributed_target_price=True,
    )

    assert result.get("passed") is True
    assert not any(flag.startswith("advice:") for flag in (result.get("flags") or []) or [])


def test_compliance_checker_blocks_unattributed_target_price():
    checker = ComplianceChecker()

    result = checker.check(f"这只标的的{TARGET_PRICE}为100元。")

    assert result.get("passed") is False
    assert any(flag.startswith("advice:") for flag in (result.get("flags") or []) or [])


def test_compliance_checker_requires_article_for_compliance_role():
    checker = ComplianceChecker()

    result = checker.check("根据相关规定，需要披露。", user_role=ROLE_COMPLIANCE)

    assert result.get("passed") is False
    assert "citation_precision:missing_article" in (result.get("flags") or [])


def test_compliance_checker_adds_suitability_warning_for_advisor_client():
    checker = ComplianceChecker()

    result = checker.check(
        f"该{HIGH_RISK_PRODUCT}风险等级较高。",
        user_role=ROLE_ADVISOR,
        client_id="fixture_client_id",
    )

    assert result.get("passed") is True
    assert "适当性" in (result.get("suitability_warning") or "")


# ══════════════════════════════════════════════════════════════════════
# 改写绕过测试：验证正则模糊匹配能拦截关键词中间插入字符的情况
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("text", [
    "推荐你买入这只股票",
    "推荐可以买入",
    "建议你买入",
    "建议可以买入",
    "可以考虑买入",
    "建议你卖出",
    "可以考虑卖出",
    "建议你增持",
    "建议你减持",
    "目标价格为100元",
])
def test_compliance_blocks_paraphrased_advice(text):
    """关键词中间插入字符的改写应被正则拦截。"""
    checker = ComplianceChecker()
    result = checker.check(text)
    assert result.get("passed") is False, f"应拦截: {text}"
    assert any(flag.startswith("advice:") for flag in (result.get("flags") or []) or [])


def test_compliance_allows_neutral_buy_mention():
    """中性提及'买入'但非建议句式不应被拦截。"""
    checker = ComplianceChecker()
    result = checker.check("该报告中提到了买入和卖出两种操作策略。")
    assert result.get("passed") is True


# ══════════════════════════════════════════════════════════════════════
# ISSUE-31：投资建议索取意图的确定性识别（query 层）
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("query", [
    "这款产品现在适合买入吗？给我一个操作建议。",
    "XX货币市场基金适合买入吗",
    "这只股票值得入手吗",
    "该买吗？现在是不是好时机",
    "能买不？帮我看看",
    "现在加仓合适吗",
    "什么时候建仓比较好",
    "给我一个操作建议",
    "请提供投资建议",
])
def test_advice_seeking_intent_detected(query):
    from src.utils.compliance import matches_advice_seeking_intent

    assert matches_advice_seeking_intent(query), f"应识别索取意图: {query}"


@pytest.mark.parametrize("query", [
    "什么是买入评级",  # 知识咨询
    "投资建议有哪些类型",  # 知识咨询（无索取动词）
    "报告中提到买入和卖出两种评级",  # 引用性提及
    "货币基金的风险等级是多少",  # 普通产品咨询
    "买入基金的费率怎么算",  # 规则咨询
])
def test_advice_seeking_intent_not_triggered_by_knowledge_queries(query):
    from src.utils.compliance import matches_advice_seeking_intent

    assert not matches_advice_seeking_intent(query), f"知识咨询不得误伤: {query}"


def test_advice_intent_flags_are_prefixed_for_audit():
    from src.utils.compliance import advice_intent_flags

    flags = advice_intent_flags("这款产品适合买入吗？")
    assert flags
    assert all(flag.startswith("advice_intent:") for flag in flags)
