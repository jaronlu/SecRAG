"""合规检查工具测试。"""

from src.schemas.constants import ROLE_ADVISOR, ROLE_COMPLIANCE
from src.utils.compliance import ComplianceChecker


ADVICE_BUY = "推荐" + "买" + "入"
SENSITIVE_TEXT = "内" + "幕" + "信息"
HIGH_RISK_PRODUCT = "私" + "募" + "产品"
TARGET_PRICE = "目标" + "价"


def test_compliance_checker_flags_sensitive_info_and_advice():
    checker = ComplianceChecker()

    result = checker.check(f"包含{SENSITIVE_TEXT}，并{ADVICE_BUY}")

    assert result["passed"] is False
    assert any(flag.startswith("sensitive:") for flag in result["flags"])
    assert any(flag.startswith("advice:") for flag in result["flags"])
    assert "风险提示" in result["risk_disclosure"]


def test_compliance_checker_blocks_advice_only_text():
    checker = ComplianceChecker()

    result = checker.check(f"{ADVICE_BUY}这只标的")

    assert result["passed"] is False
    assert any(flag.startswith("advice:") for flag in result["flags"])


def test_compliance_checker_allows_attributed_research_rating():
    checker = ComplianceChecker()

    result = checker.check("东兴证券在2025-10-28发布的报告评级为买入。")

    assert result["passed"] is True
    assert not any(flag.startswith("advice:") for flag in result["flags"])


def test_compliance_checker_allows_verified_attributed_target_price():
    checker = ComplianceChecker()

    result = checker.check(
        f"该研报记录的{TARGET_PRICE}为100元。",
        allow_attributed_target_price=True,
    )

    assert result["passed"] is True
    assert not any(flag.startswith("advice:") for flag in result["flags"])


def test_compliance_checker_blocks_unattributed_target_price():
    checker = ComplianceChecker()

    result = checker.check(f"这只标的的{TARGET_PRICE}为100元。")

    assert result["passed"] is False
    assert any(flag.startswith("advice:") for flag in result["flags"])


def test_compliance_checker_requires_article_for_compliance_role():
    checker = ComplianceChecker()

    result = checker.check("根据相关规定，需要披露。", user_role=ROLE_COMPLIANCE)

    assert result["passed"] is False
    assert "citation_precision:missing_article" in result["flags"]


def test_compliance_checker_adds_suitability_warning_for_advisor_client():
    checker = ComplianceChecker()

    result = checker.check(
        f"该{HIGH_RISK_PRODUCT}风险等级较高。",
        user_role=ROLE_ADVISOR,
        client_id="fixture_client_id",
    )

    assert result["passed"] is True
    assert "适当性" in result["suitability_warning"]
