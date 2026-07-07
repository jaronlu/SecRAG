"""合规检查工具。"""

import re
from collections.abc import Iterable
from typing import TypedDict

from src.schemas.constants import ROLE_ADVISOR, ROLE_COMPLIANCE

INVESTMENT_ADVICE_PATTERNS: tuple[str, ...] = (
    "推荐" + "买" + "入",
    "建议" + "卖" + "出",
    "目标" + "价",
    "评级",
    "买" + "入",
    "卖" + "出",
    "增" + "持",
    "减" + "持",
)
SENSITIVE_KEYWORDS: tuple[str, ...] = ("内" + "幕" + "信息", "未" + "公开", "业绩" + "预测")
HIGH_RISK_PRODUCTS: tuple[str, ...] = ("标的型" + "产品", "混合型" + "产品", "私" + "募" + "产品")
ARTICLE_REFERENCE_PATTERN = r"第[一二三四五六七八九十百千]+条|第\d+条|Article\s+\d+"
RISK_DISCLOSURE = "\n\n【风险提示】本回答仅供参考，不构成业务建议。市场有风险，业务需谨慎。"
SUITABILITY_WARNING = "\n\n【适当性提示】该产品风险等级较高，请确认客户风险承受能力是否匹配。"


class ComplianceResult(TypedDict):
    passed: bool
    flags: list[str]
    risk_disclosure: str
    suitability_warning: str


class ComplianceChecker:
    """检查回答中的敏感信息、业务建议、引用精度和适当性提示。"""

    def __init__(
        self,
        sensitive_keywords: Iterable[str] = SENSITIVE_KEYWORDS,
        investment_advice_patterns: Iterable[str] = INVESTMENT_ADVICE_PATTERNS,
        high_risk_products: Iterable[str] = HIGH_RISK_PRODUCTS,
    ):
        self.sensitive_keywords = tuple(sensitive_keywords)
        self.investment_advice_patterns = tuple(investment_advice_patterns)
        self.high_risk_products = tuple(high_risk_products)

    def check(
        self,
        text: str,
        *,
        user_role: str | None = None,
        client_id: str | None = None,
    ) -> ComplianceResult:
        flags: list[str] = []

        for keyword in self._matched_sensitive_keywords(text):
            flags.append(f"sensitive:{keyword}")

        for pattern in self._matched_investment_advice_patterns(text):
            flags.append(f"advice:{pattern}")

        if user_role == ROLE_COMPLIANCE and not self._has_article_reference(text):
            flags.append("citation_precision:missing_article")

        suitability_warning = ""
        if user_role == ROLE_ADVISOR and client_id:
            for product in self.high_risk_products:
                if product in text:
                    suitability_warning = SUITABILITY_WARNING
                    flags.append(f"suitability:{product}")
                    break

        passed = not any(
            flag.startswith(("sensitive:", "advice:", "citation_precision:")) for flag in flags
        )

        return {
            "passed": passed,
            "flags": flags,
            "risk_disclosure": self._generate_risk_disclosure(),
            "suitability_warning": suitability_warning,
        }

    def _contains_sensitive_info(self, text: str) -> bool:
        return any(self._matched_sensitive_keywords(text))

    def _contains_investment_advice(self, text: str) -> bool:
        return any(self._matched_investment_advice_patterns(text))

    def _generate_risk_disclosure(self) -> str:
        return RISK_DISCLOSURE

    def _matched_sensitive_keywords(self, text: str) -> list[str]:
        return [keyword for keyword in self.sensitive_keywords if keyword in text]

    def _matched_investment_advice_patterns(self, text: str) -> list[str]:
        return [pattern for pattern in self.investment_advice_patterns if pattern in text]

    def _has_article_reference(self, text: str) -> bool:
        return re.search(ARTICLE_REFERENCE_PATTERN, text) is not None
