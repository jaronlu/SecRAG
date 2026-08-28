"""合规检查工具。

两层防护：
1. 正则模糊匹配层（fast-path）：允许关键词中间插入 0-3 个字符，覆盖"推荐你买入"等改写。
2. 模式变体扩展：目标价格、TP、给予买入评级、可以考虑买入等常见绕过写法。
"""

import re
from collections.abc import Iterable

from src.schemas.constants import ROLE_ADVISOR, ROLE_COMPLIANCE
from src.schemas.typed_dicts import ComplianceResult

TARGET_PRICE_PATTERN = "目标" + "价"
# 显示用模式名（保持向后兼容，用于 flag 标识和 nodes.py 的 verify 层）
INVESTMENT_ADVICE_PATTERNS: tuple[str, ...] = (
    "推荐" + "买" + "入",
    "建议" + "买" + "入",
    "建议" + "卖" + "出",
    "建议" + "增" + "持",
    "建议" + "减" + "持",
    TARGET_PRICE_PATTERN,
)

# ══════════════════════════════════════════════════════════════════════
# 正则模式：允许关键词中间插入 0-3 个字符，覆盖常见改写绕过
# 注意：不匹配"评级为买入"等研报引用句式（给予+评级组合易与引用混淆，不纳入）
# ══════════════════════════════════════════════════════════════════════
# 推荐买入：推荐买入 / 推荐你买入 / 推荐可以买入
_RECOMMEND_BUY = (re.compile(r"推[荐议].{0,3}买[入进]"),)
# 建议买入：建议买入 / 建议你买入 / 建议可以买入 / 可以考虑买入
_ADVISE_BUY = (
    re.compile(r"建[议].{0,3}买[入进]"),
    re.compile(r"可以考虑.{0,3}买[入进]"),
)
# 建议卖出：建议卖出 / 建议你卖出 / 可以考虑卖出
_ADVISE_SELL = (
    re.compile(r"建[议].{0,3}卖[出]"),
    re.compile(r"推[荐议].{0,3}卖[出]"),
    re.compile(r"可以考虑.{0,3}卖[出]"),
)
# 建议增持：建议增持 / 建议你增持
_ADVISE_INCREASE = (
    re.compile(r"建[议].{0,3}增[持]"),
    re.compile(r"推[荐议].{0,3}增[持]"),
)
# 建议减持：建议减持 / 建议你减持
_ADVISE_DECREASE = (
    re.compile(r"建[议].{0,3}减[持]"),
    re.compile(r"推[荐议].{0,3}减[持]"),
)
# 目标价：目标价 / 目标价格 / TP / target price
_TARGET_PRICE_REGEXES = (
    re.compile(r"目标[价]"),
    re.compile(r"目标价格"),
    re.compile(r"\bTP\b", re.IGNORECASE),
    re.compile(r"target\s*price", re.IGNORECASE),
)

# 模式名 → 正则列表 的映射，每个模式名独立匹配
ADVICE_REGEX_MAP: dict[str, tuple[re.Pattern[str], ...]] = {
    "推荐买入": _RECOMMEND_BUY,
    "建议买入": _ADVISE_BUY,
    "建议卖出": _ADVISE_SELL,
    "建议增持": _ADVISE_INCREASE,
    "建议减持": _ADVISE_DECREASE,
    TARGET_PRICE_PATTERN: _TARGET_PRICE_REGEXES,
}

SENSITIVE_KEYWORDS: tuple[str, ...] = ("内" + "幕" + "信息", "未" + "公开", "业绩" + "预测")
HIGH_RISK_PRODUCTS: tuple[str, ...] = ("标的型" + "产品", "混合型" + "产品", "私" + "募" + "产品")
ARTICLE_REFERENCE_PATTERN = r"第[一二三四五六七八九十百千]+条|第\d+条|Article\s+\d+"
RISK_DISCLOSURE = "\n\n【风险提示】本回答仅供参考，不构成业务建议。市场有风险，业务需谨慎。"
SUITABILITY_WARNING = "\n\n【适当性提示】该产品风险等级较高，请确认客户风险承受能力是否匹配。"


def matches_investment_advice(text: str) -> list[str]:
    """用正则模糊匹配检测投资建议，返回命中的模式名列表。

    允许关键词中间插入 0-3 个字符，覆盖"推荐你买入"、"可以考虑买入"等改写。
    """
    matched: list[str] = []
    seen: set[str] = set()
    for pattern_name, regexes in ADVICE_REGEX_MAP.items():
        if pattern_name in seen:
            continue
        if any(rx.search(text) for rx in regexes):
            matched.append(pattern_name)
            seen.add(pattern_name)
    return matched


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
        allow_attributed_target_price: bool = False,
    ) -> ComplianceResult:
        flags: list[str] = []

        for keyword in self._matched_sensitive_keywords(text):
            flags.append(f"sensitive:{keyword}")

        for pattern in self._matched_investment_advice_patterns(text):
            if pattern == TARGET_PRICE_PATTERN and allow_attributed_target_price:
                continue
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
        """用正则模糊匹配替代原始子串匹配，覆盖改写绕过。"""
        return matches_investment_advice(text)

    def _has_article_reference(self, text: str) -> bool:
        return re.search(ARTICLE_REFERENCE_PATTERN, text) is not None
