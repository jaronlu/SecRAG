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
# TP 边界只排除 ASCII 字母相邻：匹配前的全空白归一化把 "TP 12.5" 变成 "TP12.5"，
# 且 \b 的 \w 包含中文，"TP为12.5" 这类紧邻汉字写法同样令 \b 失效（DEF-001）
_TARGET_PRICE_REGEXES = (
    re.compile(r"目标[价]"),
    re.compile(r"目标价格"),
    re.compile(r"(?<![A-Za-z])TP(?![A-Za-z])", re.IGNORECASE),
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
    匹配前去除空白字符，防止"推 荐 你 买 入"等空格分隔绕过。
    """
    matched: list[str] = []
    seen: set[str] = set()
    # 去除所有空白字符，防止空格/制表符分隔绕过
    normalized = re.sub(r"\s+", "", text)
    for pattern_name, regexes in ADVICE_REGEX_MAP.items():
        if pattern_name in seen:
            continue
        if any(rx.search(normalized) for rx in regexes):
            matched.append(pattern_name)
            seen.add(pattern_name)
    return matched


# ══════════════════════════════════════════════════════════════════════
# ISSUE-31：投资建议索取意图的确定性识别（query 层，不依赖生成文本）。
# 真实 LLM 会自律规避"建议买入"类表述，答案文本规则永不触发，blocked 终态
# 在默认链路不可达。PRD §合规"投顾不得输出投资建议"+ impl-08 §3.2"必须拦截
# 系统自身的建议性改写"要求系统行为确定性优先：查询本身索取买卖时点/操作
# 建议时，直接以 blocked 终态拦截，不再进入推理。模式刻意收敛为高置信问法，
# 知识型咨询（"什么是买入评级""投资建议有哪些类型"）不得误伤。
# ══════════════════════════════════════════════════════════════════════
ADVICE_SEEKING_INTENT_REGEXES: tuple[re.Pattern[str], ...] = (
    re.compile(r"适合.{0,4}买"),  # 适合买入吗 / 适合现在买入吗
    re.compile(r"值得.{0,4}(买|入手)"),  # 值得买吗 / 值得入手吗
    re.compile(r"(该|要|能|可[以不]).{0,3}买(吗|么|不)"),  # 该买吗 / 能买吗 / 可以买不
    re.compile(r"(能不能|可不可以|该不该).{0,3}买"),  # 能买吗的口语变体（无句尾语气词）
    re.compile(r"(现在|马上|立刻|及时).{0,3}(买|买入|加仓|建仓)"),  # 现在买入合适吗
    # 时点操作/仓位词在问答中出现即索取操作指引（重仓/满仓为 BUG-4 补充：
    # "现在重仓什么好" 不带任何上述句式，仅靠仓位词命中）
    re.compile(r"(加仓|建仓|清仓|抄底|重仓|满仓)"),
    re.compile(r"买哪(只|个|款)"),  # 该买哪只股票 / 买哪个产品好
    re.compile(r"推荐.{0,4}(股票|产品)"),  # 帮我推荐合适的股票/产品
    re.compile(r"(给我|请|帮忙|帮).{0,6}(操作|投资|买卖|交易)建议"),  # 给我一个操作建议
)

# 防空格绕过与 matches_investment_advice 相同：匹配前去除全部空白
_ADVICE_INTENT_FLAG_PREFIX = "advice_intent"


def matches_advice_seeking_intent(query: str) -> list[str]:
    """检测查询中的投资建议索取意图，返回命中的意图名列表（ISSUE-31）。

    与 matches_investment_advice（答案文本层）互补：本函数在查询理解后
    确定性命中"索取操作建议"的问法，不依赖 LLM 是否自律规避建议表述。
    """
    normalized = re.sub(r"\s+", "", query)
    return [
        regex.pattern
        for regex in ADVICE_SEEKING_INTENT_REGEXES
        if regex.search(normalized)
    ]


def advice_intent_flags(query: str) -> list[str]:
    """把意图命中转成 compliance flags（审计层可见，前缀 advice_intent）。"""
    return [f"{_ADVICE_INTENT_FLAG_PREFIX}:{pattern}" for pattern in matches_advice_seeking_intent(query)]


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
