"""Citation extraction and comprehensive answer verification."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation

from src.schemas.constants import (
    CONFIDENCE_HIGH,
    CONFIDENCE_LOW,
    CONFIDENCE_MEDIUM,
    META_CHUNK_ID,
    META_DATE,
    META_DOC_TYPE,
    META_PAGE_NUMBER,
    META_PERMISSION_LEVEL,
    META_SOURCE,
    META_STOCK_CODE,
    META_TITLE,
    PERMISSION_PUBLIC,
    RR_CONTENT,
    RR_DENIED,
    RR_METADATA,
    RR_SCORE,
)
from src.schemas.models import Citation
from src.schemas.typed_dicts import CitationDict, RetrievalResult, ToolCallDict

_STRUCTURED_METADATA_FIELDS = (
    (META_TITLE, "标题"),
    ("institution", "机构"),
    ("rating", "评级"),
    (META_DATE, "来源日期"),
    (META_STOCK_CODE, "股票代码"),
)


def _structured_metadata_evidence(metadata: dict) -> str:
    return "；".join(
        f"{label}={metadata[key]}"
        for key, label in _STRUCTURED_METADATA_FIELDS
        if metadata.get(key)
    )


class CitationExtractor:
    def extract(self, retrieval_results: list[RetrievalResult], query: str) -> list[CitationDict]:
        """提取引用，编号与 prompt 中的来源序号对齐。

        issues.md 一.7：prompt 按检索结果顺序给证据编号 [来源1..N]，
        引用列表也必须使用同一编号——同一证据去重跳过时保持其 prompt 序号，
        不再按提取后的列表位置重新编号，避免答案中的 [来源2] 展示时移位。
        """
        citations: list[CitationDict] = []
        eligible = [result for result in retrieval_results if not result.get(RR_DENIED)]
        seen_evidence = set()
        # 与 _build_reason_system_prompt 相同的上限：prompt 最多展示 5 条来源
        for index, result in enumerate(eligible, start=1):
            if index > 5:
                break
            metadata = result.get(RR_METADATA, {})
            quote = self._extract_quote(result.get(RR_CONTENT, ""), query)
            structured_evidence = self._structured_evidence(metadata)
            if structured_evidence:
                quote = f"{quote}\n结构化证据：{structured_evidence}"
            evidence_key = (metadata.get(META_SOURCE), self._normalize_evidence(quote))
            if evidence_key in seen_evidence:
                continue
            seen_evidence.add(evidence_key)
            citation = Citation(
                citation_id=f"cite_{index:03d}",
                doc_title=str(metadata.get(META_TITLE, "未知文档")),
                source=str(metadata.get(META_SOURCE, "")),
                doc_type=str(metadata.get(META_DOC_TYPE, "")),
                chunk_id=str(metadata.get(META_CHUNK_ID, "")),
                quote=quote,
                relevance_score=round(float(result.get(RR_SCORE, 0.0)), 4),
                permission_level=str(metadata.get(META_PERMISSION_LEVEL, PERMISSION_PUBLIC)),
                page_number=metadata.get(META_PAGE_NUMBER),
                retrieval_path=list(metadata.get("retrieval_path", ["vector_search"])),
                timestamp=datetime.now(timezone.utc).isoformat(),
                metadata=dict(metadata),
            )
            citations.append(CitationDict(**asdict(citation)))
        return citations

    def _structured_evidence(self, metadata: dict) -> str:
        return _structured_metadata_evidence(metadata)

    def _normalize_evidence(self, evidence: str) -> str:
        return re.sub(r"\s+", "", evidence).lower()

    def _extract_quote(self, content: str, query: str) -> str:
        sentences = [part.strip() for part in re.split(r"[。；\n]", content) if part.strip()]
        if not sentences:
            return content[:200]
        terms = {term.lower() for term in re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", query)}
        return max(
            sentences,
            key=lambda sentence: sum(term in sentence.lower() for term in terms),
        )[:200]


class SourceVerifier:
    # 与 CitationExtractor / prompt 的来源上限一致
    MAX_SOURCES = 5

    def verify(
        self,
        answer: str,
        citations: list[CitationDict],
        retrieval_results: list[RetrievalResult],
    ) -> dict:
        issues: list[str] = []
        # 编号有效性按 prompt 中的来源序号判定（非 denied 结果数，上限 5），
        # 而不是引用列表长度：去重会让引用列表短于来源序号（issues.md 一.7）
        source_count = min(
            self.MAX_SOURCES,
            sum(1 for result in retrieval_results if not result.get(RR_DENIED)),
        )
        for cite_id in re.findall(r"\[来源([^\]]*)\]", answer):
            if not cite_id.isdigit() or int(cite_id) < 1:
                issues.append(f"引用来源编号无效: {cite_id or '<empty>'}")
            elif int(cite_id) > source_count:
                issues.append(f"引用来源 {cite_id} 不存在")
        if "[来源" in answer and not citations:
            issues.append("答案包含引用标注但无检索结果")

        evidence_keys = {
            (
                result.get(RR_METADATA, {}).get(META_SOURCE),
                result.get(RR_METADATA, {}).get(META_CHUNK_ID),
            )
            for result in retrieval_results
            if not result.get(RR_DENIED)
        }
        for citation in citations:
            key = (citation.get("source"), citation.get("chunk_id"))
            if key not in evidence_keys:
                issues.append(f"引用不属于当前轮检索结果: {key}")
            quote = str(citation.get("quote", ""))
            metadata = citation.get("metadata", {})
            for field in ("institution", "rating", META_DATE, META_STOCK_CODE):
                value = str(metadata.get(field, ""))
                if value and value in answer and value not in quote:
                    issues.append(f"可见引用未包含答案使用的结构化事实: {field}={value}")
        return {"passed": not issues, "issues": issues}


class NumberVerifier:
    # 答案中的引用编号 [来源N] 不是业务数字
    _CITATION_MARKER_RE = re.compile(r"\[来源[^\]]*\]")
    # Markdown 有序列表的序号（行首 "1." / "2、"）不是业务数字
    _LIST_MARKER_RE = re.compile(r"(?m)^\s*\d+[.、)]\s*")
    # 数字复合词（日期 2024-04-26、区间 10-20）：把连字符两侧数字合并成一个 token，
    # 两侧（答案与证据）同规则归一化，避免 "04"、"26" 这类成分被单独校验。
    # 用 lookaround 而非 \b：中文语境下 \b 在 CJK 字符旁不成立
    _DATE_FULL_RE = re.compile(r"(?<![\d-])(\d{4})-(\d{1,2})-(\d{1,2})(?![\d-])")
    _DATE_YEAR_MONTH_RE = re.compile(r"(?<![\d-])(\d{4})-(\d{1,2})(?![\d-])")
    _NUMBER_RANGE_RE = re.compile(r"(?<![\d-])(\d{1,3})-(\d{1,3})(?![\d-])")

    @classmethod
    def _normalize_numeric_text(cls, text: str) -> str:
        text = text.replace(",", "")
        text = cls._DATE_FULL_RE.sub(r"\1\2\3", text)
        text = cls._DATE_YEAR_MONTH_RE.sub(r"\1\2", text)
        text = cls._NUMBER_RANGE_RE.sub(r"\1\2", text)
        return text

    @classmethod
    def _extract_answer_numbers(cls, answer: str) -> list[str]:
        """提取答案中的业务数字：先剔除引用编号与列表序号，再带符号/百分号提取。"""
        text = cls._CITATION_MARKER_RE.sub("", answer)
        text = cls._LIST_MARKER_RE.sub("", text)
        text = cls._normalize_numeric_text(text)
        return re.findall(r"-?\d+(?:\.\d+)?%?", text)

    @staticmethod
    def _number_in_evidence(number: str, evidence: str) -> bool:
        """检查数字是否出现在证据中，绑定数值边界与正负号（issues.md 一.7）。

        - "20" 不得命中 "120"（前向不能是数字或小数点）
        - "10%" 不得命中 "-10%"（负号必须显式出现在答案数字中才允许匹配）
        - 千分位逗号在两侧同时归一化后再匹配
        - ISSUE-13 数值等价兜底："1" ≡ "1.00"、"3.9%" ≡ "3.90%"——
          词面边界匹配会把同值异写的数字误判为编造
        """
        evidence = NumberVerifier._normalize_numeric_text(evidence)
        pattern = re.escape(number.replace(",", ""))
        # 前向：不能是数字/小数点；数字本身不带负号时，前向也不能是负号
        if number.startswith("-"):
            lookbehind = r"(?<![\d.])"
        else:
            lookbehind = r"(?<![\d.-])"
        # 后向：不能紧跟数字或小数点（"20" 不得命中 "2024"）
        lookahead = r"(?![\d.])"
        if re.search(f"{lookbehind}{pattern}{lookahead}", evidence) is not None:
            return True
        target = NumberVerifier._canonical_number(number.replace(",", ""))
        return any(
            NumberVerifier._canonical_number(match) == target
            for match in NumberVerifier._NUMBER_TOKEN_RE.findall(evidence)
        )

    _NUMBER_TOKEN_RE = re.compile(r"-?\d+(?:\.\d+)?%?")

    @staticmethod
    def _canonical_number(number: str) -> str:
        """数值等价键：小数去尾零后按值比较，保留正负号与百分号单位。"""
        unit = "%" if number.endswith("%") else ""
        value = number[:-1] if unit else number
        try:
            canonical = format(Decimal(value).normalize(), "f")
        except InvalidOperation:
            canonical = value
        if canonical in ("-0", "+0"):
            canonical = "0"
        return canonical + unit

    def verify(
        self,
        answer: str,
        retrieval_results: list[RetrievalResult],
        tool_calls: list[ToolCallDict],
    ) -> dict:
        numbers = self._extract_answer_numbers(answer)
        evidence = [
            "\n".join(
                filter(
                    None,
                    (
                        result.get(RR_CONTENT, ""),
                        _structured_metadata_evidence(result.get(RR_METADATA, {})),
                    ),
                )
            )
            for result in retrieval_results
        ]
        evidence.extend(
            str(call.get("output", "")) for call in tool_calls if call.get("success", False)
        )
        all_content = " ".join(evidence)
        issues = [
            f"数字 {number} 在检索或工具结果中未找到"
            for number in numbers
            if not self._number_in_evidence(number, all_content)
        ]
        return {
            "passed": not issues,
            "issues": issues,
            "numbers_found": len(numbers) - len(issues),
            "numbers_total": len(numbers),
        }


class ConsistencyVerifier:
    def verify(self, answer: str, citations: list[CitationDict]) -> dict:
        del citations
        issues = []
        for positive, negative in (("增持", "减持"), ("买入", "卖出"), ("看多", "看空")):
            if positive in answer and negative in answer:
                issues.append(f"发现矛盾信息：'{positive}' 和 '{negative}' 同时出现")
        return {"passed": not issues, "issues": issues}


# ══════════════════════════════════════════════════════════════════════
# ISSUE-22：口径标签与数值的成对校验
# ══════════════════════════════════════════════════════════════════════

# 口径别名 → 规范口径。同一口径的不同写法必须归一到同一个键，否则
# "归母净利润"与"归属于上市公司股东的净利润"会被误判成口径混用。
CALIBER_ALIASES: dict[str, str] = {
    "营业总收入": "total_operating_revenue",
    "营业收入": "operating_revenue",
    "归属于上市公司股东的净利润": "net_profit_attributable",
    "归母净利润": "net_profit_attributable",
    "扣除非经常性损益的净利润": "net_profit_deducted",
    "扣非归母净利润": "net_profit_deducted",
    "净利润": "net_profit",
    "营业利润": "operating_profit",
    "利润总额": "total_profit",
    "基本每股收益": "eps_basic",
    "摊薄每股收益": "eps_diluted",
    "每股收益": "eps",
    "毛利率": "gross_margin",
    "净利率": "net_margin",
}
# 长标签优先匹配："营业总收入" 不得被 "营业收入" 抢先切走，
# "扣非归母净利润" 不得被 "归母净利润" 抢先切走
_CALIBER_LABEL_RE = re.compile(
    "|".join(re.escape(label) for label in sorted(CALIBER_ALIASES, key=len, reverse=True))
)
# 金额单位 → 基准量纲倍数；长单位在前，避免 "百万元" 被 "万元" 抢先匹配
_UNIT_SCALES: dict[str, float] = {
    "百亿元": 1e10,
    "千万元": 1e7,
    "十亿元": 1e9,
    "百万元": 1e6,
    "亿元": 1e8,
    "万元": 1e4,
    "亿": 1e8,
    "万": 1e4,
    "元": 1.0,
    "%": 1.0,
    "倍": 1.0,
}
_UNIT_ALTERNATION = "|".join(
    re.escape(unit) for unit in sorted(_UNIT_SCALES, key=len, reverse=True)
)
_NUMBER_AND_UNIT_RE = re.compile(rf"\s*(-?\d[\d,]*(?:\.\d+)?)\s*({_UNIT_ALTERNATION})?")
# 口径标签与数值的绑定窗口（字符数）：够覆盖"营业收入（元）"这类写法
_CALIBER_WINDOW = 16
# 相对容差：覆盖"907.03 亿元"与"90,703,260,964.48 元"这类单位换算的舍入误差，
# 又不会把 445.17 与 444.64（相差 0.12%）判成同一个值
_CALIBER_RELATIVE_TOLERANCE = 5e-4
_UNIT_FAMILIES = {"%": "ratio", "倍": "multiple"}


def _unit_family(unit: str) -> str:
    return _UNIT_FAMILIES.get(unit, "currency")


@dataclass(frozen=True)
class _CaliberPair:
    """一处"口径标签 + 数值"绑定。"""

    label: str
    caliber: str
    raw_number: str
    raw_value: float
    scaled_value: float
    unit: str


class CaliberVerifier:
    """口径标签与数值的成对校验（ISSUE-22）。

    08-evaluation §2 要求结构化数字精确率 100%：数字取自证据但口径标错
    （例如把营业总收入写成营业收入）同样是错误答案，必须拦截。

    两类问题会判失败：
    1. 同一数值在答案中被绑定到与证据不同的口径（口径混用）；
    2. 答案使用了证据中从未出现的口径标签（凭空引入口径）。

    只在答案出现已知口径标签时生效；无口径标签的答案（产品、规则、FAQ）
    不受影响。
    """

    def verify(
        self,
        answer: str,
        retrieval_results: list[RetrievalResult],
        tool_calls: list[ToolCallDict],
    ) -> dict:
        answer_pairs = self._extract_pairs(answer)
        if not answer_pairs:
            return {"passed": True, "issues": [], "pairs_checked": 0}

        evidence_text = "\n".join(
            filter(
                None,
                (
                    result.get(RR_CONTENT, "")
                    for result in retrieval_results
                    if not result.get(RR_DENIED)
                ),
            )
        )
        tool_text = "\n".join(
            str(call.get("output", "")) for call in tool_calls if call.get("success", False)
        )
        evidence_pairs = self._extract_pairs(f"{evidence_text}\n{tool_text}")

        issues: list[str] = []
        for pair in answer_pairs:
            if not self._label_present(pair, evidence_text, tool_text):
                issues.append(f"口径标签未在证据中出现: {pair.label}")
                continue
            if any(
                candidate.caliber == pair.caliber and self._equivalent(pair, candidate)
                for candidate in evidence_pairs
            ):
                continue
            conflicting = sorted({
                candidate.label
                for candidate in evidence_pairs
                if candidate.caliber != pair.caliber and self._equivalent(pair, candidate)
            })
            if conflicting:
                issues.append(
                    f"口径与数值不匹配: {pair.label}={pair.raw_number}{pair.unit} "
                    f"在证据中对应 {'、'.join(conflicting)}"
                )
            else:
                issues.append(
                    f"口径与数值未在证据中成对出现: {pair.label}={pair.raw_number}{pair.unit}"
                )
        return {"passed": not issues, "issues": issues, "pairs_checked": len(answer_pairs)}

    def _extract_pairs(self, text: str) -> list[_CaliberPair]:
        pairs: list[_CaliberPair] = []
        for match in _CALIBER_LABEL_RE.finditer(text):
            label = match.group(0)
            window = text[match.end() : match.end() + _CALIBER_WINDOW]
            number_match = _NUMBER_AND_UNIT_RE.search(window)
            if number_match is None:
                continue
            raw_number = number_match.group(1)
            unit = number_match.group(2) or ""
            try:
                raw_value = float(raw_number.replace(",", ""))
            except ValueError:
                continue
            pairs.append(
                _CaliberPair(
                    label=label,
                    caliber=CALIBER_ALIASES[label],
                    raw_number=raw_number,
                    raw_value=raw_value,
                    scaled_value=raw_value * _UNIT_SCALES.get(unit, 1.0),
                    unit=unit,
                )
            )
        return pairs

    @staticmethod
    def _label_present(pair: _CaliberPair, *texts: str) -> bool:
        """证据中是否出现过该口径的任一同义写法（含全称与简称）。"""
        aliases = [alias for alias, caliber in CALIBER_ALIASES.items() if caliber == pair.caliber]
        return any(alias in text for text in texts for alias in aliases)

    @staticmethod
    def _equivalent(left: _CaliberPair, right: _CaliberPair) -> bool:
        """数值等价判定：单位族必须一致，量纲换算后或原值在容差内相等。"""
        if left.unit and right.unit and _unit_family(left.unit) != _unit_family(right.unit):
            return False
        if CaliberVerifier._close(left.scaled_value, right.scaled_value):
            return True
        # 任一侧未写单位时，原值相同即视为同一数值（"922.78" ≡ "922.78亿元"）
        if not left.unit or not right.unit:
            return CaliberVerifier._close(left.raw_value, right.raw_value)
        return False

    @staticmethod
    def _close(left: float, right: float) -> bool:
        if left == right:
            return True
        scale = max(abs(left), abs(right))
        if scale == 0:
            return False
        return abs(left - right) / scale <= _CALIBER_RELATIVE_TOLERANCE


class HallucinationDetector:
    # 中文功能词/虚词：不承载业务事实，句子比对时从两侧剔除，
    # 降低同义改写（语序、措辞）造成的覆盖率噪声（ISSUE-13）。
    # 刻意不含否定词（不/没/无）——否定歧义交由 ConsistencyVerifier 把关
    _FUNCTION_WORD_CHARS = frozenset(
        "的了在是和与及或有为以于由从将已等均也都还并但因所该这那之其它她把被向就才只很更最而"
    )
    # 中文日期 → 数字写法："2024年4月26日" 与证据 "2024-04-26" 是同一日期
    _CN_FULL_DATE_RE = re.compile(r"(\d{4})年(\d{1,2})月(\d{1,2})[日号]")
    _CN_YEAR_MONTH_RE = re.compile(r"(\d{4})年(\d{1,2})月(?!\d)")
    _CN_MONTH_DAY_RE = re.compile(r"(?<!\d)(\d{1,2})月(\d{1,2})[日号](?!\d)")

    @classmethod
    def _content_tokens(cls, text: str) -> set[str]:
        """内容词归一化：剔除功能字、统一中英日期写法、数字去前导零。"""
        text = text.replace(",", "").replace("，", "")
        text = cls._CN_FULL_DATE_RE.sub(r"\1-\2-\3", text)
        text = cls._CN_YEAR_MONTH_RE.sub(r"\1-\2", text)
        text = cls._CN_MONTH_DAY_RE.sub(r"\1-\2", text)
        tokens = set(re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", text.lower()))
        normalized: set[str] = set()
        for token in tokens:
            if token.isdigit():
                # "04" 与 "4" 同一数字（月/日零填充差异）
                normalized.add(token.lstrip("0") or "0")
            elif token not in cls._FUNCTION_WORD_CHARS:
                normalized.add(token)
        return normalized

    def detect(
        self,
        answer: str,
        retrieval_results: list[RetrievalResult],
        tool_calls: list[ToolCallDict],
    ) -> dict:
        usable = [result for result in retrieval_results if not result.get(RR_DENIED)]
        evidence = [
            "\n".join(
                filter(
                    None,
                    (
                        result.get(RR_CONTENT, ""),
                        _structured_metadata_evidence(result.get(RR_METADATA, {})),
                    ),
                )
            )
            for result in usable
        ]
        evidence.extend(
            str(call.get("output", "")) for call in tool_calls if call.get("success", False)
        )
        structured_tool_evidence = [
            str(call.get("output", ""))
            for call in tool_calls
            if call.get("success", False) and self._is_structured_output(call.get("output", ""))
        ]
        if not evidence:
            return {
                "passed": False,
                "issues": ["无检索结果或成功工具输出支撑"],
                "hallucination_score": 1.0,
            }

        sentences = self._answer_sentences(answer)
        coverage = [
            any(self._similar(sentence, item) for item in evidence)
            or any(
                self._structured_claim_supported(sentence, item)
                for item in structured_tool_evidence
            )
            for sentence in sentences
            if not sentence.startswith(("【风险提示】", "【适当性提示】"))
        ]
        ratio = 1 - (sum(coverage) / len(coverage) if coverage else 1.0)
        issues = [f"幻觉比例过高：{ratio:.1%}"] if ratio > 0.3 else []
        return {"passed": not issues, "issues": issues, "hallucination_score": ratio}

    def _similar(self, text1: str, text2: str) -> bool:
        if text1 in text2 or text2 in text1:
            return True
        tokens1 = self._content_tokens(text1)
        tokens2 = self._content_tokens(text2)
        if not tokens1:
            return False
        return len(tokens1 & tokens2) / len(tokens1) > 0.5

    def _answer_sentences(self, answer: str) -> list[str]:
        lines = answer.splitlines()
        sentences: list[str] = []
        for index, line in enumerate(lines):
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or self._is_table_separator(stripped):
                continue
            next_line = lines[index + 1].strip() if index + 1 < len(lines) else ""
            if stripped.startswith("|") and self._is_table_separator(next_line):
                continue
            sentences.extend(part.strip() for part in re.split(r"[。；]", stripped) if part.strip())
        return sentences

    def _is_table_separator(self, text: str) -> bool:
        return bool(re.fullmatch(r"\|?[\s:|-]+\|?", text)) and "-" in text

    def _is_structured_output(self, output: object) -> bool:
        text = str(output).strip()
        return (text.startswith("[") and text.endswith("]")) or (
            text.startswith("{") and text.endswith("}")
        )

    def _structured_claim_supported(self, sentence: str, evidence: str) -> bool:
        claims = re.findall(
            r"[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?(?:-\d+)*",
            sentence.lower(),
        )
        return bool(claims) and all(claim in evidence.lower() for claim in claims)


def summarize_verification_attempts(attempts: list[dict]) -> dict:
    """按轮次汇总验证快照，定位重推原因（ISSUE-25）。

    "误判类重推"的判定依据：某轮失败但失败原因全部属于格式/引用标注
    （``failure_kind == "format"``），即证据本身可支撑、只是标注写法不合规。
    这类重推不需要重新取证，是 ISSUE-13 类验证器误判的残留信号，
    ``format_only_retries`` 应长期为 0。
    """
    failures = [snapshot for snapshot in attempts if not snapshot.get("passed")]
    first_failure = failures[0] if failures else None
    return {
        "attempts": len(attempts),
        "failures": len(failures),
        "first_failure_round": first_failure.get("round") if first_failure else None,
        "first_failure_kind": first_failure.get("failure_kind") if first_failure else None,
        "first_failure_issues": list(first_failure.get("issues", [])) if first_failure else [],
        "format_only_retries": sum(
            1 for snapshot in failures if snapshot.get("failure_kind") == "format"
        ),
    }


class ComprehensiveVerifier:
    def __init__(self):
        self.source_verifier = SourceVerifier()
        self.number_verifier = NumberVerifier()
        self.caliber_verifier = CaliberVerifier()
        self.consistency_verifier = ConsistencyVerifier()
        self.hallucination_detector = HallucinationDetector()

    def verify(
        self,
        answer: str,
        citations: list[CitationDict],
        retrieval_results: list[RetrievalResult],
        tool_calls: list[ToolCallDict],
    ) -> dict:
        checks = {
            "source_verification": self.source_verifier.verify(
                answer, citations, retrieval_results
            ),
            "number_verification": self.number_verifier.verify(
                answer, retrieval_results, tool_calls
            ),
            "caliber_verification": self.caliber_verifier.verify(
                answer, retrieval_results, tool_calls
            ),
            "consistency_verification": self.consistency_verifier.verify(answer, citations),
            "hallucination_detection": self.hallucination_detector.detect(
                answer, retrieval_results, tool_calls
            ),
        }
        issues = [
            f"{name}: {issue}"
            for name, result in checks.items()
            for issue in result.get("issues", [])
        ]
        passed = all(result.get("passed", False) for result in checks.values())
        # ISSUE-13：失败分类——仅来源/引用格式问题为 "format"（可局部修复），
        # 数字/一致性/幻觉为 "facts"（需重新取证或删除无依据内容）。
        # 重跑节点据此给出不同的修正指令，避免"格式不符"触发整段重检索式重跑
        failure_kind = None
        if not passed and issues:
            failure_kind = (
                "format"
                if all(issue.startswith("source_verification:") for issue in issues)
                else "facts"
            )
        score = checks["hallucination_detection"].get("hallucination_score", 1.0)
        confidence = (
            CONFIDENCE_LOW if not passed else CONFIDENCE_MEDIUM if score > 0.1 else CONFIDENCE_HIGH
        )
        return {
            "passed": passed,
            "issues": issues,
            "checks": checks,
            "confidence": confidence,
            "failure_kind": failure_kind,
        }
