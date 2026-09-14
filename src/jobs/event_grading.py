"""Rule based event grading for daily portfolio scans.

Why rules and not a model: the grade decides what an analyst sees first, so it
must be reproducible, reviewable, and tunable by an operator without touching
code. Every decision therefore carries `matched_rules` and `reasons`. A P2
verdict is never silent either — it always reports which rules were evaluated
and why nothing cleared the bar, because that difference is what later lets
someone retune recall instead of guessing at it.

All thresholds live in `GradingThresholds`; nothing is hard coded in the
comparison logic.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from typing_extensions import TypedDict

from src.schemas.constants import META_DATE, META_STOCK_CODE, META_TITLE

GRADE_P0 = "P0"
GRADE_P1 = "P1"
GRADE_P2 = "P2"
VALID_GRADES = (GRADE_P0, GRADE_P1, GRADE_P2)

# Grades allowed into downstream briefings. P2 is stored, not pushed.
PUSHABLE_GRADES = (GRADE_P0, GRADE_P1)

RULE_P0_DOCUMENT_KEYWORD = "p0_document_keyword"
RULE_P1_DOCUMENT_KEYWORD = "p1_document_keyword"
RULE_P0_RATING = "p0_rating"
RULE_P1_RATING = "p1_rating"
RULE_P0_PRICE_MOVE = "p0_price_move"
RULE_P1_PRICE_MOVE = "p1_price_move"
RULE_NO_SIGNAL = "no_signal"


@dataclass(frozen=True)
class GradingThresholds:
    """Every tunable knob used by the grader.

    Defaults are heuristic starting points for analyst review, not calibrated
    values; retune them from observed miss/false-positive rates rather than
    treating them as fixed.
    """

    p0_document_keywords: tuple[str, ...] = (
        "业绩预告",
        "业绩快报",
        "业绩预增",
        "业绩预亏",
        "重大资产重组",
        "发行股份购买资产",
        "停牌",
        "退市",
        "风险警示",
        "立案调查",
        "问询函",
        "监管函",
        "要约收购",
    )
    p1_document_keywords: tuple[str, ...] = (
        "年度报告",
        "半年度报告",
        "季度报告",
        "股东大会",
        "对外投资",
        "对外担保",
        "股份回购",
        "权益分派",
    )
    p0_ratings: tuple[str, ...] = ("卖出", "减持")
    p1_ratings: tuple[str, ...] = ("买入", "增持")
    p0_pct_change_abs: float = 7.0
    p1_pct_change_abs: float = 3.0


DEFAULT_THRESHOLDS = GradingThresholds()


class GradingDecisionDict(TypedDict):
    """Grade plus the justification that produced it.

    `reasons` is human readable and meant for the briefing; `matched_rules` is
    the stable machine identifier used for auditing and threshold retuning.
    """

    grade: str
    reasons: list[str]
    matched_rules: list[str]
    evidence: dict[str, str]


def _decision(
    grade: str,
    matched_rules: list[str],
    reasons: list[str],
    evidence: dict[str, str],
) -> GradingDecisionDict:
    if grade not in VALID_GRADES:
        raise ValueError(f"unknown grade: {grade}")
    if not reasons:
        raise ValueError("a decision must state at least one reason")
    return {
        "grade": grade,
        "reasons": reasons,
        "matched_rules": matched_rules,
        "evidence": evidence,
    }


def _extract_title(record: Mapping[str, object]) -> str:
    value = record.get(META_TITLE) or record.get("title") or ""
    return str(value)


def grade_document(
    record: Mapping[str, object],
    *,
    thresholds: GradingThresholds = DEFAULT_THRESHOLDS,
) -> GradingDecisionDict:
    """Grade an announcement or document metadata record by its title."""
    title = _extract_title(record)
    evidence = {
        "title": title,
        "date": str(record.get(META_DATE) or ""),
        "stock_code": str(record.get(META_STOCK_CODE) or ""),
    }

    p0_hits = [word for word in thresholds.p0_document_keywords if word in title]
    if p0_hits:
        return _decision(
            GRADE_P0,
            [RULE_P0_DOCUMENT_KEYWORD],
            [f"标题命中 P0 关键词：{'、'.join(p0_hits)}"],
            {**evidence, "matched_keywords": "、".join(p0_hits)},
        )

    p1_hits = [word for word in thresholds.p1_document_keywords if word in title]
    if p1_hits:
        return _decision(
            GRADE_P1,
            [RULE_P1_DOCUMENT_KEYWORD],
            [f"标题命中 P1 关键词：{'、'.join(p1_hits)}"],
            {**evidence, "matched_keywords": "、".join(p1_hits)},
        )

    checked = len(thresholds.p0_document_keywords) + len(thresholds.p1_document_keywords)
    return _decision(
        GRADE_P2,
        [RULE_NO_SIGNAL],
        [f"标题未命中任何 P0/P1 关键词（已比对 {checked} 条关键词）"],
        evidence,
    )


def grade_research_report(
    row: Mapping[str, object],
    *,
    thresholds: GradingThresholds = DEFAULT_THRESHOLDS,
) -> GradingDecisionDict:
    """Grade a research report index row by its rating value."""
    rating = str(row.get("东财评级") or row.get("rating") or "").strip()
    institution = str(row.get("机构") or row.get("institution") or "").strip()
    title = str(row.get("报告名称") or row.get("title") or "")
    evidence = {
        "title": title,
        "institution": institution,
        "rating": rating,
        "date": str(row.get("日期") or row.get("date") or ""),
        "stock_code": str(row.get("sample_stock_code") or row.get("stock_code") or ""),
    }

    if rating in thresholds.p0_ratings:
        return _decision(
            GRADE_P0,
            [RULE_P0_RATING],
            [f"评级为「{rating}」，命中 P0 负向评级"],
            evidence,
        )
    if rating in thresholds.p1_ratings:
        return _decision(
            GRADE_P1,
            [RULE_P1_RATING],
            [f"评级为「{rating}」，命中 P1 正向评级"],
            evidence,
        )

    candidates = "、".join((*thresholds.p0_ratings, *thresholds.p1_ratings))
    return _decision(
        GRADE_P2,
        [RULE_NO_SIGNAL],
        [f"评级「{rating or '(空)'}」不在关注清单内（{candidates}）"],
        evidence,
    )


def grade_quote_move(
    row: Mapping[str, object],
    *,
    thresholds: GradingThresholds = DEFAULT_THRESHOLDS,
) -> GradingDecisionDict:
    """Grade a single day price move row by absolute percentage change."""
    raw = row.get("pct_change", row.get("pctChg"))
    try:
        pct = abs(float(str(raw)))
    except (TypeError, ValueError):
        return _decision(
            GRADE_P2,
            [RULE_NO_SIGNAL],
            [f"涨跌幅取值无效（{raw!r}），无法参与阈值判定"],
            {"pct_change": str(raw)},
        )

    evidence = {
        "pct_change": f"{pct:.2f}",
        "date": str(row.get("date") or ""),
        "stock_code": str(row.get("code") or row.get("stock_code") or ""),
    }

    if pct >= thresholds.p0_pct_change_abs:
        return _decision(
            GRADE_P0,
            [RULE_P0_PRICE_MOVE],
            [f"日涨跌幅 {pct:.2f}% ≥ P0 阈值 {thresholds.p0_pct_change_abs}%"],
            evidence,
        )
    if pct >= thresholds.p1_pct_change_abs:
        return _decision(
            GRADE_P1,
            [RULE_P1_PRICE_MOVE],
            [f"日涨跌幅 {pct:.2f}% ≥ P1 阈值 {thresholds.p1_pct_change_abs}%"],
            evidence,
        )
    return _decision(
        GRADE_P2,
        [RULE_NO_SIGNAL],
        [
            f"日涨跌幅 {pct:.2f}% 低于 P1 阈值 {thresholds.p1_pct_change_abs}%，"
            f"未达 P0 阈值 {thresholds.p0_pct_change_abs}%"
        ],
        evidence,
    )
