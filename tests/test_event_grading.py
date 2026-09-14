"""Offline tests for daily scanning and event grading.

Nothing here touches the network or the real data directory: every fixture is
built into pytest's tmp_path and the portfolio store uses a throwaway sqlite
file.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.jobs.daily_scan import (
    EVENT_STATUS_FILTERED,
    EVENT_STATUS_PENDING,
    SOURCE_DOCUMENT,
    collect_candidates,
    run_daily_scan,
    SQLiteDailyScanStore,
)
from src.jobs.event_grading import (
    GRADE_P0,
    GRADE_P1,
    GRADE_P2,
    GradingThresholds,
    grade_document,
    grade_quote_move,
    grade_research_report,
)
from src.portfolio.store import SQLitePortfolioStore
from src.schemas.constants import (
    META_ALLOWED_ROLES,
    META_DATE,
    META_DOC_TYPE,
    META_PERMISSION_LEVEL,
    META_RETRIEVAL_SOURCE,
    META_SOURCE,
    META_STOCK_CODE,
    META_TITLE,
)

SCAN_DATE = "2026-09-14"


def build_output_dir(root: Path, *, symbols: list[str] | None = None) -> Path:
    """Create a minimal artifact tree shaped like the T-001 fetch output."""
    output_dir = root / "real_securities_data"
    (output_dir / "financials").mkdir(parents=True)
    for symbol in symbols or []:
        _write_document(output_dir, symbol, f"{symbol} 关于重大资产重组的公告", "2026-09-14")
        _write_document(output_dir, symbol, f"{symbol} 2025年年度报告", "2026-09-13")
        _write_document(output_dir, symbol, f"{symbol} 办公地址变更公告", "2026-09-10")
        _write_quotes(output_dir, symbol)
    _write_research_index(output_dir)
    return output_dir


def _write_document(output_dir: Path, symbol: str, title: str, date: str) -> None:
    report_dir = output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = report_dir / f"{symbol}_{title}.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")
    meta = {
        META_DOC_TYPE: "announcement",
        META_RETRIEVAL_SOURCE: "report",
        META_PERMISSION_LEVEL: "internal",
        META_ALLOWED_ROLES: ["advisor"],
        META_TITLE: title,
        META_DATE: date,
        META_STOCK_CODE: symbol,
        META_SOURCE: "http://example.invalid/doc",
        "provider": "cninfo",
        "sha256": "0" * 64,
    }
    pdf_path.with_name(pdf_path.name + ".meta.json").write_text(
        json.dumps(meta, ensure_ascii=False), encoding="utf-8"
    )


def _write_quotes(output_dir: Path, symbol: str) -> None:
    frame = pd.DataFrame(
        [
            {"date": "2026-09-11", "code": symbol, "pct_change": 0.4},
            {"date": "2026-09-12", "code": symbol, "pct_change": -4.2},
            {"date": "2026-09-14", "code": symbol, "pct_change": 9.1},
        ]
    )
    frame.to_csv(output_dir / "financials" / f"efinance_{symbol}_quote_history.csv", index=False)


def _write_research_index(output_dir: Path) -> None:
    frame = pd.DataFrame(
        [
            {
                "日期": "2026-09-14",
                "报告名称": "深度报告：增长确定性提升",
                "机构": "示例证券",
                "东财评级": "买入",
                "报告PDF链接": "http://example.invalid/1.pdf",
                "sample_stock_code": "600519",
            },
            {
                "日期": "2026-09-14",
                "报告名称": "风险提示：需求走弱",
                "机构": "示例证券",
                "东财评级": "减持",
                "报告PDF链接": "http://example.invalid/2.pdf",
                "sample_stock_code": "000001",
            },
        ]
    )
    frame.to_csv(output_dir / "financials" / "research_reports_index.csv", index=False)


@pytest.fixture()
def workspace(tmp_path: Path) -> tuple[Path, SQLitePortfolioStore, SQLiteDailyScanStore]:
    output_dir = build_output_dir(tmp_path, symbols=["600519", "000001"])
    portfolio = SQLitePortfolioStore(tmp_path / "portfolio.db")
    portfolio.add_position(user_id="u1", stock_code="600519", stock_name="贵州茅台")
    portfolio.add_position(user_id="u1", stock_code="000001", stock_name="平安银行")
    portfolio.add_position(user_id="u2", stock_code="600036", stock_name="招商银行")
    scan_store = SQLiteDailyScanStore(tmp_path / "scan.db")
    return output_dir, portfolio, scan_store


def scan(workspace, **kwargs):
    output_dir, portfolio, scan_store = workspace
    kwargs.setdefault("scan_date", SCAN_DATE)
    return run_daily_scan(
        output_dir=output_dir,
        portfolio_store=portfolio,
        scan_store=scan_store,
        **kwargs,
    )


# --------------------------------------------------------------------------
# grading rules
# --------------------------------------------------------------------------


def test_document_title_decides_grade():
    assert grade_document({META_TITLE: "600519 关于重大资产重组的公告"})["grade"] == GRADE_P0
    assert grade_document({META_TITLE: "600519 2025年年度报告"})["grade"] == GRADE_P1
    assert grade_document({META_TITLE: "600519 办公地址变更公告"})["grade"] == GRADE_P2


def test_every_decision_carries_reasons_and_rule_ids():
    decision = grade_document({META_TITLE: "600519 关于重大资产重组的公告"})
    assert decision["reasons"], "a grade without a reason cannot be reviewed"
    assert decision["matched_rules"], "a grade must name the rule that fired"
    assert decision["evidence"], "a grade must carry the evidence it was based on"


def test_rejected_grade_still_states_why():
    decision = grade_document({META_TITLE: "600519 办公地址变更公告"})
    assert decision["grade"] == GRADE_P2
    joined = " ".join(decision["reasons"])
    assert "未命中" in joined
    assert decision["matched_rules"] == ["no_signal"]


def test_quote_move_uses_absolute_threshold():
    assert grade_quote_move({"pct_change": 9.1})["grade"] == GRADE_P0
    assert grade_quote_move({"pct_change": -9.1})["grade"] == GRADE_P0
    assert grade_quote_move({"pct_change": 4.2})["grade"] == GRADE_P1
    assert grade_quote_move({"pct_change": 0.4})["grade"] == GRADE_P2


def test_invalid_quote_value_is_filtered_not_fatal():
    decision = grade_quote_move({"pct_change": "n/a"})
    assert decision["grade"] == GRADE_P2
    assert "无效" in " ".join(decision["reasons"])


def test_research_rating_decides_grade():
    assert grade_research_report({"东财评级": "减持"})["grade"] == GRADE_P0
    assert grade_research_report({"东财评级": "买入"})["grade"] == GRADE_P1
    assert grade_research_report({"东财评级": "中性"})["grade"] == GRADE_P2


def test_thresholds_are_externalised_and_tunable():
    tuned = GradingThresholds(p1_document_keywords=("办公地址",), p0_pct_change_abs=0.1)
    assert (
        grade_document({META_TITLE: "600519 办公地址变更公告"}, thresholds=tuned)["grade"]
        == GRADE_P1
    )
    assert grade_quote_move({"pct_change": 0.4}, thresholds=tuned)["grade"] == GRADE_P0


# --------------------------------------------------------------------------
# scan behaviour
# --------------------------------------------------------------------------


def test_scan_collects_all_three_source_kinds(workspace):
    candidates = collect_candidates(workspace[0])
    kinds = {item["source_kind"] for item in candidates}
    assert kinds == {SOURCE_DOCUMENT, "quote", "research"}


def test_scan_builds_events_only_for_held_symbols(workspace):
    summary = scan(workspace, user_ids=["u1"])
    _, _, scan_store = workspace
    events = scan_store.list_events(user_id="u1")
    assert summary["events_inserted"] == len(events)
    # u1 holds two symbols; the user without any match must see nothing.
    assert scan_store.list_events(user_id="u2") == []


def test_research_rows_route_to_the_holder_not_the_index_symbol(workspace):
    scan(workspace, user_ids=["u1"])
    _, _, scan_store = workspace
    research = [
        item for item in scan_store.list_events(user_id="u1") if item["source_kind"] == "research"
    ]
    codes = {item["stock_code"] for item in research}
    assert codes == {"600519", "000001"}


def test_rerunning_same_day_produces_no_duplicate_cards(workspace):
    first = scan(workspace, user_ids=["u1"])
    second = scan(workspace, user_ids=["u1"])
    _, _, scan_store = workspace

    assert first["events_inserted"] > 0
    assert second["events_inserted"] == 0
    assert second["duplicates_skipped"] == first["events_inserted"]

    events = scan_store.list_events(user_id="u1", scan_date=SCAN_DATE)
    keys = [item["dedupe_key"] for item in events]
    assert len(keys) == len(set(keys)), "duplicate cards slipped past the dedupe key"
    assert len(events) == first["events_inserted"]


def test_rerunning_on_a_later_day_still_produces_no_duplicate_cards(workspace):
    first = scan(workspace, user_ids=["u1"])
    again = scan(workspace, user_ids=["u1"], scan_date="2026-09-15")
    _, _, scan_store = workspace

    assert again["events_inserted"] == 0
    total = len(scan_store.list_events(user_id="u1"))
    assert total == first["events_inserted"], "the same event resurfaced on a later day"


def test_p2_cards_are_stored_as_filtered_with_reason(workspace):
    scan(workspace, user_ids=["u1"])
    _, _, scan_store = workspace
    filtered = scan_store.list_events(user_id="u1", statuses=[EVENT_STATUS_FILTERED])
    assert filtered, "suppressed events must remain visible for audit"
    for event in filtered:
        assert event["grade"] == GRADE_P2
        assert event["reasons"], "every filtered card must state why it was dropped"


def test_pushable_cards_exclude_p2(workspace):
    scan(workspace, user_ids=["u1"])
    _, _, scan_store = workspace
    pushable = scan_store.list_events(user_id="u1", statuses=[EVENT_STATUS_PENDING])
    assert pushable
    assert all(item["grade"] in (GRADE_P0, GRADE_P1) for item in pushable)


def test_one_user_cannot_read_another_users_events(workspace):
    scan(workspace, user_ids=["u1", "u2"])
    _, _, scan_store = workspace
    assert scan_store.list_events(user_id="u1")
    assert scan_store.list_events(user_id="u2") == []
    assert scan_store.list_events(user_id="u3") == []


def test_since_window_excludes_older_artifacts(workspace):
    _, portfolio, _ = workspace
    output_dir = workspace[0]
    summary = run_daily_scan(
        output_dir=output_dir,
        portfolio_store=portfolio,
        scan_store=SQLiteDailyScanStore(workspace[2].db_path),
        user_ids=["u1"],
        since="2026-09-14",
        scan_date=SCAN_DATE,
    )
    _, _, scan_store = workspace
    events = scan_store.list_events(user_id="u1")
    assert summary["events_inserted"] > 0
    assert all(item["date"] >= "2026-09-14" for item in events)


def test_run_leaves_a_queryable_watermark(workspace):
    summary = scan(workspace, user_ids=["u1"])
    _, _, scan_store = workspace
    runs = scan_store.list_runs(user_id="u1")
    assert len(runs) == 1
    assert runs[0]["scan_date"] == SCAN_DATE
    assert runs[0]["events_inserted"] == summary["events_inserted"]
    assert runs[0]["grade_counts"], "counts must record what the run saw"


def test_second_run_appends_its_own_watermark(workspace):
    scan(workspace, user_ids=["u1"])
    scan(workspace, user_ids=["u1"])
    _, _, scan_store = workspace
    runs = scan_store.list_runs(user_id="u1")
    assert len(runs) == 2, "history must show that the job ran twice"


def test_summary_reports_per_user_breakdown(workspace):
    summary = scan(workspace, user_ids=["u1", "u2"])
    assert summary["users_total"] == 2
    assert summary["per_user"][0]["user_id"] == "u1"
    assert summary["per_user"][1]["symbols_total"] == 1
    assert set(summary["grade_counts"]) == {GRADE_P0, GRADE_P1, GRADE_P2}
