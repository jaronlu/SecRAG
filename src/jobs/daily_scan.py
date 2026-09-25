"""Daily incremental scan over artifacts produced by the T-001 fetch batch.

The job turns raw artifacts into per-user event cards. Three properties matter
and are covered by tests:

- **Idempotent.** The dedupe key is derived from the *event identity*
  (user, symbol, source kind, reference, title, date) plus the current
  `rule_version` rather than the run it was seen in, so re-running the same
  trade day — or any later day — never produces a second card for the same
  event. The key deliberately excludes `scan_date` because a duplicate card in
  tomorrow's briefing is worse than no card. Retuning the grading thresholds
  changes the rule version, which re-opens the event under the new rules
  instead of silently keeping the grade computed earlier (P2-2).
- **Justified.** Every card carries the grading reasons and rule ids, including
  the P2 cards that get filtered out, so suppression is auditable rather than
  invisible.
- **Watermarked.** Each run persists a row with per-grade counts so tomorrow's
  run can be compared against yesterday's instead of silently drifting.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import sqlite3
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from typing_extensions import TypedDict

from src.jobs.event_grading import (
    DEFAULT_THRESHOLDS,
    GRADE_P0,
    GRADE_P1,
    GRADE_P2,
    PUSHABLE_GRADES,
    GradingDecisionDict,
    GradingThresholds,
    grade_document,
    grade_quote_move,
    grade_research_report,
)
from src.portfolio.store import SQLitePortfolioStore
from src.schemas.constants import META_DATE, META_STOCK_CODE, META_TITLE

QUOTE_FILE_GLOB = "efinance_*_quote_history.csv"
RESEARCH_INDEX_RELATIVE_PATH = "financials/research_reports_index.csv"
METADATA_SUFFIX = ".meta.json"

SOURCE_DOCUMENT = "document"
SOURCE_QUOTE = "quote"
SOURCE_RESEARCH = "research"

EVENT_STATUS_PENDING = "pending"
EVENT_STATUS_FILTERED = "filtered"

_SYMBOL_IN_NAME = re.compile(r"(\d{6})")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def rule_version_for(thresholds: GradingThresholds) -> str:
    """Content-derived stamp of the grading rules (P2-2).

    版本直接由阈值内容哈希派生：任何调参都会自动改变版本，从而让同一事件
    在新规则下重新分级并入库，而不是被旧规则算出的记录静默吞掉。
    """
    payload = json.dumps(dataclasses.asdict(thresholds), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def dedupe_key(
    *,
    user_id: str,
    stock_code: str,
    source_kind: str,
    source_ref: str,
    title: str,
    date: str,
    rule_version: str,
) -> str:
    """Stable identity of an event for one user, independent of the run that found it.

    rule_version 参与哈希（P2-2）：阈值/规则变更后同一事件生成新键，
    新等级得以落库；旧版本的记录保持原样以供审计。
    """
    raw = "|".join((user_id, stock_code, source_kind, source_ref, title, date, rule_version))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class CandidateDict(TypedDict):
    source_kind: str
    stock_code: str
    date: str
    title: str
    source_ref: str
    payload: dict[str, Any]


class ScanEventDict(TypedDict):
    event_id: str
    user_id: str
    stock_code: str
    source_kind: str
    grade: str
    status: str
    title: str
    date: str
    source_ref: str
    reasons: list[str]
    matched_rules: list[str]
    dedupe_key: str
    first_seen_at: str
    created_at: str


class ScanRunSummaryDict(TypedDict):
    run_id: str
    scan_date: str
    users_total: int
    events_inserted: int
    duplicates_skipped: int
    grade_counts: dict[str, int]
    started_at: str
    finished_at: str
    per_user: list[dict[str, Any]]


def _dump_list(values: Sequence[str]) -> str:
    return json.dumps(list(values), ensure_ascii=False)


def _load_list(raw: Any) -> list[str]:
    if not raw:
        return []
    try:
        parsed = json.loads(str(raw))
    except json.JSONDecodeError:
        return []
    return [str(item) for item in parsed] if isinstance(parsed, list) else []


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    frame = pd.read_csv(path, dtype=str).fillna("")
    return [dict(row) for row in frame.to_dict(orient="records")]


def _within_window(date_value: str, since: str | None) -> bool:
    if not since:
        return True
    return bool(date_value) and str(date_value)[:10] >= since


def _symbol_from_filename(path: Path) -> str:
    match = _SYMBOL_IN_NAME.search(path.stem)
    return match.group(1) if match else ""


def load_documents(output_dir: Path, *, since: str | None = None) -> list[CandidateDict]:
    """Load document side-car metadata written by the fetch batch."""
    candidates: list[CandidateDict] = []
    for meta_path in sorted(output_dir.rglob(f"*{METADATA_SUFFIX}")):
        try:
            record = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(record, Mapping):
            continue
        stock_code = str(record.get(META_STOCK_CODE) or "")
        if not stock_code:
            continue
        date = str(record.get(META_DATE) or "")
        if not _within_window(date, since):
            continue
        artifact = meta_path.with_name(meta_path.name[: -len(METADATA_SUFFIX)])
        candidates.append(
            {
                "source_kind": SOURCE_DOCUMENT,
                "stock_code": stock_code,
                "date": date,
                "title": str(record.get(META_TITLE) or artifact.name),
                "source_ref": str(artifact.relative_to(output_dir)),
                "payload": dict(record),
            }
        )
    return candidates


def load_quote_moves(output_dir: Path, *, since: str | None = None) -> list[CandidateDict]:
    """Load daily price moves from per-symbol quote history CSVs."""
    candidates: list[CandidateDict] = []
    for csv_path in sorted(output_dir.rglob(QUOTE_FILE_GLOB)):
        stock_code = _symbol_from_filename(csv_path)
        if not stock_code:
            continue
        for row in _read_csv_rows(csv_path):
            date = str(row.get("date") or "")
            if not _within_window(date, since):
                continue
            candidates.append(
                {
                    "source_kind": SOURCE_QUOTE,
                    "stock_code": stock_code,
                    "date": date[:10],
                    "title": f"{stock_code} {date[:10]} 行情波动",
                    "source_ref": str(csv_path.relative_to(output_dir)),
                    "payload": row,
                }
            )
    return candidates


def load_research_rows(output_dir: Path, *, since: str | None = None) -> list[CandidateDict]:
    """Load research report rows from the shared index CSV."""
    index_path = output_dir / RESEARCH_INDEX_RELATIVE_PATH
    if not index_path.exists():
        return []
    candidates: list[CandidateDict] = []
    for row in _read_csv_rows(index_path):
        date = str(row.get("日期") or "")
        if not _within_window(date, since):
            continue
        stock_code = str(row.get("sample_stock_code") or row.get("stock_code") or "")
        if not stock_code:
            continue
        candidates.append(
            {
                "source_kind": SOURCE_RESEARCH,
                "stock_code": stock_code,
                "date": date[:10],
                "title": str(row.get("报告名称") or "研究报告"),
                "source_ref": str(row.get("报告PDF链接") or index_path.name),
                "payload": row,
            }
        )
    return candidates


def collect_candidates(output_dir: Path, *, since: str | None = None) -> list[CandidateDict]:
    """Every scannable candidate across documents, quotes and research reports."""
    return [
        *load_documents(output_dir, since=since),
        *load_quote_moves(output_dir, since=since),
        *load_research_rows(output_dir, since=since),
    ]


def grade_candidate(
    candidate: CandidateDict,
    *,
    thresholds: GradingThresholds = DEFAULT_THRESHOLDS,
) -> GradingDecisionDict:
    """Dispatch a candidate to the grader that matches its source kind."""
    payload = candidate["payload"]
    if candidate["source_kind"] == SOURCE_QUOTE:
        return grade_quote_move(payload, thresholds=thresholds)
    if candidate["source_kind"] == SOURCE_RESEARCH:
        return grade_research_report(payload, thresholds=thresholds)
    return grade_document(payload, thresholds=thresholds)


def build_events(
    *,
    user_id: str,
    candidates: Sequence[CandidateDict],
    symbols: Sequence[str],
    scan_date: str,
    thresholds: GradingThresholds = DEFAULT_THRESHOLDS,
) -> list[ScanEventDict]:
    """Grade every candidate matching the user's symbols into event cards.

    P2 cards are built too, marked `filtered`; downstream briefing reads only
    the pushable grades. Keeping them stored is what makes suppression traceable.
    """
    wanted = {str(symbol).strip() for symbol in symbols if str(symbol).strip()}
    events: list[ScanEventDict] = []
    version = rule_version_for(thresholds)
    for candidate in candidates:
        if candidate["stock_code"] not in wanted:
            continue
        decision = grade_candidate(candidate, thresholds=thresholds)
        now = utc_now()
        events.append(
            {
                "event_id": str(uuid.uuid4()),
                "user_id": user_id,
                "stock_code": candidate["stock_code"],
                "source_kind": candidate["source_kind"],
                "grade": decision["grade"],
                "status": (
                    EVENT_STATUS_PENDING
                    if decision["grade"] in PUSHABLE_GRADES
                    else EVENT_STATUS_FILTERED
                ),
                "title": candidate["title"],
                "date": candidate["date"],
                "source_ref": candidate["source_ref"],
                "reasons": decision["reasons"],
                "matched_rules": decision["matched_rules"],
                "dedupe_key": dedupe_key(
                    user_id=user_id,
                    stock_code=candidate["stock_code"],
                    source_kind=candidate["source_kind"],
                    source_ref=candidate["source_ref"],
                    title=candidate["title"],
                    date=candidate["date"],
                    rule_version=version,
                ),
                "first_seen_at": scan_date,
                "created_at": now,
            }
        )
    return events


class SQLiteDailyScanStore:
    """Per-user event card persistence with run watermarks."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)

    def _ensure_schema(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS daily_scan_events (
                event_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                stock_code TEXT NOT NULL,
                source_kind TEXT NOT NULL,
                grade TEXT NOT NULL,
                status TEXT NOT NULL,
                title TEXT NOT NULL,
                date TEXT NOT NULL,
                source_ref TEXT NOT NULL,
                reasons TEXT NOT NULL,
                matched_rules TEXT NOT NULL,
                dedupe_key TEXT NOT NULL,
                first_seen_at TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_scan_events_dedupe
            ON daily_scan_events (dedupe_key)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_scan_events_user_date
            ON daily_scan_events (user_id, first_seen_at)
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS daily_scan_runs (
                run_id TEXT PRIMARY KEY,
                scan_date TEXT NOT NULL,
                user_id TEXT NOT NULL,
                symbols_total INTEGER NOT NULL,
                events_inserted INTEGER NOT NULL,
                duplicates_skipped INTEGER NOT NULL,
                grade_counts TEXT NOT NULL,
                started_at TEXT NOT NULL,
                finished_at TEXT NOT NULL
            )
            """
        )

    def record_events(
        self, *, user_id: str, scan_date: str, events: Sequence[ScanEventDict]
    ) -> dict[str, int]:
        """Insert unseen cards only; return how many were inserted versus skipped."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        inserted = 0
        with sqlite3.connect(str(self.db_path), timeout=5) as conn:
            conn.row_factory = sqlite3.Row
            self._ensure_schema(conn)
            for event in events:
                if event["user_id"] != user_id:
                    continue
                cursor = conn.execute(
                    """
                    INSERT OR IGNORE INTO daily_scan_events (
                        event_id, user_id, stock_code, source_kind, grade, status,
                        title, date, source_ref, reasons, matched_rules,
                        dedupe_key, first_seen_at, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event["event_id"],
                        event["user_id"],
                        event["stock_code"],
                        event["source_kind"],
                        event["grade"],
                        event["status"],
                        event["title"],
                        event["date"],
                        event["source_ref"],
                        _dump_list(event["reasons"]),
                        _dump_list(event["matched_rules"]),
                        event["dedupe_key"],
                        event["first_seen_at"],
                        event["created_at"],
                    ),
                )
                inserted += int(cursor.rowcount or 0)
        return {"inserted": inserted, "duplicates_skipped": len(events) - inserted}

    def list_events(
        self,
        *,
        user_id: str,
        scan_date: str | None = None,
        grades: Sequence[str] | None = None,
        statuses: Sequence[str] | None = None,
    ) -> list[ScanEventDict]:
        """List one user's cards, optionally narrowed by day, grade or status."""
        clauses = ["user_id = ?"]
        params: list[object] = [user_id]
        if scan_date is not None:
            clauses.append("first_seen_at = ?")
            params.append(scan_date)
        if grades is not None:
            clauses.append(f"grade IN ({','.join('?' * len(grades))})")
            params.extend(grades)
        if statuses is not None:
            clauses.append(f"status IN ({','.join('?' * len(statuses))})")
            params.extend(statuses)
        where = " AND ".join(clauses)

        rows: list[ScanEventDict] = []
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(self.db_path), timeout=5) as conn:
            conn.row_factory = sqlite3.Row
            self._ensure_schema(conn)
            cursor = conn.execute(
                f"SELECT * FROM daily_scan_events WHERE {where} ORDER BY date DESC, created_at DESC",
                params,
            )
            for row in cursor.fetchall():
                rows.append(self._to_event(dict(row)))
        return rows

    def record_run(
        self, summary: Mapping[str, Any], *, per_user: Sequence[Mapping[str, Any]]
    ) -> None:
        """Persist one row per user so each run leaves a queryable watermark."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(self.db_path), timeout=5) as conn:
            self._ensure_schema(conn)
            for entry in per_user:
                conn.execute(
                    """
                    INSERT INTO daily_scan_runs (
                        run_id, scan_date, user_id, symbols_total, events_inserted,
                        duplicates_skipped, grade_counts, started_at, finished_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        f"{summary['run_id']}:{entry['user_id']}",
                        summary["scan_date"],
                        entry["user_id"],
                        int(entry["symbols_total"]),
                        int(entry["events_inserted"]),
                        int(entry["duplicates_skipped"]),
                        json.dumps(entry["grade_counts"], ensure_ascii=False),
                        summary["started_at"],
                        summary["finished_at"],
                    ),
                )

    def list_runs(self, *, user_id: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        """Recent run watermarks, newest first."""
        clauses: list[str] = []
        params: list[object] = []
        if user_id is not None:
            clauses.append("user_id = ?")
            params.append(user_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(self.db_path), timeout=5) as conn:
            conn.row_factory = sqlite3.Row
            self._ensure_schema(conn)
            cursor = conn.execute(
                f"SELECT * FROM daily_scan_runs {where} ORDER BY started_at DESC LIMIT ?",
                [*params, limit],
            )
            rows = [dict(row) for row in cursor.fetchall()]
        for row in rows:
            row["grade_counts"] = json.loads(str(row.get("grade_counts") or "{}"))
        return rows

    @staticmethod
    def _to_event(row: Mapping[str, Any]) -> ScanEventDict:
        return {
            "event_id": str(row["event_id"]),
            "user_id": str(row["user_id"]),
            "stock_code": str(row["stock_code"]),
            "source_kind": str(row["source_kind"]),
            "grade": str(row["grade"]),
            "status": str(row["status"]),
            "title": str(row["title"]),
            "date": str(row["date"]),
            "source_ref": str(row["source_ref"]),
            "reasons": _load_list(row["reasons"]),
            "matched_rules": _load_list(row["matched_rules"]),
            "dedupe_key": str(row["dedupe_key"]),
            "first_seen_at": str(row["first_seen_at"]),
            "created_at": str(row["created_at"]),
        }


def run_daily_scan(
    *,
    output_dir: Path,
    portfolio_store: SQLitePortfolioStore,
    scan_store: SQLiteDailyScanStore,
    user_ids: Sequence[str],
    since: str | None = None,
    scan_date: str | None = None,
    thresholds: GradingThresholds = DEFAULT_THRESHOLDS,
) -> ScanRunSummaryDict:
    """Scan artifacts for each user's positions and persist newly seen cards."""
    started_at = utc_now()
    resolved_date = scan_date or started_at[:10]
    run_id = str(uuid.uuid4())
    candidates = collect_candidates(output_dir, since=since)

    per_user: list[dict[str, Any]] = []
    grade_counts = {GRADE_P0: 0, GRADE_P1: 0, GRADE_P2: 0}
    inserted_total = 0
    duplicates_total = 0

    for user_id in user_ids:
        # list_symbols_for_user already scopes to active positions for this user.
        symbols = portfolio_store.list_symbols_for_user(user_id=user_id)
        events = build_events(
            user_id=user_id,
            candidates=candidates,
            symbols=symbols,
            scan_date=resolved_date,
            thresholds=thresholds,
        )
        outcome = scan_store.record_events(user_id=user_id, scan_date=resolved_date, events=events)
        counts = {GRADE_P0: 0, GRADE_P1: 0, GRADE_P2: 0}
        for event in events:
            counts[event["grade"]] = counts.get(event["grade"], 0) + 1
            grade_counts[event["grade"]] = grade_counts.get(event["grade"], 0) + 1
        inserted_total += outcome["inserted"]
        duplicates_total += outcome["duplicates_skipped"]
        per_user.append(
            {
                "user_id": user_id,
                "symbols_total": len(symbols),
                "events_inserted": outcome["inserted"],
                "duplicates_skipped": outcome["duplicates_skipped"],
                "grade_counts": counts,
            }
        )

    summary: ScanRunSummaryDict = {
        "run_id": run_id,
        "scan_date": resolved_date,
        "users_total": len(user_ids),
        "events_inserted": inserted_total,
        "duplicates_skipped": duplicates_total,
        "grade_counts": grade_counts,
        "started_at": started_at,
        "finished_at": utc_now(),
        "per_user": per_user,
    }
    scan_store.record_run(summary, per_user=per_user)
    return summary
