"""审计日志模型与记录器

⚡ 字段统一：AuditEntry 模型以 SCHEMA-REFERENCE §3.3 为权威，
本文件对应设计 impl-06 §6.1 中的 src/utils/audit.py。
"""

import json
import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from src.agents.state import AssistantState
from src.schemas.constants import (
    AUDIT_COMPLIANCE,
    AUDIT_DB_PATH,
    AUDIT_QUERY,
    AUDIT_QUERY_ENTITIES,
    AUDIT_QUERY_INTENT,
    AUDIT_QUERY_ORIGINAL,
    AUDIT_QUERY_REWRITTEN,
    AUDIT_QUERY_TYPE,
    AUDIT_REASONING,
    AUDIT_REASONING_DURATION_MS,
    AUDIT_REASONING_ITERATIONS,
    AUDIT_REASONING_TOOL_CALLS,
    AUDIT_REQUEST_ID,
    AUDIT_RESPONSE,
    AUDIT_RESPONSE_CITATIONS,
    AUDIT_RESPONSE_CONFIDENCE,
    AUDIT_RESPONSE_RISK_DISCLOSURE,
    AUDIT_RETRIEVAL,
    AUDIT_RETRIEVAL_FILTERED_CHUNKS,
    AUDIT_RETRIEVAL_PLAN,
    AUDIT_RETRIEVAL_SOURCES,
    AUDIT_RETRIEVAL_TOTAL_CHUNKS,
    AUDIT_STARTED_PERF_COUNTER,
    AUDIT_TIMESTAMP,
    AUDIT_VERIFICATION,
    CONFIDENCE_LOW,
    META_SOURCE,
    RR_METADATA,
    STATE_AUDIT_TRAIL,
    STATE_CITATIONS,
    STATE_COMPLIANCE,
    STATE_CONFIDENCE,
    STATE_DEPARTMENT,
    STATE_ENTITIES,
    STATE_INTENT,
    STATE_INTERMEDIATE_STEPS,
    STATE_ORIGINAL_QUERY,
    STATE_QUERY_TYPE,
    STATE_RETRIEVAL_PLAN,
    STATE_RETRIEVAL_RESULTS,
    STATE_REWRITTEN_QUERY,
    STATE_RISK_DISCLOSURE,
    STATE_TOOL_CALLS,
    STATE_USER_ID,
    STATE_USER_ROLE,
    STATE_VERIFICATION,
)


@dataclass
class AuditEntry:
    """审计日志条目 — 权威定义，字段对应 SCHEMA-REFERENCE §3.3"""

    request_id: str
    timestamp: str
    user_id: str
    user_role: str
    department: str
    query: dict
    retrieval: dict
    reasoning: dict
    verification: dict
    compliance: dict
    response: dict


def _unique_sources(results: list[dict]) -> list[str]:
    sources = []
    seen = set()
    for result in results:
        source = result.get(RR_METADATA, {}).get(META_SOURCE)
        if not source or source in seen:
            continue
        seen.add(source)
        sources.append(source)
    return sources


class AuditLogger:
    """构建全链路审计条目（Query → Retrieve → Reason → Verify → Compose）"""

    def log(self, state: AssistantState) -> AuditEntry:
        """记录完整审计日志"""
        audit_trail = state.get(STATE_AUDIT_TRAIL, {})
        request_id = audit_trail.get(AUDIT_REQUEST_ID) or str(uuid.uuid4())
        timestamp = audit_trail.get(AUDIT_TIMESTAMP) or datetime.now(timezone.utc).isoformat()
        started_perf_counter = audit_trail.get(AUDIT_STARTED_PERF_COUNTER)

        return AuditEntry(
            request_id=request_id,
            timestamp=timestamp,
            user_id=state.get(STATE_USER_ID, ""),
            user_role=state.get(STATE_USER_ROLE, ""),
            department=state.get(STATE_DEPARTMENT, ""),
            query={
                AUDIT_QUERY_ORIGINAL: state.get(STATE_ORIGINAL_QUERY, ""),
                AUDIT_QUERY_REWRITTEN: state.get(STATE_REWRITTEN_QUERY, ""),
                AUDIT_QUERY_INTENT: state.get(STATE_INTENT, ""),
                AUDIT_QUERY_TYPE: state.get(STATE_QUERY_TYPE, ""),
                AUDIT_QUERY_ENTITIES: state.get(STATE_ENTITIES, {}),
            },
            retrieval={
                AUDIT_RETRIEVAL_PLAN: state.get(STATE_RETRIEVAL_PLAN, []),
                AUDIT_RETRIEVAL_SOURCES: _unique_sources(state.get(STATE_RETRIEVAL_RESULTS, [])),
                AUDIT_RETRIEVAL_TOTAL_CHUNKS: len(state.get(STATE_RETRIEVAL_RESULTS, [])),
                AUDIT_RETRIEVAL_FILTERED_CHUNKS: len(state.get(STATE_RETRIEVAL_RESULTS, [])),
            },
            reasoning={
                AUDIT_REASONING_TOOL_CALLS: state.get(STATE_TOOL_CALLS, []),
                AUDIT_REASONING_ITERATIONS: len(state.get(STATE_INTERMEDIATE_STEPS, [])),
                AUDIT_REASONING_DURATION_MS: self._calculate_duration_ms(
                    timestamp,
                    started_perf_counter,
                ),
            },
            verification=state.get(STATE_VERIFICATION, {}),
            compliance=state.get(STATE_COMPLIANCE, {}),
            response={
                AUDIT_RESPONSE_CITATIONS: state.get(STATE_CITATIONS, []),
                AUDIT_RESPONSE_CONFIDENCE: state.get(STATE_CONFIDENCE, CONFIDENCE_LOW),
                AUDIT_RESPONSE_RISK_DISCLOSURE: state.get(STATE_RISK_DISCLOSURE, ""),
            },
        )

    def _calculate_duration_ms(
        self, timestamp: str, started_perf_counter: float | int | str | None = None
    ) -> float:
        if started_perf_counter is not None:
            try:
                return max((time.perf_counter() - float(started_perf_counter)) * 1000, 0.0)
            except (TypeError, ValueError):
                pass

        try:
            started_at = datetime.fromisoformat(timestamp)
        except ValueError:
            return 0.0

        if started_at.tzinfo is None:
            started_at = started_at.replace(tzinfo=timezone.utc)

        duration_ms = (datetime.now(timezone.utc) - started_at).total_seconds() * 1000
        return max(duration_ms, 0.0)


class SQLiteAuditStore:
    """SQLite-backed audit store for queryable compliance trace records."""

    def __init__(self, db_path: str | Path = AUDIT_DB_PATH):
        self.db_path = Path(db_path)

    def insert(self, entry: AuditEntry) -> None:
        payload = self._to_payload(entry)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(self.db_path), timeout=5) as conn:
            self._ensure_schema(conn)
            conn.execute(
                """
                INSERT OR REPLACE INTO audit_entries (
                    request_id,
                    timestamp,
                    user_id,
                    user_role,
                    department,
                    query_text,
                    compliance_passed,
                    confidence,
                    payload_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload[AUDIT_REQUEST_ID],
                    payload[AUDIT_TIMESTAMP],
                    payload[STATE_USER_ID],
                    payload[STATE_USER_ROLE],
                    payload[STATE_DEPARTMENT],
                    payload[AUDIT_QUERY].get(AUDIT_QUERY_ORIGINAL, ""),
                    self._bool_to_int(payload[AUDIT_COMPLIANCE].get("passed")),
                    payload[AUDIT_RESPONSE].get(AUDIT_RESPONSE_CONFIDENCE, CONFIDENCE_LOW),
                    json.dumps(payload, ensure_ascii=False, sort_keys=True),
                ),
            )

    def get_by_request_id(self, request_id: str) -> dict | None:
        with sqlite3.connect(str(self.db_path), timeout=5) as conn:
            self._ensure_schema(conn)
            row = conn.execute(
                "SELECT payload_json FROM audit_entries WHERE request_id = ?",
                (request_id,),
            ).fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    def _ensure_schema(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_entries (
                request_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                user_id TEXT NOT NULL,
                user_role TEXT NOT NULL,
                department TEXT NOT NULL,
                query_text TEXT NOT NULL,
                compliance_passed INTEGER,
                confidence TEXT NOT NULL,
                payload_json TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_audit_entries_timestamp ON audit_entries(timestamp)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_audit_entries_user_role ON audit_entries(user_role)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_audit_entries_compliance ON audit_entries(compliance_passed)"
        )

    def _to_payload(self, entry: AuditEntry) -> dict:
        payload = asdict(entry)
        return {
            AUDIT_REQUEST_ID: payload[AUDIT_REQUEST_ID],
            AUDIT_TIMESTAMP: payload[AUDIT_TIMESTAMP],
            STATE_USER_ID: payload[STATE_USER_ID],
            STATE_USER_ROLE: payload[STATE_USER_ROLE],
            STATE_DEPARTMENT: payload[STATE_DEPARTMENT],
            AUDIT_QUERY: payload[AUDIT_QUERY],
            AUDIT_RETRIEVAL: payload[AUDIT_RETRIEVAL],
            AUDIT_REASONING: payload[AUDIT_REASONING],
            AUDIT_VERIFICATION: payload[AUDIT_VERIFICATION],
            AUDIT_COMPLIANCE: payload[AUDIT_COMPLIANCE],
            AUDIT_RESPONSE: payload[AUDIT_RESPONSE],
        }

    def _bool_to_int(self, value: object) -> int | None:
        if value is None:
            return None
        return int(bool(value))
