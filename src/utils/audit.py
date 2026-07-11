"""审计日志模型与记录器

⚡ 字段统一：AuditEntry 模型以 SCHEMA-REFERENCE §3.5 为权威，
本文件对应设计 impl-06 §6.1 中的 src/utils/audit.py。
"""

import json
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

from src.agents.state import AssistantState
from src.schemas.constants import (
    AUDIT_DB_PATH,
    AUDIT_QUERY_ORIGINAL,
    AUDIT_REQUEST_ID,
    AUDIT_RESPONSE_CONFIDENCE,
    AUDIT_STARTED_PERF_COUNTER,
    AUDIT_TIMESTAMP,
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
    STATE_RETRIEVAL_FILTERED_CHUNKS,
    STATE_RETRIEVAL_PLAN,
    STATE_RETRIEVAL_RESULTS,
    STATE_RETRIEVAL_TOTAL_CHUNKS,
    STATE_REWRITTEN_QUERY,
    STATE_RISK_DISCLOSURE,
    STATE_TOOL_CALLS,
    STATE_USER_ID,
    STATE_USER_ROLE,
    STATE_VERIFICATION,
)
from src.schemas.models import AuditEntry
from src.schemas.typed_dicts import (
    AuditQuery,
    AuditReasoning,
    AuditResponse,
    AuditRetrieval,
    AuditTrail,
    RetrievalResult,
)

def audit_entry_to_trail(entry: AuditEntry) -> AuditTrail:
    """Serialize an AuditEntry into the AssistantState audit_trail shape."""
    return AuditTrail(
        request_id=entry.request_id,
        timestamp=entry.timestamp,
        user_id=entry.user_id,
        user_role=entry.user_role,
        department=entry.department,
        query=entry.query,
        retrieval=entry.retrieval,
        reasoning=entry.reasoning,
        verification=entry.verification,
        compliance=entry.compliance,
        response=entry.response,
        total_duration_ms=entry.total_duration_ms,
    )


def _unique_sources(results: list[RetrievalResult]) -> list[str]:
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
        node_timings = state.get(STATE_INTERMEDIATE_STEPS, [])
        reason_timings = [step for step in node_timings if step.get("step") == "reason"]

        return AuditEntry(
            request_id=request_id,
            timestamp=timestamp,
            user_id=state.get(STATE_USER_ID, ""),
            user_role=state.get(STATE_USER_ROLE, ""),
            department=state.get(STATE_DEPARTMENT, ""),
            query=AuditQuery(
                original=state.get(STATE_ORIGINAL_QUERY, ""),
                rewritten=state.get(STATE_REWRITTEN_QUERY, ""),
                intent=state.get(STATE_INTENT, ""),
                query_type=state.get(STATE_QUERY_TYPE, ""),
                entities=state.get(STATE_ENTITIES, {}),
            ),
            retrieval=AuditRetrieval(
                plan=state.get(STATE_RETRIEVAL_PLAN, []),
                sources=_unique_sources(state.get(STATE_RETRIEVAL_RESULTS, [])),
                total_chunks=state.get(
                    STATE_RETRIEVAL_TOTAL_CHUNKS,
                    len(state.get(STATE_RETRIEVAL_RESULTS, [])),
                ),
                filtered_chunks=state.get(
                    STATE_RETRIEVAL_FILTERED_CHUNKS,
                    len(state.get(STATE_RETRIEVAL_RESULTS, [])),
                ),
            ),
            reasoning=AuditReasoning(
                tool_calls=state.get(STATE_TOOL_CALLS, []),
                iterations=len(reason_timings),
                duration_ms=sum(float(step.get("duration_ms", 0.0)) for step in reason_timings),
                execution_path=[str(step.get("step", "")) for step in node_timings]
                + ["audit_log"],
                node_timings=node_timings,
            ),
            verification=state.get(STATE_VERIFICATION, {}),
            compliance=state.get(STATE_COMPLIANCE, {}),
            response=AuditResponse(
                citations=state.get(STATE_CITATIONS, []),
                confidence=state.get(STATE_CONFIDENCE, CONFIDENCE_LOW),
                risk_disclosure=state.get(STATE_RISK_DISCLOSURE, ""),
            ),
            total_duration_ms=self._calculate_duration_ms(timestamp, started_perf_counter),
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
                    entry.request_id,
                    entry.timestamp,
                    entry.user_id,
                    entry.user_role,
                    entry.department,
                    entry.query.get(AUDIT_QUERY_ORIGINAL, ""),
                    self._bool_to_int(entry.compliance.get("passed")),
                    entry.response.get(AUDIT_RESPONSE_CONFIDENCE, CONFIDENCE_LOW),
                    json.dumps(payload, ensure_ascii=False, sort_keys=True),
                ),
            )

    def get_by_request_id(self, request_id: str) -> AuditTrail | None:
        with sqlite3.connect(str(self.db_path), timeout=5) as conn:
            self._ensure_schema(conn)
            row = conn.execute(
                "SELECT payload_json FROM audit_entries WHERE request_id = ?",
                (request_id,),
            ).fetchone()
        if row is None:
            return None
        return cast(AuditTrail, json.loads(row[0]))

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

    def _to_payload(self, entry: AuditEntry) -> AuditTrail:
        return audit_entry_to_trail(entry)

    def _bool_to_int(self, value: object) -> int | None:
        if value is None:
            return None
        return int(bool(value))
