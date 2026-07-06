"""审计日志模型与记录器

⚡ 字段统一：AuditEntry 模型以 SCHEMA-REFERENCE §3.3 为权威，
本文件对应设计 impl-06 §6.1 中的 src/utils/audit.py。
"""

import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from src.agents.state import AssistantState
from src.schemas.constants import (
    AUDIT_QUERY_ENTITIES,
    AUDIT_QUERY_INTENT,
    AUDIT_QUERY_ORIGINAL,
    AUDIT_QUERY_REWRITTEN,
    AUDIT_QUERY_TYPE,
    AUDIT_REASONING_DURATION_MS,
    AUDIT_REASONING_ITERATIONS,
    AUDIT_REASONING_TOOL_CALLS,
    AUDIT_REQUEST_ID,
    AUDIT_RESPONSE_CITATIONS,
    AUDIT_RESPONSE_CONFIDENCE,
    AUDIT_RESPONSE_RISK_DISCLOSURE,
    AUDIT_RETRIEVAL_FILTERED_CHUNKS,
    AUDIT_RETRIEVAL_PLAN,
    AUDIT_RETRIEVAL_SOURCES,
    AUDIT_RETRIEVAL_TOTAL_CHUNKS,
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

    def _calculate_duration_ms(self, timestamp: str, started_perf_counter: object = None) -> float:
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
