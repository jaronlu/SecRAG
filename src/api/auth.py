from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Annotated, cast

from fastapi import Header, HTTPException, status

from src.agents.state import AssistantState
from src.schemas.constants import (
    AUDIT_REQUEST_ID,
    AUDIT_STARTED_PERF_COUNTER,
    AUDIT_TIMESTAMP,
    ROLE_ADVISOR,
    ROLE_COMPLIANCE,
    ROLE_DATA_PERMISSIONS,
    ROLE_INSTITUTIONAL_SALES,
    ROLE_OPERATIONS,
    ROLE_TECHNICAL,
    STATE_AMBIGUITY,
    STATE_AUDIT_TRAIL,
    STATE_CITATIONS,
    STATE_CHAT_HISTORY,
    STATE_CLIENT_ID,
    STATE_COMPLIANCE,
    STATE_CONVERSATION_SUMMARY,
    STATE_CONFIDENCE,
    STATE_DATA_PERMISSIONS,
    STATE_DEPARTMENT,
    STATE_ENTITIES,
    STATE_FINAL_ANSWER,
    STATE_INTENT,
    STATE_INTERMEDIATE_STEPS,
    STATE_MESSAGES,
    STATE_ORIGINAL_QUERY,
    STATE_QUERY_TYPE,
    STATE_RESOLVED_QUERY,
    STATE_REASON_ATTEMPTS,
    STATE_RETRIEVAL_PLAN,
    STATE_RETRIEVAL_ATTEMPTS,
    STATE_RETRIEVAL_FILTERED_CHUNKS,
    STATE_RETRIEVAL_RESULTS,
    STATE_RETRIEVAL_TOTAL_CHUNKS,
    STATE_REWRITTEN_QUERY,
    STATE_RISK_DISCLOSURE,
    STATE_THREAD_ID,
    STATE_TOOL_CALLS,
    STATE_TURN_ID,
    STATE_TURN_INDEX,
    STATE_USER_ID,
    STATE_USER_ROLE,
    STATE_VERIFICATION,
)
from src.schemas.request_response import AssistantQARequest


@dataclass(frozen=True)
class AuthenticatedUser:
    user_id: str
    role: str
    department: str


TOKEN_USER_BINDINGS: dict[str, AuthenticatedUser] = {
    "demo-advisor": AuthenticatedUser("user_advisor", ROLE_ADVISOR, "wealth"),
    "demo-sales": AuthenticatedUser("user_sales", ROLE_INSTITUTIONAL_SALES, "sales"),
    "demo-compliance": AuthenticatedUser("user_compliance", ROLE_COMPLIANCE, "control"),
    "demo-ops": AuthenticatedUser("user_ops", ROLE_OPERATIONS, "ops"),
    "demo-tech": AuthenticatedUser("user_tech", ROLE_TECHNICAL, "tech"),
}


def authenticate_user(
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
) -> AuthenticatedUser:
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing bearer token",
        )

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid authorization header",
        )

    user = TOKEN_USER_BINDINGS.get(token)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="unknown demo token",
        )
    return user


def build_assistant_initial_state(
    request: AssistantQARequest,
    user: AuthenticatedUser,
    *,
    thread_id: str | None = None,
    turn_id: str | None = None,
    request_id: str | None = None,
    turn_index: int = 0,
) -> AssistantState:
    data_permissions = ROLE_DATA_PERMISSIONS.get(user.role, ["public"])
    effective_thread_id = thread_id or request.thread_id or str(uuid.uuid4())
    effective_turn_id = turn_id or str(uuid.uuid4())
    effective_request_id = request_id or str(uuid.uuid4())

    return cast(AssistantState, {
        STATE_USER_ID: user.user_id,
        STATE_USER_ROLE: user.role,
        STATE_DEPARTMENT: user.department,
        STATE_DATA_PERMISSIONS: data_permissions,
        STATE_CLIENT_ID: request.client_id,
        STATE_THREAD_ID: effective_thread_id,
        STATE_TURN_ID: effective_turn_id,
        STATE_TURN_INDEX: turn_index,
        STATE_CHAT_HISTORY: [],
        STATE_CONVERSATION_SUMMARY: "",
        STATE_RESOLVED_QUERY: "",
        STATE_ORIGINAL_QUERY: request.query,
        STATE_REWRITTEN_QUERY: "",
        STATE_INTENT: "",
        STATE_ENTITIES: {},
        STATE_AMBIGUITY: [],
        STATE_QUERY_TYPE: "",
        STATE_RETRIEVAL_ATTEMPTS: 0,
        STATE_RETRIEVAL_PLAN: [],
        STATE_RETRIEVAL_RESULTS: [],
        STATE_RETRIEVAL_TOTAL_CHUNKS: 0,
        STATE_RETRIEVAL_FILTERED_CHUNKS: 0,
        STATE_MESSAGES: [],
        STATE_TOOL_CALLS: [],
        STATE_INTERMEDIATE_STEPS: [],
        STATE_REASON_ATTEMPTS: 0,
        STATE_VERIFICATION: {},
        STATE_COMPLIANCE: {},
        STATE_FINAL_ANSWER: "",
        STATE_CITATIONS: [],
        STATE_CONFIDENCE: "low",
        STATE_RISK_DISCLOSURE: "",
        STATE_AUDIT_TRAIL: {
            AUDIT_REQUEST_ID: effective_request_id,
            AUDIT_TIMESTAMP: datetime.now(timezone.utc).isoformat(),
            AUDIT_STARTED_PERF_COUNTER: time.perf_counter(),
        },
    })
