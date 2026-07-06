from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Annotated

from fastapi import Header, HTTPException, status

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
    STATE_CLIENT_ID,
    STATE_COMPLIANCE,
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
    STATE_RETRIEVAL_PLAN,
    STATE_RETRIEVAL_RESULTS,
    STATE_REWRITTEN_QUERY,
    STATE_RISK_DISCLOSURE,
    STATE_TOOL_CALLS,
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
            status_code=status.HTTP_403_FORBIDDEN,
            detail="unknown demo token",
        )
    return user


def build_assistant_initial_state(
    request: AssistantQARequest,
    user: AuthenticatedUser,
) -> dict:
    data_permissions = ROLE_DATA_PERMISSIONS.get(user.role, ["public"])

    return {
        STATE_USER_ID: user.user_id,
        STATE_USER_ROLE: user.role,
        STATE_DEPARTMENT: user.department,
        STATE_DATA_PERMISSIONS: data_permissions,
        STATE_CLIENT_ID: request.client_id,
        STATE_ORIGINAL_QUERY: request.query,
        STATE_REWRITTEN_QUERY: "",
        STATE_INTENT: "",
        STATE_ENTITIES: {},
        STATE_AMBIGUITY: [],
        STATE_QUERY_TYPE: "",
        STATE_RETRIEVAL_PLAN: [],
        STATE_RETRIEVAL_RESULTS: [],
        STATE_MESSAGES: [],
        STATE_TOOL_CALLS: [],
        STATE_INTERMEDIATE_STEPS: [],
        STATE_VERIFICATION: {},
        STATE_COMPLIANCE: {},
        STATE_FINAL_ANSWER: "",
        STATE_CITATIONS: [],
        STATE_CONFIDENCE: "low",
        STATE_RISK_DISCLOSURE: "",
        STATE_AUDIT_TRAIL: {
            AUDIT_REQUEST_ID: str(uuid.uuid4()),
            AUDIT_TIMESTAMP: datetime.now(timezone.utc).isoformat(),
            AUDIT_STARTED_PERF_COUNTER: time.perf_counter(),
        },
    }
