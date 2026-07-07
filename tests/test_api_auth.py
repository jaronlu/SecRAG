import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from src.api.auth import authenticate_user, build_assistant_initial_state
from src.schemas.constants import (
    AUDIT_REQUEST_ID,
    AUDIT_STARTED_PERF_COUNTER,
    AUDIT_TIMESTAMP,
    ROLE_ALLOWED_SOURCES,
    ROLE_TECHNICAL,
    SOURCE_FAQ,
    SOURCE_REPORT,
    STATE_AUDIT_TRAIL,
    STATE_DATA_PERMISSIONS,
    STATE_DEPARTMENT,
    STATE_REASON_ATTEMPTS,
    STATE_RETRIEVAL_ATTEMPTS,
    STATE_USER_ID,
    STATE_USER_ROLE,
)
from src.schemas.request_response import AssistantQARequest


def test_authenticate_user_accepts_demo_token():
    user = authenticate_user("Bearer demo-tech")

    assert user.user_id == "user_tech"
    assert user.role == ROLE_TECHNICAL
    assert user.department == "tech"


@pytest.mark.parametrize("header", [None, "", "demo-tech", "Basic demo-tech"])
def test_authenticate_user_rejects_missing_or_malformed_header(header):
    with pytest.raises(HTTPException) as exc:
        authenticate_user(header)

    assert exc.value.status_code == 401


def test_authenticate_user_rejects_unknown_token():
    with pytest.raises(HTTPException) as exc:
        authenticate_user("Bearer missing-token")

    assert exc.value.status_code == 403


def test_build_initial_state_uses_authenticated_user():
    request = AssistantQARequest(query="查询")
    user = authenticate_user("Bearer demo-tech")

    state = build_assistant_initial_state(request, user)

    assert state[STATE_USER_ID] == "user_tech"
    assert state[STATE_USER_ROLE] == ROLE_TECHNICAL
    assert state[STATE_DEPARTMENT] == "tech"
    assert state[STATE_DATA_PERMISSIONS]
    audit_trail = state[STATE_AUDIT_TRAIL]
    assert audit_trail.get(AUDIT_REQUEST_ID)
    assert audit_trail.get(AUDIT_TIMESTAMP)
    assert audit_trail.get(AUDIT_STARTED_PERF_COUNTER, 0) > 0
    assert state[STATE_RETRIEVAL_ATTEMPTS] == 0
    assert state[STATE_REASON_ATTEMPTS] == 0
    assert SOURCE_FAQ in ROLE_ALLOWED_SOURCES[state[STATE_USER_ROLE]]
    assert SOURCE_REPORT not in ROLE_ALLOWED_SOURCES[state[STATE_USER_ROLE]]


def test_assistant_request_rejects_removed_identity_fields():
    with pytest.raises(ValidationError):
        AssistantQARequest.model_validate({
            "query": "查询",
            "user_id": "forged_user",
            "user_role": "advisor",
            "department": "forged_department",
        })
