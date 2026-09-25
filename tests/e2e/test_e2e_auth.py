"""全链路测试案例：环节 A 认证与接入（TC-001 / TC-002）。"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.auth import AuthenticatedUser, build_assistant_initial_state
from src.api.main import app
from src.schemas.constants import (
    API_ROUTE_ASSISTANT_QA,
    ROLE_ADVISOR,
    ROLE_COMPLIANCE,
    ROLE_DATA_PERMISSIONS,
    ROLE_INSTITUTIONAL_SALES,
    ROLE_OPERATIONS,
    ROLE_TECHNICAL,
    PERMISSION_CONFIDENTIAL,
    PERMISSION_INTERNAL,
    PERMISSION_PUBLIC,
    STATE_DATA_PERMISSIONS,
    STATE_DEPARTMENT,
    STATE_USER_ID,
    STATE_USER_ROLE,
    STATE_ORIGINAL_QUERY,
)
from src.schemas.request_response import AssistantQARequest

TOKEN_BINDINGS = {
    "demo-advisor": AuthenticatedUser("user_advisor", ROLE_ADVISOR, "wealth"),
    "demo-sales": AuthenticatedUser("user_sales", ROLE_INSTITUTIONAL_SALES, "sales"),
    "demo-compliance": AuthenticatedUser("user_compliance", ROLE_COMPLIANCE, "control"),
    "demo-ops": AuthenticatedUser("user_ops", ROLE_OPERATIONS, "ops"),
    "demo-tech": AuthenticatedUser("user_tech", ROLE_TECHNICAL, "tech"),
}


# ══════════════════════════════════════════════════════════════════════
# TC-001 认证与角色映射
# ══════════════════════════════════════════════════════════════════════


@pytest.fixture()
def auth_client():
    return TestClient(app)


@pytest.mark.parametrize("headers,expected_detail", [
    pytest.param({}, "missing bearer token", id="no-token"),
    pytest.param({"Authorization": "Basic abc"}, "invalid authorization header", id="bad-scheme"),
    pytest.param({"Authorization": "Bearer demo-hacker"}, "unknown demo token", id="unknown-token"),
])
def test_tc001_qa_rejects_invalid_credentials(auth_client, headers, expected_detail):
    """TC-001：无 token / 错误 scheme / 未知 token 一律 401，不进入业务处理。"""
    res = auth_client.post(
        API_ROUTE_ASSISTANT_QA, json={"query": "货币基金风险"}, headers=headers
    )
    assert res.status_code == 401
    assert res.json()["detail"] == expected_detail


@pytest.mark.parametrize("token,binding", sorted(TOKEN_BINDINGS.items()))
def test_tc001_demo_tokens_map_to_expected_identity(token, binding):
    """TC-001：五个 demo token 映射到正确的 user_id/role/department。"""
    from src.api.auth import authenticate_user

    user = authenticate_user(authorization=f"Bearer {token}")
    assert user == binding


def test_tc001_initial_state_data_permissions_by_role():
    """TC-001：初始 state 按角色注入数据权限——compliance/tech 含 confidential，
    advisor/sales/ops 不含；user_id/department/user_role 正确。"""
    request = AssistantQARequest(query="货币基金风险等级")
    for role, binding in TOKEN_BINDINGS.items():
        state = build_assistant_initial_state(request, binding)
        assert state[STATE_USER_ID] == binding.user_id
        assert state[STATE_DEPARTMENT] == binding.department
        assert state[STATE_USER_ROLE] == binding.role
        assert state[STATE_ORIGINAL_QUERY] == request.query
        assert state[STATE_DATA_PERMISSIONS] == ROLE_DATA_PERMISSIONS[binding.role]

    confidential_roles = {
        ROLE_COMPLIANCE,
        ROLE_TECHNICAL,
    }
    for role, permissions in ROLE_DATA_PERMISSIONS.items():
        if role in confidential_roles:
            assert PERMISSION_CONFIDENTIAL in permissions
        else:
            assert PERMISSION_CONFIDENTIAL not in permissions
        assert {PERMISSION_PUBLIC, PERMISSION_INTERNAL} <= set(permissions)
