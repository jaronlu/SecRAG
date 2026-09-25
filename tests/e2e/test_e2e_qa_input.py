"""TC-002：QA 请求非法输入边界（空 query / 超长 query / 未知字段）。"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.auth import AuthenticatedUser, authenticate_user
from src.api.main import app
from src.schemas.constants import API_ROUTE_ASSISTANT_QA, ROLE_ADVISOR


@pytest.fixture()
def qa_client():
    app.dependency_overrides[authenticate_user] = lambda: AuthenticatedUser(
        "user_advisor", ROLE_ADVISOR, "wealth"
    )
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.mark.parametrize("payload,desc", [
    pytest.param({"query": ""}, "空 query", id="empty-query"),
    pytest.param({"query": "货" * 501}, "501 字符超长 query", id="too-long-query"),
    pytest.param({"query": "货币基金风险", "foo": "bar"}, "未知字段", id="extra-field"),
])
def test_tc002_qa_rejects_invalid_payload(qa_client, payload, desc):
    """TC-002：三种非法输入均 422，不触发 Agent 执行。"""
    res = qa_client.post(API_ROUTE_ASSISTANT_QA, json=payload)
    assert res.status_code == 422, f"{desc} 应被 Pydantic 拒绝"
