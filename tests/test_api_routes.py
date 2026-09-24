"""HTTP 契约测试：通过 TestClient 发真实 ASGI 请求验证路由优先级。

issues.md 一.2：SPA 通配路由必须注册在业务路由之后，否则会截走
GET /health、GET /metrics 等接口。直接调用 health_check() 无法发现该类问题，
必须走完整 HTTP 栈。
"""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture()
def client():
    return TestClient(app)


def test_health_endpoint_reachable(client):
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"


def test_metrics_endpoint_reachable(client):
    res = client.get("/metrics")
    assert res.status_code == 200
    assert "text/plain" in res.headers["content-type"]


def test_unknown_api_path_returns_404_not_index_html(client):
    res = client.get("/v1/nonexistent")
    assert res.status_code == 404


def test_favicon_returns_204(client):
    res = client.get("/favicon.ico")
    assert res.status_code == 204


def test_root_serves_html(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]
