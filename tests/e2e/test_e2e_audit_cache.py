"""全链路测试案例：环节 F 审计日志与语义缓存（TC-030~TC-035）。"""

from __future__ import annotations

import json
import sqlite3
import time
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.auth import AuthenticatedUser, authenticate_user
from src.api.main import app
from src.schemas.constants import (
    API_ROUTE_ASSISTANT_QA,
    ROLE_ADVISOR,
    ROLE_INSTITUTIONAL_SALES,
    STATE_CITATIONS,
    STATE_CONFIDENCE,
    STATE_FINAL_ANSWER,
    STATE_THREAD_ID,
    STATE_USER_ID,
)
from src.utils.audit import SQLiteAuditStore
from src.utils.semantic_cache import SemanticCache
from src.utils.conversation import SQLiteConversationStore


# ══════════════════════════════════════════════════════════════════════
# TC-030 审计留痕完整性
# ══════════════════════════════════════════════════════════════════════


def test_tc030_audit_trail_complete_after_qa(run_agent_graph, isolated_stores, fake_llm):
    """TC-030：正常 QA 后审计库按 request_id 记录全链路字段。"""
    from tests.e2e.conftest import build_state

    thread = isolated_stores.conversation.create_thread(
        user_id="user_advisor", user_role=ROLE_ADVISOR, client_id=None, title="TC-030"
    )
    state = run_agent_graph(build_state(thread_id=thread["thread_id"]))

    trail = isolated_stores.audit.get_by_request_id(state["audit_trail"]["request_id"])
    assert trail is not None, "审计条目必须落库"

    assert trail["user_id"] == "user_advisor"
    assert trail["user_role"] == ROLE_ADVISOR
    assert trail["query"]["original"] == "XX货币市场基金的风险等级是什么？"
    assert trail["query"]["intent"] == "产品咨询"

    retrieval = trail["retrieval"]
    assert retrieval["total_chunks"] >= 1
    assert retrieval["filtered_chunks"] >= 1
    assert retrieval["sources"], "审计必须记录检索来源"
    assert any("xx_money_fund_2024.html" in s for s in retrieval["sources"])

    path = trail["reasoning"]["execution_path"]
    for node in (
        "load_conversation_context",
        "query_understand",
        "planner",
        "retrieve",
        "grade_and_filter",
        "reason",
        "verify",
        "compliance_check",
        "compose",
        "audit_log",
    ):
        assert node in path, f"execution_path 缺少 {node}"

    assert trail["verification"]["passed"] is True
    assert trail["compliance"]["passed"] is True
    assert trail["response"]["citations"]
    assert trail["response"]["confidence"] in ("medium", "high")
    assert trail["total_duration_ms"] >= 0


# ══════════════════════════════════════════════════════════════════════
# TC-031 审计写入失败不阻断
# ══════════════════════════════════════════════════════════════════════


class _FailingAuditStore:
    """insert 必失败的审计存储替身（模拟审计库不可写）。"""

    def __init__(self) -> None:
        self.insert_calls = 0

    def insert(self, entry: Any) -> None:
        self.insert_calls += 1
        raise RuntimeError("audit db disk full")


def test_tc031_audit_write_failure_does_not_break_qa(
    run_agent_graph, isolated_stores, fake_llm, monkeypatch
):
    """TC-031：审计库写入失败 → 回答链路继续，outbox 落盘并标记 audit_write_failed。"""
    from src.agents import nodes as agent_nodes

    from tests.e2e.conftest import build_state

    failing = _FailingAuditStore()
    monkeypatch.setattr(agent_nodes, "_get_audit_store", lambda: failing)

    thread = isolated_stores.conversation.create_thread(
        user_id="user_advisor", user_role=ROLE_ADVISOR, client_id=None, title="TC-031"
    )
    state = run_agent_graph(build_state(thread_id=thread["thread_id"]))

    assert failing.insert_calls == 1
    assert state[STATE_FINAL_ANSWER], "审计失败不得影响回答"
    assert state["audit_trail"]["audit_write_failed"] is True
    assert "disk full" in state["audit_trail"]["audit_write_error"]

    outbox_path = isolated_stores.tmp_path / "audit_outbox.jsonl"
    assert outbox_path.exists(), "失败审计必须进入本地 outbox 待重试"
    record = json.loads(outbox_path.read_text(encoding="utf-8").splitlines()[0])
    assert record["error"] == "audit db disk full"
    assert record["entry"]["request_id"] == state["audit_trail"]["request_id"]


# ══════════════════════════════════════════════════════════════════════
# TC-032/TC-034 共享：QA API + 可控缓存替身
# ══════════════════════════════════════════════════════════════════════


class _HitCache:
    """命中替身：返回 store 时保存的终态合规快照。"""

    STORED_COMPLIANCE = {
        "passed": True,
        "flags": [],
        "risk_disclosure": "",
        "suitability_warning": "",
    }
    STORED_VERIFICATION = {"passed": True, "issues": [], "confidence": "high"}

    def lookup(self, query, role=""):
        return {
            "query": query,
            "answer": "货币基金风险等级为低。",
            "citations": [{"source": "cached.pdf"}],
            "confidence": "high",
            "similarity": 0.95,
            "hit_count": 1,
            "compliance": dict(self.STORED_COMPLIANCE),
            "verification": dict(self.STORED_VERIFICATION),
        }

    def store(self, *args: Any, **kwargs: Any) -> bool:
        raise AssertionError("命中路径不得再写缓存")


class _SpyCache:
    """记录 store 调用的未命中替身。"""

    def __init__(self) -> None:
        self.store_calls: list[dict[str, Any]] = []

    def lookup(self, query, role=""):
        return None

    def store(self, query, answer, citations=None, confidence="", role="",
              compliance=None, verification=None):
        self.store_calls.append(
            {"query": query, "answer": answer, "role": role,
             "compliance": compliance, "verification": verification}
        )
        return True


class _FakeAgentApp:
    """固定终态的 Agent 替身：可注入 verification/compliance 结果。"""

    def __init__(self, answer="货币基金风险等级为低风险等级 R1。", compliance=None,
                 verification=None):
        self.answer = answer
        self.compliance = compliance or {"passed": True, "flags": [], "risk_disclosure": ""}
        self.verification = verification or {"passed": True, "issues": []}
        self.invoked = 0

    def invoke(self, state, config=None):
        self.invoked += 1
        return {
            **state,
            STATE_FINAL_ANSWER: self.answer,
            STATE_CITATIONS: [{"source": "a.pdf"}],
            STATE_CONFIDENCE: "high",
            "compliance": self.compliance,
            "verification": self.verification,
        }


@pytest.fixture()
def cache_api(monkeypatch, tmp_path):
    """QA API 上下文：cache holder 可替换缓存替身，agent holder 注入 Agent。"""
    audit_store = SQLiteAuditStore(tmp_path / "audit.db")
    conversation = SQLiteConversationStore(tmp_path / "conversations.db")
    monkeypatch.setattr("src.api.main._get_conversation_store", lambda: conversation)
    monkeypatch.setattr("src.api.main._get_cache_hit_audit_store", lambda: audit_store)

    holder: dict[str, Any] = {"cache": _SpyCache(), "agent": None}
    monkeypatch.setattr("src.api.main.get_semantic_cache", lambda: holder["cache"])

    def _agent():
        assert holder["agent"] is not None, "测试未注入 agent 替身"
        return holder["agent"]

    monkeypatch.setattr("src.api.main._get_agent_app", _agent)
    app.dependency_overrides[authenticate_user] = lambda: AuthenticatedUser(
        "user_advisor", ROLE_ADVISOR, "wealth"
    )
    holder["client"] = TestClient(app)
    holder["audit_db"] = tmp_path / "audit.db"
    yield holder
    app.dependency_overrides.clear()


# ══════════════════════════════════════════════════════════════════════
# TC-032 缓存命中返回存储合规快照并补审计
# ══════════════════════════════════════════════════════════════════════


def test_tc032_cache_hit_returns_stored_snapshot_and_persists_audit_event(cache_api):
    """TC-032：命中路径返回存储快照、跳过 Agent、泄露内部字段、补 semantic_cache_hit 审计。"""
    cache_api["cache"] = _HitCache()
    # Agent 替身带哨兵：命中路径不得执行
    class _SentinelAgent:
        def invoke(self, state, config=None):
            raise AssertionError("缓存命中不应执行 Agent")

    cache_api["agent"] = _SentinelAgent()

    res = cache_api["client"].post(API_ROUTE_ASSISTANT_QA, json={"query": "货币基金风险等级"})

    assert res.status_code == 200
    body = res.json()
    assert body["compliance"] == _HitCache.STORED_COMPLIANCE, "必须返回存储的终态快照"
    assert body["answer"] == "货币基金风险等级为低。"
    assert "cached" not in body and "cache_similarity" not in body, "内部字段不得泄露"

    # 命中路径补持久化审计事件
    with sqlite3.connect(str(cache_api["audit_db"])) as conn:
        rows = conn.execute("SELECT payload_json FROM audit_entries").fetchall()
    assert len(rows) == 1
    trail = json.loads(rows[0][0])
    assert trail["reasoning"]["execution_path"] == ["semantic_cache_hit"]
    assert trail["compliance"] == _HitCache.STORED_COMPLIANCE


# ══════════════════════════════════════════════════════════════════════
# TC-034 失败终态不入缓存
# ══════════════════════════════════════════════════════════════════════


def test_tc034_failed_terminal_state_is_not_cached(cache_api):
    """TC-034：合规未通过/答案过短不入缓存；成功终态才写入（对照）。"""
    cache_api["agent"] = _FakeAgentApp(
        answer="当前请求未通过合规检查。",
        compliance={"passed": False, "flags": ["advice:建议买入"], "risk_disclosure": ""},
        verification={"passed": True, "issues": []},
    )
    res = cache_api["client"].post(API_ROUTE_ASSISTANT_QA, json={"query": "推荐个能买的基金"})
    assert res.status_code == 200
    assert cache_api["cache"].store_calls == [], "合规失败终态不得入缓存"

    cache_api["cache"].store_calls.clear()
    cache_api["agent"] = _FakeAgentApp()  # 默认成功终态
    res = cache_api["client"].post(API_ROUTE_ASSISTANT_QA, json={"query": "货币基金风险等级"})
    assert res.status_code == 200
    assert len(cache_api["cache"].store_calls) == 1, "成功终态应写入缓存"
    stored = cache_api["cache"].store_calls[0]
    assert stored["role"] == ROLE_ADVISOR
    assert stored["compliance"]["passed"] is True
    assert stored["verification"]["passed"] is True
