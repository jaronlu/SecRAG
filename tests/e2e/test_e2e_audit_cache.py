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
