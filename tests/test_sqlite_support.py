"""SQLite 存储统一连接约定测试（ISSUE-18）。

各库此前不开 WAL、每操作重放 DDL；50 QPS 目标下是尾延迟放大器。
"""

from __future__ import annotations

import sqlite3

from src.utils import conversation as conversation_module
from src.utils.audit import SQLiteAuditStore
from src.utils.conversation import SQLiteConversationStore
from src.utils.sqlite_support import connect_sqlite


def test_connect_sqlite_enables_wal_and_busy_timeout(tmp_path):
    db = tmp_path / "support.db"

    with connect_sqlite(db) as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000


def test_connect_sqlite_wal_persists_for_fresh_connections(tmp_path):
    db = tmp_path / "persist.db"
    with connect_sqlite(db) as conn:
        conn.execute("CREATE TABLE t (x INTEGER)")

    fresh = sqlite3.connect(str(db))
    assert fresh.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    fresh.close()


def test_conversation_store_enables_wal(tmp_path):
    db = tmp_path / "conversations.db"
    store = SQLiteConversationStore(db)
    store.create_thread(user_id="user-1", user_role="operations", client_id=None)

    fresh = sqlite3.connect(str(db))
    assert fresh.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    fresh.close()


def test_conversation_schema_ddl_runs_once_per_process(tmp_path, monkeypatch):
    """同一库路径不再每操作重放 8 条 DDL（每请求约 36 条 DDL 的来源）。"""
    real_apply = conversation_module.apply_conversation_schema
    calls: list[int] = []
    monkeypatch.setattr(
        conversation_module,
        "apply_conversation_schema",
        lambda conn: (calls.append(1), real_apply(conn)),
    )

    store = SQLiteConversationStore(tmp_path / "conversations.db")
    store.create_thread(user_id="user-1", user_role="operations", client_id=None)
    store.create_thread(user_id="user-2", user_role="operations", client_id=None)
    store.get_outbox_status("nonexistent")

    assert len(calls) == 1


def test_audit_store_enables_wal(tmp_path):
    db = tmp_path / "audit.db"
    store = SQLiteAuditStore(db)
    # 空查询即触发建库建表
    assert store.get_by_request_id("missing") is None

    fresh = sqlite3.connect(str(db))
    assert fresh.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    fresh.close()


def test_audit_schema_guard_engages_once(tmp_path):
    """首个操作标记库路径为已应用 DDL；后续操作不再重放（ISSUE-18）。"""
    from src.schemas.models import AuditEntry

    db = tmp_path / "audit.db"
    store = SQLiteAuditStore(db)
    assert str(db) not in SQLiteAuditStore._schema_applied_paths

    entry = AuditEntry(
        request_id="req-1",
        timestamp="2026-09-29T00:00:00+00:00",
        user_id="user-1",
        user_role="operations",
        department="",
        query={"original": "q"},
        retrieval={"total_chunks": 0, "filtered_chunks": 0},
        reasoning={"iterations": 0, "duration_ms": 0.0, "execution_path": []},
        verification={"passed": True},
        compliance={"passed": True},
        response={"citations": [], "confidence": "high"},
        total_duration_ms=1.0,
    )
    store.insert(entry)
    store.insert(entry)

    assert str(db) in SQLiteAuditStore._schema_applied_paths


def test_portfolio_store_enables_wal(tmp_path):
    from src.portfolio.store import SQLitePortfolioStore

    db = tmp_path / "portfolio.db"
    store = SQLitePortfolioStore(str(db))
    store.list_positions(user_id="user-1")

    fresh = sqlite3.connect(str(db))
    assert fresh.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    fresh.close()
