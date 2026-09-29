"""SQLite 连接统一入口（ISSUE-18）。

conversation/audit/portfolio/registry 此前每操作裸建连接且不开 WAL；
50 QPS 目标下，写锁竞争与 DDL 重放是尾延迟放大器。WAL 是库文件级
持久属性（首次设置后随文件保留），busy_timeout 是连接级属性。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path


def connect_sqlite(
    db_path: str | Path,
    timeout: float = 5.0,
    **kwargs: object,
) -> sqlite3.Connection:
    """创建带 WAL 与 busy_timeout（5s，与 timeout 参数对齐）的 SQLite 连接。

    只读连接（file:...?mode=ro）不得使用本入口：切换 journal_mode 需要
    写权限，调用方自行用 sqlite3.connect。
    """
    conn = sqlite3.connect(str(db_path), timeout=timeout, **kwargs)  # type: ignore[arg-type]
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn
