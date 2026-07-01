"""Read-only SQL tool for local financial datasets."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from langchain_core.tools import tool

DEFAULT_DB_PATH = Path("data/financial.db")
MAX_SQL_ROWS = 100
FORBIDDEN_KEYWORDS = (
    "DROP",
    "DELETE",
    "UPDATE",
    "INSERT",
    "ALTER",
    "TRUNCATE",
    "CREATE",
    "GRANT",
    "REVOKE",
    "EXEC",
    "EXECUTE",
    "ATTACH",
    "DETACH",
    "PRAGMA",
    "UNION",
    "--",
    ";",
    "/*",
    "*/",
)


def normalize_select_sql(query: str, limit: int = MAX_SQL_ROWS) -> str | None:
    """Validate a query as read-only SELECT SQL and enforce LIMIT."""
    normalized = query.strip()
    if not normalized:
        return None

    if normalized.endswith(";"):
        normalized = normalized[:-1].rstrip()

    query_upper = normalized.upper()
    if not query_upper.startswith("SELECT"):
        return None

    for keyword in FORBIDDEN_KEYWORDS:
        if keyword in query_upper:
            return None

    if "LIMIT" not in query_upper:
        normalized = f"{normalized} LIMIT {limit}"
    return normalized


def run_select_query(query: str, db_path: str | Path = DEFAULT_DB_PATH) -> list[dict]:
    """Execute a validated read-only query and return rows as dicts."""
    safe_query = normalize_select_sql(query)
    if safe_query is None:
        raise ValueError("仅允许 SELECT 查询，且不能包含危险关键字")

    conn = sqlite3.connect(str(db_path), timeout=5)
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.cursor()
        cursor.execute(safe_query)
        return [dict(row) for row in cursor.fetchall()[:MAX_SQL_ROWS]]
    finally:
        conn.close()


@tool
def sql_query_tool(query: str, db_path: str = str(DEFAULT_DB_PATH)) -> str:
    """Query a local financial SQLite database with read-only safety checks."""
    try:
        rows = run_select_query(query=query, db_path=db_path)
        return json.dumps(rows, ensure_ascii=False)
    except (ValueError, sqlite3.Error) as exc:
        return f"查询错误: {exc}"
