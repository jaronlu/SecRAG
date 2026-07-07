"""Read-only SQL tool for local financial datasets."""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

from src.schemas.constants import FINANCIAL_DB_PATH

DEFAULT_DB_PATH = Path(FINANCIAL_DB_PATH)
MAX_SQL_ROWS = 100
ALLOWED_SQL_TABLES: dict[str, set[str]] = {
    "financial_ratios": {
        "stock_code",
        "year",
        "report_type",
        "pe",
        "pb",
        "roe",
        "gross_margin",
        "net_margin",
    },
    "income_statement": {"stock_code", "year", "report_type", "revenue", "net_profit"},
    "balance_sheet": {"stock_code", "year", "report_type", "total_assets", "total_equity"},
    "market_history": {
        "date",
        "code",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "pctChg",
    },
    "market_snapshot": {
        "date",
        "code",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "pctChg",
        "pe",
        "pb",
    },
}

_SQL_PATTERN = re.compile(
    r"^SELECT\s+(?P<columns>\*|[A-Za-z_][A-Za-z0-9_]*(?:\s*,\s*[A-Za-z_][A-Za-z0-9_]*)*)"
    r"\s+FROM\s+(?P<table>[A-Za-z_][A-Za-z0-9_]*)"
    r"(?:\s+WHERE\s+(?P<where>.*?))?"
    r"(?:\s+ORDER\s+BY\s+(?P<order>[A-Za-z_][A-Za-z0-9_]*)(?:\s+(?P<direction>ASC|DESC))?)?"
    r"(?:\s+LIMIT\s+(?P<limit>\d+))?$",
    re.IGNORECASE,
)
_WHERE_TERM_PATTERN = re.compile(
    r"^(?P<column>[A-Za-z_][A-Za-z0-9_]*)\s*"
    r"(?P<op>=|>=|<=|>|<|LIKE)\s*"
    r"(?P<value>\?|[-+]?\d+(?:\.\d+)?|'[^']*')$",
    re.IGNORECASE,
)


def normalize_select_sql(query: str, limit: int = MAX_SQL_ROWS) -> str | None:
    """Validate a single-table allowlisted SELECT SQL and enforce LIMIT."""
    normalized = query.strip()
    if not normalized:
        return None

    if normalized.endswith(";"):
        return None

    if any(token in normalized for token in ("--", "/*", "*/")):
        return None

    match = _SQL_PATTERN.fullmatch(normalized)
    if match is None:
        return None

    table = match.group("table")
    allowed_columns = ALLOWED_SQL_TABLES.get(table)
    if allowed_columns is None:
        return None

    columns = match.group("columns")
    if columns != "*":
        selected = {column.strip() for column in columns.split(",")}
        if not selected or not selected <= allowed_columns:
            return None

    where_clause = match.group("where")
    if where_clause:
        terms = re.split(r"\s+AND\s+", where_clause, flags=re.IGNORECASE)
        for term in terms:
            term_match = _WHERE_TERM_PATTERN.fullmatch(term.strip())
            if term_match is None:
                return None
            if term_match.group("column") not in allowed_columns:
                return None

    order_column = match.group("order")
    if order_column and order_column not in allowed_columns:
        return None

    limit_value = match.group("limit")
    if limit_value is None:
        normalized = f"{normalized} LIMIT {limit}"
    elif int(limit_value) > limit:
        return None
    return normalized


def run_select_query(
    query: str,
    db_path: str | Path = DEFAULT_DB_PATH,
    params: tuple[Any, ...] = (),
) -> list[dict]:
    """Execute a validated read-only query and return rows as dicts."""
    safe_query = normalize_select_sql(query)
    if safe_query is None:
        raise ValueError("仅允许白名单表/字段上的单表 SELECT 查询")

    conn = sqlite3.connect(f"file:{Path(db_path)}?mode=ro", timeout=5, uri=True)
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.cursor()
        cursor.execute(safe_query, params)
        return [dict(row) for row in cursor.fetchall()[:MAX_SQL_ROWS]]
    finally:
        conn.close()


@tool
def sql_query_tool(query: str) -> str:
    """Query a local financial SQLite database with read-only safety checks."""
    try:
        rows = run_select_query(query=query)
        return json.dumps(rows, ensure_ascii=False)
    except (ValueError, sqlite3.Error) as exc:
        return f"查询错误: {exc}"
