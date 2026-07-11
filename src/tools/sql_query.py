"""Read-only SQL tool for local financial datasets."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from langchain_core.tools import tool
from sqlglot import exp, parse
from sqlglot.errors import ParseError

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
    "research_reports_index": {
        "sequence",
        "stock_code",
        "stock_name",
        "report_name",
        "rating",
        "institution",
        "report_count_1m",
        "eps_2026",
        "pe_2026",
        "eps_2027",
        "pe_2027",
        "eps_2028",
        "pe_2028",
        "industry",
        "report_date",
        "pdf_url",
        "sample_stock_code",
    },
}

def normalize_select_sql(query: str, limit: int = MAX_SQL_ROWS) -> str | None:
    """Validate a single-table allowlisted SELECT AST and enforce LIMIT."""
    if not query.strip() or any(token in query for token in ("--", "/*", "*/")):
        return None
    try:
        statements = [statement for statement in parse(query, read="sqlite") if statement]
    except ParseError:
        return None
    if len(statements) != 1:
        return None
    statement = statements[0]
    if not isinstance(statement, exp.Select):
        return None
    if any(statement.find(node_type) is not None for node_type in (
        exp.Join,
        exp.Subquery,
        exp.Union,
        exp.Intersect,
        exp.Except,
        exp.With,
    )):
        return None
    tables = list(statement.find_all(exp.Table))
    if len(tables) != 1:
        return None
    table = tables[0].name
    allowed_columns = ALLOWED_SQL_TABLES.get(table)
    if allowed_columns is None:
        return None
    for column in statement.find_all(exp.Column):
        if column.name not in allowed_columns:
            return None
    limit_node = statement.args.get("limit")
    if limit_node is None:
        statement = statement.limit(limit)
    else:
        limit_expression = limit_node.expression
        if not isinstance(limit_expression, exp.Literal) or not limit_expression.is_int:
            return None
        if int(limit_expression.this) > limit:
            return None
    return statement.sql(dialect="sqlite")


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
    """Query the local financial SQLite database with read-only safety checks.

    Available datasets include financial statements, market data, and
    `research_reports_index` for AKShare/Eastmoney research report samples.
    """
    try:
        rows = run_select_query(query=query)
        return json.dumps(rows, ensure_ascii=False)
    except (ValueError, sqlite3.Error) as exc:
        return f"查询错误: {exc}"
