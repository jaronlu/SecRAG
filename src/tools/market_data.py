"""Market data tool with local SQLite fallback and optional BaoStock support."""

from __future__ import annotations

import json
import re
import sqlite3
from datetime import date, timedelta
from importlib import import_module
from pathlib import Path

from langchain_core.tools import tool

from src.tools.sql_query import DEFAULT_DB_PATH, MAX_SQL_ROWS, run_select_query

DEFAULT_MARKET_FIELDS = "date,code,open,high,low,close,volume,amount,pctChg"
_FIELD_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_STOCK_CODE_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


def _default_dates(start_date: str, end_date: str) -> tuple[str, str]:
    end = end_date or date.today().isoformat()
    start = start_date or (date.fromisoformat(end) - timedelta(days=30)).isoformat()
    return start, end


def _local_market_data(
    stock_code: str,
    start_date: str,
    end_date: str,
    fields: str,
    db_path: str | Path,
) -> list[dict]:
    if not _STOCK_CODE_PATTERN.fullmatch(stock_code):
        raise ValueError("标的代码格式不合法")

    selected_fields = [field.strip() for field in fields.split(",") if field.strip()]
    if not selected_fields:
        selected_fields = DEFAULT_MARKET_FIELDS.split(",")
    if any(not _FIELD_PATTERN.fullmatch(field) for field in selected_fields):
        raise ValueError("行情字段格式不合法")

    columns = ", ".join(selected_fields)
    for table in ("market_history", "market_snapshot"):
        try:
            return run_select_query(
                query=(
                    f"SELECT {columns} FROM {table} "
                    "WHERE code = ? "
                    "AND date >= ? AND date <= ? "
                    "ORDER BY date DESC"
                ),
                db_path=db_path,
                params=(stock_code, start_date, end_date),
            )
        except (ValueError, sqlite3.Error):
            continue
    return []


def _baostock_market_data(
    stock_code: str,
    start_date: str,
    end_date: str,
    fields: str,
) -> list[dict]:
    try:
        bs = import_module("baostock")
    except ImportError as exc:
        raise RuntimeError("未安装 baostock，且本地行情表无数据") from exc

    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(f"BaoStock 连接失败: {login.error_msg}")

    try:
        rs = bs.query_history_k_data_plus(
            stock_code,
            fields,
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag="2",
        )
        if rs.error_code != "0":
            raise RuntimeError(f"BaoStock 查询失败: {rs.error_msg}")

        rows = []
        while rs.next() and len(rows) < MAX_SQL_ROWS:
            rows.append(dict(zip(rs.fields, rs.get_row_data())))
        return rows
    finally:
        bs.logout()


@tool
def market_data_tool(
    stock_code: str,
    start_date: str = "",
    end_date: str = "",
    fields: str = DEFAULT_MARKET_FIELDS,
) -> str:
    """Get A-share market data from local tables, falling back to BaoStock when available."""
    return query_market_data(stock_code, start_date=start_date, end_date=end_date, fields=fields)


def query_market_data(
    stock_code: str,
    start_date: str = "",
    end_date: str = "",
    fields: str = DEFAULT_MARKET_FIELDS,
    db_path: str = str(DEFAULT_DB_PATH),
) -> str:
    """Get A-share market data; db_path is internal for tests and offline fixtures."""
    try:
        start, end = _default_dates(start_date, end_date)
        rows = _local_market_data(stock_code, start, end, fields, db_path)
        if not rows:
            rows = _baostock_market_data(stock_code, start, end, fields)
        return json.dumps(rows[:MAX_SQL_ROWS], ensure_ascii=False)
    except (RuntimeError, ValueError, sqlite3.Error) as exc:
        return f"行情查询错误: {exc}"
