"""Financial ratios tool backed by the local financial SQLite database."""

from __future__ import annotations

import json
import re
import sqlite3

from langchain_core.tools import tool

from src.tools.sql_query import DEFAULT_DB_PATH, run_select_query

_STOCK_CODE_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
_REPORT_TYPE_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


def _missing_payload(stock_code: str, year: int | None, report_type: str | None) -> str:
    return json.dumps(
        {
            "stock_code": stock_code,
            "year": year,
            "report_type": report_type,
            "missing": True,
            "required_subjects": ["income_statement", "balance_sheet", "market_snapshot"],
        },
        ensure_ascii=False,
    )


@tool
def financial_ratios_tool(
    stock_code: str,
    year: int | None = None,
    report_type: str | None = None,
) -> str:
    """Query PE/PB/ROE and other financial ratios from the financial_ratios table."""
    return query_financial_ratios(stock_code, year=year, report_type=report_type)


def query_financial_ratios(
    stock_code: str,
    year: int | None = None,
    report_type: str | None = None,
    db_path: str = str(DEFAULT_DB_PATH),
) -> str:
    """Query PE/PB/ROE and other financial ratios from an allowlisted local table.

    输入校验与数据库故障抛异常（执行链路转为 status="error"）；
    仅"库/表不存在"视为业务上的无数据，返回 missing 载荷。
    """
    if not _STOCK_CODE_PATTERN.fullmatch(stock_code):
        raise ValueError("标的代码格式不合法")
    if report_type and not _REPORT_TYPE_PATTERN.fullmatch(report_type):
        raise ValueError("报告类型格式不合法")

    filters = ["stock_code = ?"]
    params: list[str | int] = [stock_code]
    if year is not None:
        filters.append("year = ?")
        params.append(year)
    if report_type:
        filters.append("report_type = ?")
        params.append(report_type)

    query = f"SELECT * FROM financial_ratios WHERE {' AND '.join(filters)} ORDER BY year DESC"

    try:
        rows = run_select_query(query=query, db_path=db_path, params=tuple(params))
    except sqlite3.OperationalError as exc:
        message = str(exc)
        if "no such table" in message or "unable to open database file" in message:
            return _missing_payload(stock_code, year, report_type)
        raise
    if rows:
        return json.dumps(rows, ensure_ascii=False)
    return _missing_payload(stock_code, year, report_type)
