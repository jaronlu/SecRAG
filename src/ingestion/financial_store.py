"""Load structured financial CSV samples into the local SQLite database."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from src.schemas.constants import FINANCIAL_DB_PATH

RESEARCH_REPORT_INDEX_COLUMNS = {
    "序号": "sequence",
    "股票代码": "stock_code",
    "股票简称": "stock_name",
    "报告名称": "report_name",
    "东财评级": "rating",
    "机构": "institution",
    "近一月个股研报数": "report_count_1m",
    "2026-盈利预测-收益": "eps_2026",
    "2026-盈利预测-市盈率": "pe_2026",
    "2027-盈利预测-收益": "eps_2027",
    "2027-盈利预测-市盈率": "pe_2027",
    "2028-盈利预测-收益": "eps_2028",
    "2028-盈利预测-市盈率": "pe_2028",
    "行业": "industry",
    "日期": "report_date",
    "报告PDF链接": "pdf_url",
    "sample_stock_code": "sample_stock_code",
}


def import_research_reports_index(
    csv_path: str | Path,
    db_path: str | Path = FINANCIAL_DB_PATH,
) -> int:
    """Replace the local research report index table from a verified CSV snapshot."""
    csv_path = Path(csv_path)
    frame = pd.read_csv(csv_path, dtype={"股票代码": str, "sample_stock_code": str})
    missing = set(RESEARCH_REPORT_INDEX_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"研报索引 CSV 缺少字段: {sorted(missing)}")
    frame = frame[list(RESEARCH_REPORT_INDEX_COLUMNS)].rename(
        columns=RESEARCH_REPORT_INDEX_COLUMNS
    )
    frame["stock_code"] = frame["stock_code"].str.zfill(6)
    frame["sample_stock_code"] = frame["sample_stock_code"].str.zfill(6)

    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(db_path), timeout=5) as conn:
        frame.to_sql("research_reports_index", conn, if_exists="replace", index=False)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_reports_stock_date "
            "ON research_reports_index(stock_code, report_date DESC)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_reports_institution "
            "ON research_reports_index(institution)"
        )
    return len(frame)
