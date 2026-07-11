"""Fetch a tiny real securities sample set for ingestion/retrieval tests.

The script intentionally keeps data-provider libraries optional so SecRAG's
core runtime dependencies stay small. Run with:

    uv run --with akshare --with efinance --with baostock python scripts/fetch_real_securities_data.py
"""

from __future__ import annotations

import hashlib
import importlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import httpx
import pandas as pd
from typing_extensions import NotRequired, TypedDict

from src.schemas.constants import (
    DOC_TYPE_ANNOUNCEMENT,
    DOC_TYPE_FINANCIAL_DATA,
    DOC_TYPE_RESEARCH_REPORT,
    META_ALLOWED_ROLES,
    META_DATE,
    META_DOC_TYPE,
    META_PERMISSION_LEVEL,
    META_RETRIEVAL_SOURCE,
    META_SOURCE,
    META_STOCK_CODE,
    META_TITLE,
    PERMISSION_INTERNAL,
    PERMISSION_PUBLIC,
    ROLE_ADVISOR,
    ROLE_COMPLIANCE,
    ROLE_INSTITUTIONAL_SALES,
    ROLE_OPERATIONS,
    ROLE_TECHNICAL,
    SOURCE_REPORT,
)

CNINFO_QUERY_URL = "http://www.cninfo.com.cn/new/hisAnnouncement/query"
CNINFO_STATIC_URL = "http://static.cninfo.com.cn"
DEFAULT_OUTPUT_DIR = Path("data/raw/real_securities_data")
USER_AGENT = "Mozilla/5.0 SecRAG real securities data fetcher"


@dataclass(frozen=True)
class AnnualReportTarget:
    stock_code: str
    stock_name: str
    query_keyword: str
    cninfo_stock: str


@dataclass(frozen=True)
class ResearchReportTarget:
    stock_code: str
    stock_name: str


class CninfoAnnouncement(TypedDict):
    secCode: str
    announcementTitle: str
    announcementTime: int
    adjunctUrl: str


class MetadataRecord(TypedDict):
    relative_path: str
    doc_type: str
    retrieval_source: str
    permission_level: str
    allowed_roles: list[str]
    title: str
    date: str
    stock_code: str
    source: str
    provider: str
    sha256: str
    institution: NotRequired[str]
    rating: NotRequired[str]


class ManifestMetadata(TypedDict, total=False):
    doc_type: str
    retrieval_source: str
    permission_level: str
    allowed_roles: list[str]
    title: str
    date: str
    stock_code: str
    source: str
    provider: str
    sha256: str
    institution: str
    rating: str


ANNUAL_REPORT_TARGETS = [
    AnnualReportTarget("000001", "平安银行", "年度报告", "000001,gssz0000001"),
    AnnualReportTarget("300750", "宁德时代", "年度报告", "300750,GD165627"),
]

RESEARCH_REPORT_TARGETS = [
    ResearchReportTarget("000001", "平安银行"),
    ResearchReportTarget("600519", "贵州茅台"),
]


def import_optional_module(module_name: str) -> ModuleType:
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise RuntimeError(
            f"{module_name} is required for real securities data fetching. "
            "Run: uv run --with akshare --with efinance --with baostock "
            "python scripts/fetch_real_securities_data.py"
        ) from exc


def clean_text(value: Any) -> str:
    """Remove HTML highlighting tags from provider titles."""
    return re.sub(r"<[^>]+>", "", str(value or "")).strip()


def safe_filename(value: str) -> str:
    """Return a stable ASCII-ish filename stem while preserving useful numbers."""
    return re.sub(r"[^0-9A-Za-z._-]+", "_", value).strip("_")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_pdf(path: Path) -> None:
    with path.open("rb") as f:
        header = f.read(5)
    if header != b"%PDF-":
        raise RuntimeError(f"Downloaded file is not a PDF: {path}")


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    referer = "https://data.eastmoney.com/" if "dfcfw.com" in url else "http://www.cninfo.com.cn/"
    with httpx.stream(
        "GET",
        url,
        headers={"User-Agent": USER_AGENT, "Referer": referer},
        follow_redirects=True,
        timeout=60,
    ) as response:
        response.raise_for_status()
        with destination.open("wb") as f:
            for chunk in response.iter_bytes():
                f.write(chunk)


def query_cninfo_annual_report(
    client: httpx.Client, target: AnnualReportTarget
) -> CninfoAnnouncement:
    payload = {
        "pageNum": "1",
        "pageSize": "10",
        "column": "szse",
        "tabName": "fulltext",
        "plate": "",
        "stock": target.cninfo_stock,
        "searchkey": target.query_keyword,
        "seDate": "2024-01-01~2026-07-07",
        "isHLtitle": "true",
    }
    response = client.post(
        CNINFO_QUERY_URL,
        data=payload,
        headers={
            "User-Agent": USER_AGENT,
            "Referer": "http://www.cninfo.com.cn/new/commonUrl?url=disclosure/list/notice",
        },
        timeout=30,
    )
    response.raise_for_status()
    announcements = cast(list[dict[str, Any]], response.json().get("announcements") or [])
    for announcement in announcements:
        title = clean_text(announcement.get("announcementTitle"))
        if (
            announcement.get("secCode") == target.stock_code
            and "年度报告" in title
            and "摘要" not in title
            and str(announcement.get("adjunctUrl", "")).lower().endswith(".pdf")
        ):
            return {
                "secCode": str(announcement["secCode"]),
                "announcementTitle": str(announcement["announcementTitle"]),
                "announcementTime": int(announcement["announcementTime"]),
                "adjunctUrl": str(announcement["adjunctUrl"]),
            }
    raise RuntimeError(f"No annual report found for {target.stock_code} {target.stock_name}")


def fetch_annual_reports(output_dir: Path) -> list[MetadataRecord]:
    records: list[MetadataRecord] = []
    report_dir = output_dir / "announcements"

    with httpx.Client(follow_redirects=True) as client:
        for target in ANNUAL_REPORT_TARGETS:
            announcement = query_cninfo_annual_report(client, target)
            title = clean_text(announcement["announcementTitle"])
            date = (
                pd
                .to_datetime(announcement["announcementTime"], unit="ms", utc=True)
                .tz_convert("Asia/Shanghai")
                .strftime("%Y-%m-%d")
            )
            url = f"{CNINFO_STATIC_URL}/{announcement['adjunctUrl']}"
            filename = f"{target.stock_code}_{safe_filename(title)}.pdf"
            path = report_dir / filename
            download_file(url, path)
            ensure_pdf(path)
            records.append({
                "relative_path": str(path.relative_to(output_dir)),
                META_DOC_TYPE: DOC_TYPE_ANNOUNCEMENT,
                META_RETRIEVAL_SOURCE: SOURCE_REPORT,
                META_PERMISSION_LEVEL: PERMISSION_INTERNAL,
                META_ALLOWED_ROLES: [ROLE_ADVISOR, ROLE_INSTITUTIONAL_SALES, ROLE_COMPLIANCE],
                META_TITLE: f"{target.stock_name}{title}",
                META_DATE: date,
                META_STOCK_CODE: target.stock_code,
                META_SOURCE: url,
                "provider": "cninfo",
                "sha256": sha256_file(path),
            })
            print(f"annual_report: {path}")
    return records


def fetch_research_reports(output_dir: Path) -> list[MetadataRecord]:
    ak = import_optional_module("akshare")

    records: list[MetadataRecord] = []
    report_dir = output_dir / "reports"
    indexes: list[pd.DataFrame] = []

    for target in RESEARCH_REPORT_TARGETS:
        df = ak.stock_research_report_em(symbol=target.stock_code)
        if df.empty:
            raise RuntimeError(
                f"No research report found for {target.stock_code} {target.stock_name}"
            )
        row = df.iloc[0]
        pdf_url = str(row["报告PDF链接"])
        title = clean_text(row["报告名称"])
        filename = f"{target.stock_code}_{safe_filename(title)}.pdf"
        path = report_dir / filename
        download_file(pdf_url, path)
        ensure_pdf(path)

        records.append({
            "relative_path": str(path.relative_to(output_dir)),
            META_DOC_TYPE: DOC_TYPE_RESEARCH_REPORT,
            META_RETRIEVAL_SOURCE: SOURCE_REPORT,
            META_PERMISSION_LEVEL: PERMISSION_PUBLIC,
            META_ALLOWED_ROLES: [
                ROLE_ADVISOR,
                ROLE_INSTITUTIONAL_SALES,
                ROLE_COMPLIANCE,
                ROLE_OPERATIONS,
                ROLE_TECHNICAL,
            ],
            META_TITLE: f"{target.stock_name}{title}",
            META_DATE: str(row["日期"]),
            META_STOCK_CODE: target.stock_code,
            META_SOURCE: pdf_url,
            "provider": "akshare/eastmoney",
            "institution": str(row.get("机构", "")),
            "rating": str(row.get("东财评级", "")),
            "sha256": sha256_file(path),
        })
        indexes.append(df.head(5).assign(sample_stock_code=target.stock_code))
        print(f"research_report: {path}")

    index_path = output_dir / "financials" / "research_reports_index.csv"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(indexes, ignore_index=True).to_csv(index_path, index=False)
    records.append({
        "relative_path": str(index_path.relative_to(output_dir)),
        META_DOC_TYPE: DOC_TYPE_FINANCIAL_DATA,
        META_RETRIEVAL_SOURCE: "sql_query",
        META_PERMISSION_LEVEL: PERMISSION_INTERNAL,
        META_ALLOWED_ROLES: [ROLE_ADVISOR, ROLE_INSTITUTIONAL_SALES, ROLE_COMPLIANCE],
        META_TITLE: "AKShare 东方财富研报索引样本",
        META_DATE: "2026-07-07",
        META_STOCK_CODE: "",
        META_SOURCE: "akshare.stock_research_report_em",
        "provider": "akshare/eastmoney",
        "sha256": sha256_file(index_path),
    })
    print(f"research_index_csv: {index_path}")
    return records


def fetch_efinance_quote_history(output_dir: Path) -> MetadataRecord | None:
    ef = import_optional_module("efinance")

    try:
        df = ef.stock.get_quote_history("600519", beg="20250101", end="20250131")
    except Exception as exc:
        print(f"efinance quote history skipped: {exc}")
        base_info = ef.stock.get_base_info("600519")
        df = pd.DataFrame([base_info.to_dict()])
        df = df.rename(columns={"股票代码": "code", "股票名称": "stock_name"})
        df["year"] = "2026"
        df["provider"] = "efinance/eastmoney"
        path = output_dir / "financials" / "efinance_600519_base_info.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        print(f"efinance_base_csv: {path}")
        return {
            "relative_path": str(path.relative_to(output_dir)),
            META_DOC_TYPE: DOC_TYPE_FINANCIAL_DATA,
            META_RETRIEVAL_SOURCE: "sql_query",
            META_PERMISSION_LEVEL: PERMISSION_INTERNAL,
            META_ALLOWED_ROLES: [ROLE_ADVISOR, ROLE_INSTITUTIONAL_SALES, ROLE_COMPLIANCE],
            META_TITLE: "贵州茅台 efinance 基础财务估值样本",
            META_DATE: "2026",
            META_STOCK_CODE: "600519",
            META_SOURCE: "efinance.stock.get_base_info",
            "provider": "efinance/eastmoney",
            "sha256": sha256_file(path),
        }
    df = df.rename(
        columns={
            "股票名称": "stock_name",
            "股票代码": "code",
            "日期": "date",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
            "振幅": "amplitude",
            "涨跌幅": "pct_change",
            "涨跌额": "change",
            "换手率": "turnover_rate",
        }
    )
    df["year"] = df["date"].str[:4]
    df["provider"] = "efinance/eastmoney"
    path = output_dir / "financials" / "efinance_600519_quote_history_202501.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"efinance_csv: {path}")
    return {
        "relative_path": str(path.relative_to(output_dir)),
        META_DOC_TYPE: DOC_TYPE_FINANCIAL_DATA,
        META_RETRIEVAL_SOURCE: "sql_query",
        META_PERMISSION_LEVEL: PERMISSION_INTERNAL,
        META_ALLOWED_ROLES: [ROLE_ADVISOR, ROLE_INSTITUTIONAL_SALES, ROLE_COMPLIANCE],
        META_TITLE: "贵州茅台 2025 年 1 月行情样本",
        META_DATE: "2025",
        META_STOCK_CODE: "600519",
        META_SOURCE: "efinance.stock.get_quote_history",
        "provider": "efinance/eastmoney",
        "sha256": sha256_file(path),
    }


def fetch_baostock_stock_basic(output_dir: Path) -> MetadataRecord:
    bs = import_optional_module("baostock")

    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(f"baostock login failed: {login.error_msg}")
    try:
        rows = []
        fields: list[str] = []
        for code in ("sh.600519", "sz.000001", "sz.300750"):
            result = bs.query_stock_basic(code=code)
            fields = result.fields
            while result.error_code == "0" and result.next():
                rows.append(result.get_row_data())
            if result.error_code != "0":
                raise RuntimeError(f"baostock query_stock_basic failed: {result.error_msg}")
    finally:
        bs.logout()

    df = pd.DataFrame(rows, columns=fields)
    df["code"] = df["code"].str.replace("sh.", "", regex=False).str.replace("sz.", "", regex=False)
    df["year"] = "2026"
    df["provider"] = "baostock"
    path = output_dir / "financials" / "baostock_stock_basic.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"baostock_basic_csv: {path}")
    return {
        "relative_path": str(path.relative_to(output_dir)),
        META_DOC_TYPE: DOC_TYPE_FINANCIAL_DATA,
        META_RETRIEVAL_SOURCE: "sql_query",
        META_PERMISSION_LEVEL: PERMISSION_INTERNAL,
        META_ALLOWED_ROLES: [ROLE_ADVISOR, ROLE_INSTITUTIONAL_SALES, ROLE_COMPLIANCE],
        META_TITLE: "A 股股票基础信息样本",
        META_DATE: "2026",
        META_STOCK_CODE: "",
        META_SOURCE: "baostock.query_stock_basic",
        "provider": "baostock",
        "sha256": sha256_file(path),
    }


def fetch_baostock_valuation_history(output_dir: Path) -> MetadataRecord:
    bs = import_optional_module("baostock")

    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(f"baostock login failed: {login.error_msg}")
    try:
        fields = (
            "date,code,open,high,low,close,volume,amount,turn,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM"
        )
        result = bs.query_history_k_data_plus(
            "sh.600519",
            fields,
            start_date="2025-01-02",
            end_date="2025-01-31",
            frequency="d",
            adjustflag="3",
        )
        rows = []
        while result.error_code == "0" and result.next():
            rows.append(result.get_row_data())
        if result.error_code != "0":
            raise RuntimeError(f"baostock query failed: {result.error_msg}")
    finally:
        bs.logout()

    df = pd.DataFrame(rows, columns=result.fields)
    df["year"] = df["date"].str[:4]
    df["provider"] = "baostock"
    path = output_dir / "financials" / "baostock_600519_valuation_202501.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"baostock_csv: {path}")
    return {
        "relative_path": str(path.relative_to(output_dir)),
        META_DOC_TYPE: DOC_TYPE_FINANCIAL_DATA,
        META_RETRIEVAL_SOURCE: "sql_query",
        META_PERMISSION_LEVEL: PERMISSION_INTERNAL,
        META_ALLOWED_ROLES: [ROLE_ADVISOR, ROLE_INSTITUTIONAL_SALES, ROLE_COMPLIANCE],
        META_TITLE: "贵州茅台 2025 年 1 月估值行情样本",
        META_DATE: "2025",
        META_STOCK_CODE: "600519",
        META_SOURCE: "baostock.query_history_k_data_plus",
        "provider": "baostock",
        "sha256": sha256_file(path),
    }


def write_metadata(output_dir: Path, records: list[MetadataRecord]) -> None:
    for record in records:
        relative_path = record["relative_path"]
        manifest_record: ManifestMetadata = {
            META_DOC_TYPE: record[META_DOC_TYPE],
            META_RETRIEVAL_SOURCE: record[META_RETRIEVAL_SOURCE],
            META_PERMISSION_LEVEL: record[META_PERMISSION_LEVEL],
            META_ALLOWED_ROLES: record[META_ALLOWED_ROLES],
            META_TITLE: record[META_TITLE],
            META_DATE: record[META_DATE],
            META_STOCK_CODE: record[META_STOCK_CODE],
            META_SOURCE: record[META_SOURCE],
            "provider": record["provider"],
            "sha256": record["sha256"],
        }
        if "institution" in record:
            manifest_record["institution"] = record["institution"]
        if "rating" in record:
            manifest_record["rating"] = record["rating"]
        source_path = output_dir / relative_path
        manifest = source_path.with_name(source_path.name + ".meta.json")
        manifest.write_text(
            json.dumps(manifest_record, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"metadata: {manifest}")


def main() -> None:
    output_dir = DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[MetadataRecord] = []
    records.extend(fetch_annual_reports(output_dir))
    records.extend(fetch_research_reports(output_dir))
    efinance_record = fetch_efinance_quote_history(output_dir)
    if efinance_record is not None:
        records.append(efinance_record)
    records.append(fetch_baostock_stock_basic(output_dir))
    records.append(fetch_baostock_valuation_history(output_dir))
    write_metadata(output_dir, records)


if __name__ == "__main__":
    main()
