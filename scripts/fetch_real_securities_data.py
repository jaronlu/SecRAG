"""Fetch batch real securities data for ingestion and retrieval tests.

The script intentionally keeps data-provider libraries optional so SecRAG's
core runtime dependencies stay small. Run with:

    uv run --with akshare --with efinance --with baostock python scripts/fetch_real_securities_data.py

Batch mode resolves the index constituent list at runtime instead of relying on
a hardcoded code table, retries transient download failures, isolates per-symbol
errors into a failure journal so one bad symbol cannot abort the run, and keeps
a watermarked state file so repeated executions stay idempotent.

Provider entry points are injectable. Tests pass fake providers and never touch
external networks, so batch orchestration stays verifiable offline.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Sequence, cast

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
CNINFO_SEARCH_URL = "http://www.cninfo.com.cn/new/information/topSearch/query"
CNINFO_STATIC_URL = "http://static.cninfo.com.cn"
DEFAULT_OUTPUT_DIR = Path("data/raw/real_securities_data")
USER_AGENT = "Mozilla/5.0 SecRAG real securities data fetcher"

DEFAULT_INDEX_SYMBOL = "000300"
DEFAULT_LOOKBACK_DAYS = 365
MAX_DOWNLOAD_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 2.0
STATE_FILENAME = ".fetch_state.json"
FAILURES_FILENAME = ".fetch_failures.json"
RESEARCH_INDEX_RELATIVE_PATH = "financials/research_reports_index.csv"

# Column names follow akshare's index_stock_cons. On provider schema drift these
# candidates must be extended; there is deliberately no silent fallback column.
CONSTITUENT_CODE_FIELDS = ("品种代码", "股票代码", "证券代码", "code")
CONSTITUENT_NAME_FIELDS = ("品种名称", "股票名称", "证券简称", "name")

INTERNAL_REPORT_ROLES = [ROLE_ADVISOR, ROLE_INSTITUTIONAL_SALES, ROLE_COMPLIANCE]
PUBLIC_REPORT_ROLES = [
    ROLE_ADVISOR,
    ROLE_INSTITUTIONAL_SALES,
    ROLE_COMPLIANCE,
    ROLE_OPERATIONS,
    ROLE_TECHNICAL,
]


@dataclass(frozen=True)
class AnnualReportTarget:
    stock_code: str
    stock_name: str
    query_keyword: str
    cninfo_stock: str


@dataclass(frozen=True)
class ConstituentTarget:
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


class FetchFailure(TypedDict):
    stock_code: str
    stage: str
    error: str
    occurred_at: str


class BatchResult(TypedDict):
    records: list[MetadataRecord]
    failures: list[FetchFailure]
    skipped: list[str]


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


def read_json(path: Path, default: Any) -> Any:
    """Read a JSON document, falling back to `default` when absent or corrupt."""
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def window(months: int = 12) -> tuple[str, str]:
    """Return the inclusive [start, end] date window covering the last N months."""
    end = datetime.now(timezone.utc)
    start = end - pd.Timedelta(days=DEFAULT_LOOKBACK_DAYS if months >= 12 else months * 30)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def first_matching_column(frame: pd.DataFrame, candidates: Sequence[str]) -> str:
    """Return the first candidate present in the frame, or fail loudly."""
    for name in candidates:
        if name in frame.columns:
            return name
    raise RuntimeError(
        f"Unexpected provider schema: none of {list(candidates)} in columns {list(frame.columns)}"
    )


def load_index_constituents(
    ak: Any = None, symbol: str = DEFAULT_INDEX_SYMBOL
) -> list[ConstituentTarget]:
    """Load index constituents from the provider at runtime.

    Keeping the list provider-driven avoids a hardcoded code table that would
    silently go stale when the index is rebalanced.
    """
    provider = ak if ak is not None else import_optional_module("akshare")
    frame = provider.index_stock_cons(symbol=symbol)
    if frame is None or getattr(frame, "empty", True):
        raise RuntimeError(f"No index constituents returned for symbol {symbol}")

    code_column = first_matching_column(cast(pd.DataFrame, frame), CONSTITUENT_CODE_FIELDS)
    name_column = first_matching_column(cast(pd.DataFrame, frame), CONSTITUENT_NAME_FIELDS)
    codes = frame[code_column].astype(str).str.strip()
    names = frame[name_column].astype(str).str.strip()

    targets: list[ConstituentTarget] = []
    seen: set[str] = set()
    for code, name in zip(codes, names, strict=False):
        if not code or code in seen:
            continue
        seen.add(code)
        targets.append(ConstituentTarget(stock_code=code, stock_name=name))
    return targets


def ensure_pdf(path: Path) -> None:
    with path.open("rb") as f:
        header = f.read(5)
    if header != b"%PDF-":
        raise RuntimeError(f"Downloaded file is not a PDF: {path}")


def _download_once(url: str, destination: Path) -> None:
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


def download_file(url: str, destination: Path, attempts: int = MAX_DOWNLOAD_ATTEMPTS) -> None:
    """Download with bounded retries so transient throttling does not kill the batch."""
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            _download_once(url, destination)
            return
        except httpx.HTTPError as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)
    raise RuntimeError(f"Download failed after {attempts} attempts: {url} ({last_error})")


def resolve_cninfo_stock(client: httpx.Client, stock_code: str) -> str | None:
    """Resolve the `code,orgId` form cninfo expects, or None when unavailable."""
    response = client.post(
        CNINFO_SEARCH_URL,
        data={"keyWord": stock_code, "maxNum": "10"},
        headers={
            "User-Agent": USER_AGENT,
            "Referer": "http://www.cninfo.com.cn/new/commonUrl?url=disclosure/list/notice",
        },
        timeout=30,
    )
    response.raise_for_status()
    payload = cast(list[dict[str, Any]], response.json() or [])
    for item in payload:
        if str(item.get("code")) == stock_code and item.get("orgId"):
            return f"{stock_code},{item['orgId']}"
    return None


def query_cninfo_annual_report(
    client: httpx.Client,
    target: AnnualReportTarget,
    se_date: str = "",
) -> CninfoAnnouncement:
    start_date, end_date = window() if not se_date else tuple(se_date.split("~"))  # type: ignore[assignment]
    payload = {
        "pageNum": "1",
        "pageSize": "10",
        "column": "szse",
        "tabName": "fulltext",
        "plate": "",
        "stock": target.cninfo_stock,
        "searchkey": target.query_keyword,
        "seDate": f"{start_date}~{end_date}",
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


def read_manifest(path: Path) -> ManifestMetadata | None:
    manifest = read_json(path.with_name(path.name + ".meta.json"), None)
    return manifest if isinstance(manifest, dict) else None


def load_cached_record(output_dir: Path, path: Path) -> MetadataRecord | None:
    """Rebuild a record when the artifact still matches its recorded sha256."""
    if not path.exists():
        return None
    manifest = read_manifest(path)
    if manifest is None or not manifest.get("sha256"):
        return None
    if sha256_file(path) != manifest["sha256"]:
        return None
    cached = dict(manifest)
    cached.pop("relative_path", None)
    return MetadataRecord(relative_path=str(path.relative_to(output_dir)), **cached)  # type: ignore[arg-type]


def new_batch_result() -> BatchResult:
    return {"records": [], "failures": [], "skipped": []}


def collect_target(
    stock_code: str,
    stage: str,
    action: Callable[[], Sequence[MetadataRecord]],
) -> tuple[list[MetadataRecord], FetchFailure | None]:
    """Run one symbol's fetch step, converting failures into a journal entry."""
    try:
        return list(action()), None
    except Exception as exc:  # noqa: BLE001 - one bad symbol must not abort the batch
        return [], {
            "stock_code": stock_code,
            "stage": stage,
            "error": f"{type(exc).__name__}: {exc}",
            "occurred_at": utc_now(),
        }


def fetch_annual_reports(
    output_dir: Path,
    constituents: Sequence[ConstituentTarget],
    *,
    index_provider: Any = None,
) -> BatchResult:
    result = new_batch_result()
    report_dir = output_dir / "announcements"

    with httpx.Client(follow_redirects=True) as client:
        for target in constituents:
            records, failure = collect_target(
                target.stock_code,
                "annual_report",
                lambda t=target: _fetch_one_annual_report(
                    client, output_dir, report_dir, t, index_provider
                ),
            )
            result["records"].extend(records)
            if failure is not None:
                result["failures"].append(failure)
    return result


def _fetch_one_annual_report(
    client: httpx.Client,
    output_dir: Path,
    report_dir: Path,
    target: ConstituentTarget,
    index_provider: Any = None,
) -> list[MetadataRecord]:
    cninfo_stock = index_provider(target.stock_code) if index_provider is not None else None
    if cninfo_stock is None:
        cninfo_stock = resolve_cninfo_stock(client, target.stock_code)
    if cninfo_stock is None:
        raise RuntimeError(f"Unable to resolve cninfo orgId for {target.stock_code}")

    announcement = query_cninfo_annual_report(
        client,
        AnnualReportTarget(
            stock_code=target.stock_code,
            stock_name=target.stock_name,
            query_keyword="年度报告",
            cninfo_stock=cninfo_stock,
        ),
    )
    title = clean_text(announcement["announcementTitle"])
    date = (
        pd.to_datetime(announcement["announcementTime"], unit="ms", utc=True)
        .tz_convert("Asia/Shanghai")
        .strftime("%Y-%m-%d")
    )
    url = f"{CNINFO_STATIC_URL}/{announcement['adjunctUrl']}"
    path = report_dir / f"{target.stock_code}_{safe_filename(title)}.pdf"

    cached = load_cached_record(output_dir, path)
    if cached is not None:
        return [MetadataRecord(**cached)]

    download_file(url, path)
    ensure_pdf(path)
    return [
        {
            "relative_path": str(path.relative_to(output_dir)),
            META_DOC_TYPE: DOC_TYPE_ANNOUNCEMENT,
            META_RETRIEVAL_SOURCE: SOURCE_REPORT,
            META_PERMISSION_LEVEL: PERMISSION_INTERNAL,
            META_ALLOWED_ROLES: list(INTERNAL_REPORT_ROLES),
            META_TITLE: f"{target.stock_name}{title}",
            META_DATE: date,
            META_STOCK_CODE: target.stock_code,
            META_SOURCE: url,
            "provider": "cninfo",
            "sha256": sha256_file(path),
        }
    ]


def fetch_research_reports(
    output_dir: Path,
    constituents: Sequence[ConstituentTarget],
    *,
    ak: Any = None,
) -> BatchResult:
    provider = ak if ak is not None else import_optional_module("akshare")
    result = new_batch_result()
    report_dir = output_dir / "reports"
    indexes: list[pd.DataFrame] = []

    for target in constituents:
        records, failure = collect_target(
            target.stock_code,
            "research_report",
            lambda t=target: _fetch_one_research_report(
                provider, output_dir, report_dir, t, indexes
            ),
        )
        result["records"].extend(records)
        if failure is not None:
            result["failures"].append(failure)

    _write_research_index(output_dir, indexes, result)
    return result


def _fetch_one_research_report(
    provider: Any,
    output_dir: Path,
    report_dir: Path,
    target: ConstituentTarget,
    indexes: list[pd.DataFrame],
) -> list[MetadataRecord]:
    frame = provider.stock_research_report_em(symbol=target.stock_code)
    if frame is None or getattr(frame, "empty", True):
        raise RuntimeError(f"No research report found for {target.stock_code}")

    rows = frame.head(1)
    row = rows.iloc[0]
    pdf_url = str(row["报告PDF链接"])
    title = clean_text(row["报告名称"])
    path = report_dir / f"{target.stock_code}_{safe_filename(title)}.pdf"

    cached = load_cached_record(output_dir, path)
    if cached is not None:
        indexes.append(frame.head(5).assign(sample_stock_code=target.stock_code))
        return [MetadataRecord(**cached)]

    download_file(pdf_url, path)
    ensure_pdf(path)
    indexes.append(frame.head(5).assign(sample_stock_code=target.stock_code))
    return [
        {
            "relative_path": str(path.relative_to(output_dir)),
            META_DOC_TYPE: DOC_TYPE_RESEARCH_REPORT,
            META_RETRIEVAL_SOURCE: SOURCE_REPORT,
            META_PERMISSION_LEVEL: PERMISSION_PUBLIC,
            META_ALLOWED_ROLES: list(PUBLIC_REPORT_ROLES),
            META_TITLE: f"{target.stock_name}{title}",
            META_DATE: str(row["日期"]),
            META_STOCK_CODE: target.stock_code,
            META_SOURCE: pdf_url,
            "provider": "akshare/eastmoney",
            "institution": str(row.get("机构", "")),
            "rating": str(row.get("东财评级", "")),
            "sha256": sha256_file(path),
        }
    ]


def _write_research_index(
    output_dir: Path, indexes: list[pd.DataFrame], result: BatchResult
) -> None:
    if not indexes:
        return
    path = output_dir / RESEARCH_INDEX_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    merged = pd.concat(indexes, ignore_index=True)
    merged.to_csv(path, index=False)
    result["records"].append(
        {
            "relative_path": str(path.relative_to(output_dir)),
            META_DOC_TYPE: DOC_TYPE_FINANCIAL_DATA,
            META_RETRIEVAL_SOURCE: "sql_query",
            META_PERMISSION_LEVEL: PERMISSION_INTERNAL,
            META_ALLOWED_ROLES: list(INTERNAL_REPORT_ROLES),
            META_TITLE: "AKShare 东方财富研报索引",
            META_DATE: utc_now()[:10],
            META_STOCK_CODE: "",
            META_SOURCE: "akshare.stock_research_report_em",
            "provider": "akshare/eastmoney",
            "sha256": sha256_file(path),
        }
    )


def fetch_quote_history(
    output_dir: Path,
    constituents: Sequence[ConstituentTarget],
    *,
    ef: Any = None,
) -> BatchResult:
    provider = ef if ef is not None else import_optional_module("efinance")
    result = new_batch_result()
    start_date, end_date = window()

    for target in constituents:
        records, failure = collect_target(
            target.stock_code,
            "quote_history",
            lambda t=target: _fetch_one_quote_history(
                provider, output_dir, t, start_date, end_date
            ),
        )
        result["records"].extend(records)
        if failure is not None:
            result["failures"].append(failure)
    return result


def _fetch_one_quote_history(
    provider: Any,
    output_dir: Path,
    target: ConstituentTarget,
    start_date: str,
    end_date: str,
) -> list[MetadataRecord]:
    frame = provider.stock.get_quote_history(
        target.stock_code, beg=start_date.replace("-", ""), end=end_date.replace("-", "")
    )
    if frame is None or getattr(frame, "empty", True):
        raise RuntimeError(f"No quote history returned for {target.stock_code}")

    renamed = frame.rename(
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
    renamed["year"] = renamed["date"].astype(str).str[:4]
    renamed["provider"] = "efinance/eastmoney"

    path = output_dir / "financials" / f"efinance_{target.stock_code}_quote_history.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    renamed.to_csv(path, index=False)
    return [
        {
            "relative_path": str(path.relative_to(output_dir)),
            META_DOC_TYPE: DOC_TYPE_FINANCIAL_DATA,
            META_RETRIEVAL_SOURCE: "sql_query",
            META_PERMISSION_LEVEL: PERMISSION_INTERNAL,
            META_ALLOWED_ROLES: list(INTERNAL_REPORT_ROLES),
            META_TITLE: f"{target.stock_name} 行情快照",
            META_DATE: end_date,
            META_STOCK_CODE: target.stock_code,
            META_SOURCE: "efinance.stock.get_quote_history",
            "provider": "efinance/eastmoney",
            "sha256": sha256_file(path),
        }
    ]


def fetch_valuation_history(
    output_dir: Path,
    constituents: Sequence[ConstituentTarget],
    *,
    bs: Any = None,
) -> BatchResult:
    """Fetch valuation history, keeping one shared login for the whole batch."""
    result = new_batch_result()
    provider = bs if bs is not None else import_optional_module("baostock")
    start_date, end_date = window()

    login = provider.login()
    if getattr(login, "error_code", "0") != "0":
        raise RuntimeError(f"baostock login failed: {getattr(login, 'error_msg', '')}")
    try:
        for target in constituents:
            records, failure = collect_target(
                target.stock_code,
                "valuation_history",
                lambda t=target: _fetch_one_valuation_history(
                    provider, output_dir, t, start_date, end_date
                ),
            )
            result["records"].extend(records)
            if failure is not None:
                result["failures"].append(failure)
    finally:
        provider.logout()
    return result


def _prefixed_code(stock_code: str) -> str:
    return f"sh.{stock_code}" if stock_code.startswith(("6", "9")) else f"sz.{stock_code}"


def _fetch_one_valuation_history(
    provider: Any,
    output_dir: Path,
    target: ConstituentTarget,
    start_date: str,
    end_date: str,
) -> list[MetadataRecord]:
    fields = "date,code,open,high,low,close,volume,amount,turn,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM"
    query = provider.query_history_k_data_plus(
        _prefixed_code(target.stock_code),
        fields,
        start_date=start_date,
        end_date=end_date,
        frequency="d",
        adjustflag="3",
    )
    rows: list[list[str]] = []
    while query.error_code == "0" and query.next():
        rows.append(query.get_row_data())
    if query.error_code != "0":
        raise RuntimeError(f"baostock query failed: {query.error_msg}")
    if not rows:
        raise RuntimeError(f"No valuation rows for {target.stock_code}")

    frame = pd.DataFrame(rows, columns=query.fields)
    frame["year"] = frame["date"].str[:4]
    frame["provider"] = "baostock"

    path = output_dir / "financials" / f"baostock_{target.stock_code}_valuation.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return [
        {
            "relative_path": str(path.relative_to(output_dir)),
            META_DOC_TYPE: DOC_TYPE_FINANCIAL_DATA,
            META_RETRIEVAL_SOURCE: "sql_query",
            META_PERMISSION_LEVEL: PERMISSION_INTERNAL,
            META_ALLOWED_ROLES: list(INTERNAL_REPORT_ROLES),
            META_TITLE: f"{target.stock_name} 估值行情",
            META_DATE: end_date,
            META_STOCK_CODE: target.stock_code,
            META_SOURCE: "baostock.query_history_k_data_plus",
            "provider": "baostock",
            "sha256": sha256_file(path),
        }
    ]


def write_metadata(output_dir: Path, records: list[MetadataRecord]) -> None:
    for record in records:
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
        source_path = output_dir / record["relative_path"]
        manifest = source_path.with_name(source_path.name + ".meta.json")
        write_json(manifest, manifest_record)
        print(f"metadata: {manifest}")


def merge_results(results: Sequence[BatchResult]) -> BatchResult:
    merged = new_batch_result()
    for result in results:
        merged["records"].extend(result["records"])
        merged["failures"].extend(result["failures"])
        merged["skipped"].extend(result["skipped"])
    return merged


def run_batch(
    output_dir: Path,
    *,
    symbol: str = DEFAULT_INDEX_SYMBOL,
    limit: int | None = None,
    ak: Any = None,
    ef: Any = None,
    bs: Any = None,
) -> BatchResult:
    """Run the full incremental batch and persist watermark plus failure journal."""
    output_dir.mkdir(parents=True, exist_ok=True)
    constituents = load_index_constituents(ak, symbol=symbol)
    if limit is not None:
        constituents = constituents[:limit]
    print(f"constituents: {len(constituents)}")

    result = merge_results(
        [
            fetch_annual_reports(output_dir, constituents),
            fetch_research_reports(output_dir, constituents, ak=ak),
            fetch_quote_history(output_dir, constituents, ef=ef),
            fetch_valuation_history(output_dir, constituents, bs=bs),
        ]
    )
    write_metadata(output_dir, result["records"])

    write_json(
        output_dir / STATE_FILENAME,
        {
            "index_symbol": symbol,
            "last_run_at": utc_now(),
            "symbols_total": len(constituents),
            "symbols_failed": len({item["stock_code"] for item in result["failures"]}),
            "artifacts": len(result["records"]),
        },
    )
    write_json(output_dir / FAILURES_FILENAME, result["failures"])
    return result


def main() -> None:
    run_batch(DEFAULT_OUTPUT_DIR)


if __name__ == "__main__":
    main()
