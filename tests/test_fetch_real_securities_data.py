"""Batch orchestration tests for the real securities fetch script.

All providers are faked, so the suite exercises orchestration (idempotency,
failure isolation, runtime constituent resolution) without network access or
the optional akshare / efinance / baostock dependencies.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.fetch_real_securities_data import (  # noqa: E402
    fetch_research_reports,
    first_matching_column,
    load_index_constituents,
    run_batch,
)

PDF_STUB = b"%PDF-1.4\nstub-content"
QUOTE_COLUMNS = ["股票名称", "股票代码", "日期", "开盘", "收盘", "最高", "最低", "成交量"]


def write_stub_pdf(url: str, destination: Path, attempts: int = 3) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(PDF_STUB)


class FakeAk:
    """Minimal akshare stand-in with controllable per-symbol failures."""

    def __init__(self, failing: set[str] | None = None) -> None:
        self.failing = failing or set()

    def index_stock_cons(self, symbol: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "品种代码": ["000001", "600519", "300750"],
                "品种名称": ["平安银行", "贵州茅台", "宁德时代"],
            }
        )

    def stock_research_report_em(self, symbol: str) -> pd.DataFrame:
        if symbol in self.failing:
            raise RuntimeError(f"provider unavailable for {symbol}")
        return pd.DataFrame(
            [
                {
                    "报告名称": f"{symbol} 深度报告",
                    "报告PDF链接": f"https://example.test/{symbol}.pdf",
                    "日期": "2026-01-05",
                    "机构": "示例机构",
                    "东财评级": "增持",
                }
            ]
        )


class _FakeStockApi:
    def __init__(self, failing: set[str] | None = None) -> None:
        self.failing = failing or set()

    def get_quote_history(self, code: str, beg: str, end: str) -> pd.DataFrame:
        if code in self.failing:
            raise RuntimeError(f"quote unavailable for {code}")
        return pd.DataFrame(
            [dict(zip(QUOTE_COLUMNS, ["示例", code, "2026-01-05", 1, 2, 3, 0, 10]))]
        )


class FakeEf:
    def __init__(self, failing: set[str] | None = None) -> None:
        self.stock = _FakeStockApi(failing)


class _FakeLogin:
    error_code = "0"
    error_msg = ""


class _FakeQuery:
    def __init__(self, rows: list[list[str]], fields: list[str]) -> None:
        self._rows = rows
        self._index = 0
        self.error_code = "0"
        self.error_msg = ""
        self.fields = fields

    def next(self) -> bool:
        self._index += 1
        return self._index <= len(self._rows)

    def get_row_data(self) -> list[str]:
        return self._rows[self._index - 1]


class FakeBs:
    def __init__(self, failing: set[str] | None = None) -> None:
        self.failing = failing or set()

    def login(self) -> _FakeLogin:
        return _FakeLogin()

    def logout(self) -> None:
        return None

    def query_history_k_data_plus(
        self,
        code: str,
        fields: str,
        start_date: str,
        end_date: str,
        frequency: str,
        adjustflag: str,
    ) -> _FakeQuery:
        if code.split(".")[-1] in self.failing:
            raise RuntimeError(f"valuation unavailable for {code}")
        columns = fields.split(",")
        return _FakeQuery([["2026-01-05"] + ["1"] * (len(columns) - 1)], columns)


@pytest.fixture()
def patched_download(monkeypatch: pytest.MonkeyPatch):
    calls: list[str] = []

    def tracking_download(url: str, destination: Path, attempts: int = 3) -> None:
        calls.append(url)
        write_stub_pdf(url, destination, attempts)

    module = sys.modules["scripts.fetch_real_securities_data"]
    monkeypatch.setattr(module, "download_file", tracking_download)
    monkeypatch.setattr(module, "resolve_cninfo_stock", lambda client, code: f"{code},fakeOrg")
    monkeypatch.setattr(
        module,
        "query_cninfo_annual_report",
        lambda client, target, se_date="": {
            "secCode": target.stock_code,
            "announcementTitle": f"{target.stock_name}年度报告",
            "announcementTime": 1767225600000,
            "adjunctUrl": f"/fake/{target.stock_code}.pdf",
        },
    )
    return calls


def test_constituents_are_resolved_from_provider_not_hardcoded():
    targets = load_index_constituents(FakeAk())

    assert [item.stock_code for item in targets] == ["000001", "600519", "300750"]
    assert [item.stock_name for item in targets] == ["平安银行", "贵州茅台", "宁德时代"]


def test_constituent_schema_drift_fails_loudly():
    class DriftedProvider:
        def index_stock_cons(self, symbol: str) -> pd.DataFrame:
            return pd.DataFrame({"ticker": ["000001"], "display": ["平安银行"]})

    with pytest.raises(RuntimeError, match="Unexpected provider schema"):
        load_index_constituents(DriftedProvider())


def test_first_matching_column_rejects_unknown_frame():
    frame = pd.DataFrame({"ticker": ["000001"]})

    with pytest.raises(RuntimeError, match="Unexpected provider schema"):
        first_matching_column(frame, ("品种代码",))


def test_repeated_runs_are_idempotent(tmp_path, patched_download):
    output_dir = tmp_path / "securities"

    first = run_batch(output_dir, limit=3, ak=FakeAk(), ef=FakeEf(), bs=FakeBs())
    downloads_after_first = len(patched_download)
    second = run_batch(output_dir, limit=3, ak=FakeAk(), ef=FakeEf(), bs=FakeBs())

    assert len(first["records"]) == len(second["records"]) > 0
    assert len(patched_download) == downloads_after_first
    assert (output_dir / ".fetch_state.json").exists()


def test_one_failing_symbol_does_not_abort_the_batch(tmp_path, patched_download):
    output_dir = tmp_path / "securities"
    failing = {"600519"}

    result = run_batch(
        output_dir,
        limit=3,
        ak=FakeAk(failing),
        ef=FakeEf(failing),
        bs=FakeBs(failing),
    )

    failed_symbols = {item["stock_code"] for item in result["failures"]}
    succeeded_symbols = {record["stock_code"] for record in result["records"]}
    assert failed_symbols == failing
    assert "000001" in succeeded_symbols
    assert "300750" in succeeded_symbols


def test_failure_journal_records_stage_and_error(tmp_path, patched_download):
    output_dir = tmp_path / "securities"

    fetch_research_reports(output_dir, load_index_constituents(FakeAk({"600519"})), ak=FakeAk())

    result = fetch_research_reports(
        output_dir, load_index_constituents(FakeAk()), ak=FakeAk({"600519"})
    )
    failure = result["failures"][0]
    assert failure["stock_code"] == "600519"
    assert failure["stage"] == "research_report"
    assert "provider unavailable" in failure["error"]
    assert failure["occurred_at"]


def test_state_file_tracks_symbol_counts(tmp_path, patched_download):
    output_dir = tmp_path / "securities"

    run_batch(output_dir, limit=2, ak=FakeAk(), ef=FakeEf(), bs=FakeBs())

    import json

    state = json.loads((output_dir / ".fetch_state.json").read_text(encoding="utf-8"))
    assert state["index_symbol"] == "000300"
    assert state["symbols_total"] == 2
    assert state["symbols_failed"] == 0
    assert state["artifacts"] > 0
