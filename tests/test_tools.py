from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from src.tools.calculator import calculator, safe_eval
from src.tools.financial_ratios import financial_ratios_tool
from src.tools.market_data import market_data_tool
from src.tools.rerank import RerankService, rerank_tool
from src.tools.sql_query import normalize_select_sql, sql_query_tool
from src.tools.suitability import suitability_check
from src.utils.tracing import Tracer


def test_safe_eval_uses_decimal_precision():
    assert str(safe_eval("0.1 + 0.2")) == "0.3"


def test_calculator_formats_four_decimal_places():
    assert calculator.invoke({"expression": "0.1 + 0.2"}) == "0.3000"


def test_calculator_supports_percentage_and_chinese_units():
    assert calculator.invoke({"expression": "申购费 100万 * 1.5%"}) == "15000.0000"


def test_calculator_rejects_invalid_expression():
    result = calculator.invoke({"expression": "请帮我算一下收益"})
    assert "计算错误" in result


def test_suitability_check_returns_match_result():
    payload = json.loads(
        suitability_check.invoke(
            {"client_id": "client_balanced", "product_id": "product_bond_fund"}
        )
    )
    assert payload["matched"] is True
    assert payload["client_risk_level"] == "R3"


def test_suitability_check_handles_missing_master_data():
    payload = json.loads(
        suitability_check.invoke(
            {"client_id": "unknown_client", "product_id": "product_private_fund"}
        )
    )
    assert payload["matched"] is False
    assert "主数据" in payload["reason"]


def _create_financial_db(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            "CREATE TABLE financial_ratios (stock_code TEXT, year INTEGER, report_type TEXT, pe TEXT, pb TEXT, roe TEXT, net_margin TEXT)"
        )
        cursor.execute(
            "INSERT INTO financial_ratios VALUES ('600519', 2024, 'annual', '28.5000', '8.1000', '0.2500', '0.5300')"
        )
        cursor.execute(
            "CREATE TABLE market_history (date TEXT, code TEXT, open TEXT, high TEXT, low TEXT, close TEXT, volume TEXT, amount TEXT, pctChg TEXT)"
        )
        cursor.execute(
            "INSERT INTO market_history VALUES ('2024-01-31', 'sh.600519', '1660.00', '1690.00', '1650.00', '1680.00', '1000', '1680000', '1.2000')"
        )
        conn.commit()
    finally:
        conn.close()


def test_normalize_select_sql_adds_limit():
    assert normalize_select_sql("SELECT * FROM financial_ratios") == "SELECT * FROM financial_ratios LIMIT 100"


def test_normalize_select_sql_rejects_non_select():
    assert normalize_select_sql("DELETE FROM financial_ratios") is None


def test_sql_query_tool_returns_rows(tmp_path):
    db_path = tmp_path / "financial.db"
    _create_financial_db(db_path)

    payload = json.loads(
        sql_query_tool.invoke(
            {"query": "SELECT stock_code, year FROM financial_ratios", "db_path": str(db_path)}
        )
    )
    assert payload[0]["stock_code"] == "600519"


def test_sql_query_tool_rejects_dangerous_sql(tmp_path):
    db_path = tmp_path / "financial.db"
    _create_financial_db(db_path)

    result = sql_query_tool.invoke(
        {"query": "SELECT * FROM financial_ratios; DROP TABLE financial_ratios", "db_path": str(db_path)}
    )
    assert "查询错误" in result


def test_financial_ratios_tool_returns_phase2_skeleton():
    payload = json.loads(
        financial_ratios_tool.invoke(
            {"stock_code": "600519", "year": 2024, "report_type": "annual"}
        )
    )
    assert payload["stock_code"] == "600519"
    assert payload["year"] == 2024
    assert payload["report_type"] == "annual"
    assert payload["missing"] is True
    assert "income_statement" in payload["required_subjects"]


def test_financial_ratios_tool_returns_rows(tmp_path):
    db_path = tmp_path / "financial.db"
    _create_financial_db(db_path)

    payload = json.loads(
        financial_ratios_tool.invoke(
            {
                "stock_code": "600519",
                "year": 2024,
                "report_type": "annual",
                "db_path": str(db_path),
            }
        )
    )
    assert payload[0]["pe"] == "28.5000"


def test_market_data_tool_reads_local_history(tmp_path):
    db_path = tmp_path / "financial.db"
    _create_financial_db(db_path)

    payload = json.loads(
        market_data_tool.invoke(
            {
                "stock_code": "sh.600519",
                "start_date": "2024-01-01",
                "end_date": "2024-01-31",
                "db_path": str(db_path),
            }
        )
    )
    assert payload[0]["close"] == "1680.00"


class FakeRerankModel:
    def compute_score(self, pairs: list[tuple[str, str]]) -> list[float]:
        return [0.9 if content == "a" else 0.1 for _, content in pairs]


def test_rerank_tool_requires_configured_model():
    RerankService().model = None
    docs = json.dumps([{"score": 0.9, "content": "b"}, {"score": 0.1, "content": "a"}])
    result = rerank_tool.invoke({"query": "q", "documents": docs, "top_k": 2})
    assert "重排序错误" in result
    assert "BGE reranker 模型" in result


def test_rerank_tool_uses_model_scores():
    RerankService().model = FakeRerankModel()
    docs = json.dumps([{"score": 0.9, "content": "b"}, {"score": 0.1, "content": "a"}])
    payload = json.loads(rerank_tool.invoke({"query": "q", "documents": docs, "top_k": 2}))
    assert [doc["content"] for doc in payload] == ["a", "b"]
    assert payload[0]["score"] == 0.9
    RerankService().model = None


def test_tracer_records_success_and_error():
    tracer = Tracer()

    @tracer.trace
    def ok(value: int) -> int:
        return value + 1

    @tracer.trace
    def fail() -> None:
        raise ValueError("boom")

    assert ok(1) == 2
    try:
        fail()
    except ValueError:
        pass

    payload = tracer.to_dict()
    assert payload["total_calls"] == 2
    assert payload["success_rate"] == 0.5
