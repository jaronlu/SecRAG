from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.tools.calculator import calculator, safe_eval
from src.tools.financial_ratios import financial_ratios_tool, query_financial_ratios
from src.tools.market_data import market_data_tool, query_market_data
from src.tools.rerank import RerankService, rerank_tool
from src.tools.sql_query import normalize_select_sql, run_select_query, sql_query_tool
from src.tools.suitability import suitability_check
from src.utils.tracing import Tracer
from src.ingestion.financial_store import import_research_reports_index
from src.schemas.constants import SOURCE_SQL


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
        suitability_check.invoke({
            "client_id": "client_balanced",
            "product_id": "product_bond_fund",
        })
    )
    assert payload["matched"] is True
    assert payload["client_risk_level"] == "R3"


def test_suitability_check_handles_missing_master_data():
    payload = json.loads(
        suitability_check.invoke({
            "client_id": "unknown_client",
            "product_id": "product_private_fund",
        })
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
    assert (
        normalize_select_sql("SELECT * FROM financial_ratios")
        == "SELECT * FROM financial_ratios LIMIT 100"
    )


def test_normalize_select_sql_rejects_non_select():
    assert normalize_select_sql("DELETE FROM financial_ratios") is None


def test_normalize_select_sql_rejects_unknown_table():
    assert normalize_select_sql("SELECT * FROM users") is None


def test_normalize_select_sql_rejects_unknown_column():
    assert normalize_select_sql("SELECT password FROM financial_ratios") is None


def test_normalize_select_sql_rejects_join_union_subquery_and_oversized_limit():
    assert normalize_select_sql(
        "SELECT * FROM financial_ratios JOIN income_statement USING (stock_code)"
    ) is None
    assert normalize_select_sql(
        "SELECT stock_code FROM financial_ratios UNION SELECT stock_code FROM income_statement"
    ) is None
    assert normalize_select_sql(
        "SELECT * FROM financial_ratios WHERE stock_code IN "
        "(SELECT stock_code FROM income_statement)"
    ) is None
    assert normalize_select_sql("SELECT * FROM financial_ratios LIMIT 101") is None


def test_run_select_query_returns_rows(tmp_path):
    db_path = tmp_path / "financial.db"
    _create_financial_db(db_path)

    rows = run_select_query(
        query="SELECT stock_code, year FROM financial_ratios",
        db_path=db_path,
    )
    assert rows[0]["stock_code"] == "600519"


def test_import_and_query_research_report_index(tmp_path):
    csv_path = tmp_path / "reports.csv"
    csv_path.write_text(
        "序号,股票代码,股票简称,报告名称,东财评级,机构,近一月个股研报数,"
        "2026-盈利预测-收益,2026-盈利预测-市盈率,2027-盈利预测-收益,"
        "2027-盈利预测-市盈率,2028-盈利预测-收益,2028-盈利预测-市盈率,"
        "行业,日期,报告PDF链接,sample_stock_code\n"
        "1,000001,平安银行,年报点评,中性,国信证券,0,2.08,5.3,2.09,5.3,2.11,5.2,"
        "银行,2026-04-26,https://example.com/report.pdf,000001\n",
        encoding="utf-8",
    )
    db_path = tmp_path / "financial.db"

    assert import_research_reports_index(csv_path, db_path) == 1
    rows = run_select_query(
        "SELECT stock_code, report_name, institution FROM research_reports_index",
        db_path=db_path,
    )
    assert rows == [
        {
            "stock_code": "000001",
            "report_name": "年报点评",
            "institution": "国信证券",
        }
    ]


def test_sql_query_tool_does_not_expose_db_path():
    assert sql_query_tool.name == SOURCE_SQL
    assert "db_path" not in sql_query_tool.args


def test_sql_query_tool_rejects_dangerous_sql(tmp_path):
    db_path = tmp_path / "financial.db"
    _create_financial_db(db_path)

    result = run_select_query(
        query="SELECT * FROM financial_ratios",
        db_path=db_path,
    )
    assert result

    unsafe = normalize_select_sql("SELECT * FROM financial_ratios; DROP TABLE financial_ratios")
    assert unsafe is None

    with pytest.raises(ValueError):
        sql_query_tool.invoke({"query": "SELECT * FROM financial_ratios; DROP TABLE x"})


def test_sql_query_tool_raises_on_disallowed_table_or_column():
    with pytest.raises(ValueError):
        sql_query_tool.invoke({"query": "SELECT secret FROM not_allowed"})


def test_financial_ratios_tool_raises_on_invalid_input():
    with pytest.raises(ValueError):
        query_financial_ratios("bad code!")
    with pytest.raises(ValueError):
        query_financial_ratios("600519", report_type="bad type!")
    with pytest.raises(ValueError):
        financial_ratios_tool.invoke({"stock_code": "bad code!"})


def test_market_data_tool_raises_on_invalid_input():
    with pytest.raises(ValueError):
        query_market_data("bad code!", db_path="unused.db")
    with pytest.raises(ValueError):
        query_market_data("600519", fields="close; drop table x", db_path="unused.db")


class _RaisingImport:
    def __call__(self, name):
        raise ImportError(f"No module named {name!r}")


def test_market_data_available_reflects_local_tables_and_baostock(monkeypatch, tmp_path):
    """ISSUE-19：可用性 = 本地行情表有数据 或 baostock 可导入。"""
    from src.tools.market_data import market_data_available

    # baostock 可导入 → 可用
    monkeypatch.setattr(
        "src.tools.market_data.import_module", lambda name: SimpleNamespace()
    )
    assert market_data_available(db_path=tmp_path / "nope.db") is True

    # baostock 缺失 + 本地 market_history 有数据 → 可用
    monkeypatch.setattr("src.tools.market_data.import_module", _RaisingImport())
    db = tmp_path / "financial.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE market_history (date TEXT, code TEXT)")
        conn.execute("INSERT INTO market_history VALUES ('2026-01-05', '600519')")
    assert market_data_available(db_path=db) is True

    # baostock 缺失 + 无行情表 → 不可用
    assert market_data_available(db_path=tmp_path / "empty.db") is False

    # 表存在但无数据 → 不可用
    empty_table_db = tmp_path / "empty_table.db"
    with sqlite3.connect(empty_table_db) as conn:
        conn.execute("CREATE TABLE market_snapshot (date TEXT, code TEXT)")
    assert market_data_available(db_path=empty_table_db) is False


def test_market_data_tool_hidden_while_unconfigured(monkeypatch):
    """ISSUE-19：数据源未配置时摘除 market_data_tool，恒失败工具不得暴露给 LLM。"""
    from src.agents.tools import get_tools_for_role

    monkeypatch.setattr("src.agents.tools.market_data_available", lambda: False)

    for role in ("advisor", "technical", "operations"):
        names = {tool_item.name for tool_item in get_tools_for_role(role)}
        assert "market_data_tool" not in names


def test_market_data_tool_visible_when_configured(monkeypatch):
    from src.agents.tools import get_tools_for_role

    monkeypatch.setattr("src.agents.tools.market_data_available", lambda: True)

    names = {tool_item.name for tool_item in get_tools_for_role("technical")}
    assert "market_data_tool" in names


def test_financial_ratios_tool_returns_phase2_skeleton():
    payload = json.loads(
        financial_ratios_tool.invoke({
            "stock_code": "600519",
            "year": 2024,
            "report_type": "annual",
        })
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
        query_financial_ratios(
            stock_code="600519",
            year=2024,
            report_type="annual",
            db_path=str(db_path),
        )
    )
    assert payload[0]["pe"] == "28.5000"
    assert "db_path" not in financial_ratios_tool.args


def test_market_data_tool_reads_local_history(tmp_path):
    db_path = tmp_path / "financial.db"
    _create_financial_db(db_path)

    payload = json.loads(
        query_market_data(
            stock_code="sh.600519",
            start_date="2024-01-01",
            end_date="2024-01-31",
            db_path=str(db_path),
        )
    )
    assert payload[0]["close"] == "1680.00"
    assert "db_path" not in market_data_tool.args


class FakeRerankModel:
    def compute_score(self, pairs: list[tuple[str, str]]) -> list[float]:
        return [0.9 if content == "a" else 0.1 for _, content in pairs]


def test_rerank_tool_requires_configured_model(monkeypatch):
    """未配置 reranker（FlagEmbedding 缺失）时工具必须返回显式错误文本。"""
    from src.tools import rerank as rerank_module

    def _missing(name):
        raise ImportError(f"No module named {name!r}")

    monkeypatch.setattr(rerank_module, "import_module", _missing)
    RerankService().model = None
    docs = json.dumps([{"score": 0.9, "content": "b"}, {"score": 0.1, "content": "a"}])
    result = rerank_tool.invoke({"query": "q", "documents": docs, "top_k": 2})
    assert "重排序错误" in result
    assert "BGE reranker 模型" in result


def test_unloadable_reranker_weights_are_reported_as_not_configured(monkeypatch):
    """ISSUE-27：权重取不到（未本地化/无法联网）属"未配置"，不得抛成节点故障。"""
    from types import SimpleNamespace

    from src.tools import rerank as rerank_module
    from src.tools.rerank import RerankerNotConfigured

    class _FailingAutoReranker:
        @staticmethod
        def from_finetuned(model_name_or_path, use_fp16):
            raise OSError("couldn't find them in the cached files")

    monkeypatch.setattr(
        rerank_module,
        "import_module",
        lambda name: SimpleNamespace(FlagAutoReranker=_FailingAutoReranker),
    )
    RerankService().model = None

    with pytest.raises(RerankerNotConfigured):
        RerankService().rerank("q", [{"content": "a", "score": 0.0}], top_k=1)


def test_reranker_available_false_when_flagembedding_missing(monkeypatch):
    """ISSUE-10：FlagEmbedding 不可导入 = reranker 未配置，探测必须为 False。"""
    from src.tools import rerank as rerank_module

    def _missing(name):
        raise ImportError(f"No module named {name!r}")

    monkeypatch.setattr(rerank_module, "import_module", _missing)
    assert rerank_module.reranker_available() is False


def test_reranker_available_true_when_flagembedding_importable(monkeypatch):
    from src.tools import rerank as rerank_module

    monkeypatch.setattr(rerank_module, "import_module", lambda name: object())
    assert rerank_module.reranker_available() is True


def test_rerank_tool_uses_model_scores():
    RerankService().model = FakeRerankModel()
    docs = json.dumps([{"score": 0.9, "content": "b"}, {"score": 0.1, "content": "a"}])
    payload = json.loads(rerank_tool.invoke({"query": "q", "documents": docs, "top_k": 2}))
    assert [doc["content"] for doc in payload] == ["a", "b"]
    assert payload[0]["score"] == 0.9
    RerankService().model = None


# ══════════════════════════════════════════════════════════════════════
# ISSUE-27：reranker 模型名来自配置，且真实模型可加载执行
# ══════════════════════════════════════════════════════════════════════


def test_rerank_service_loads_configured_model(monkeypatch):
    """模型名取自 config.rerank_model（architecture.md §5.1 RERANK_MODEL）。"""
    from types import SimpleNamespace

    from src.tools import rerank as rerank_module

    loaded: list[str] = []

    class _FakeAutoReranker:
        @staticmethod
        def from_finetuned(model_name_or_path, use_fp16):
            loaded.append(model_name_or_path)
            return FakeRerankModel()

    monkeypatch.setattr(
        rerank_module,
        "import_module",
        lambda name: SimpleNamespace(FlagAutoReranker=_FakeAutoReranker),
    )
    monkeypatch.setattr(
        rerank_module, "config", SimpleNamespace(rerank_model="local/bge-reranker-v2-m3")
    )
    RerankService().model = None

    RerankService().rerank("q", [{"content": "a", "score": 0.0}], top_k=1)

    assert loaded == ["local/bge-reranker-v2-m3"]
    RerankService().model = None


def test_rerank_model_name_falls_back_to_default(monkeypatch):
    from types import SimpleNamespace

    from src.tools import rerank as rerank_module

    monkeypatch.setattr(rerank_module, "config", SimpleNamespace(rerank_model=""))

    assert rerank_module.rerank_model_name() == rerank_module.DEFAULT_RERANK_MODEL


def test_reranker_available_with_installed_flagembedding():
    """ISSUE-27：FlagEmbedding 已纳入依赖，探测必须为 True。"""
    from src.tools.rerank import reranker_available

    assert reranker_available() is True


def test_local_reranker_model_actually_reranks():
    """ISSUE-27 验收：本地 BGE reranker 真实执行语义重排。

    模型未本地化时跳过（离线环境无法下载权重），跳过原因写入报告。
    """
    from src.tools.rerank import RerankService

    service = RerankService()
    if service.model is None:
        try:
            service._ensure_model()
        except Exception as exc:  # pragma: no cover - 依赖本地模型权重
            pytest.skip(f"本地 reranker 模型不可用: {type(exc).__name__}: {exc}")

    documents = [
        {"content": "本基金主要投资于货币市场工具，风险等级为 R1（低风险）。", "score": 0.10},
        {"content": "股票型基金投资于股票市场，净值波动较大，风险等级为 R5。", "score": 0.90},
    ]

    reranked = service.rerank("货币基金的风险等级是什么？", documents, top_k=2)

    assert len(reranked) == 2
    # 交叉编码器应把货币基金段落排在股票基金之前，覆盖原始 score 顺序
    assert "R1" in reranked[0]["content"]
    assert reranked[0]["score"] > reranked[1]["score"]


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
