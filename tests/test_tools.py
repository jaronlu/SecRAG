from __future__ import annotations

import json

from src.tools.calculator import calculator, safe_eval
from src.tools.suitability import suitability_check


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
