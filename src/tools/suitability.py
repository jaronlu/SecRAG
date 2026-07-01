"""Suitability check tool."""

from __future__ import annotations

import json

from langchain_core.tools import tool

_CLIENT_RISK_LEVELS = {
    "client_conservative": "R1",
    "client_balanced": "R3",
    "client_aggressive": "R5",
}
_PRODUCT_RISK_LEVELS = {
    "product_cash_plus": "R1",
    "product_bond_fund": "R2",
    "product_mixed_fund": "R3",
    "product_equity_fund": "R4",
    "product_private_fund": "R5",
}
_RISK_ORDER = {"R1": 1, "R2": 2, "R3": 3, "R4": 4, "R5": 5}


@tool
def suitability_check(client_id: str, product_id: str) -> str:
    """Check whether a client risk level matches a product risk level."""
    client_risk = _CLIENT_RISK_LEVELS.get(client_id)
    product_risk = _PRODUCT_RISK_LEVELS.get(product_id)

    if client_risk is None or product_risk is None:
        return json.dumps(
            {
                "client_id": client_id,
                "product_id": product_id,
                "matched": False,
                "reason": "缺少客户或产品风险等级映射，请补充主数据。",
            },
            ensure_ascii=False,
        )

    matched = _RISK_ORDER[client_risk] >= _RISK_ORDER[product_risk]
    reason = (
        "客户风险承受能力覆盖产品风险等级。" if matched else "客户风险承受能力低于产品风险等级。"
    )
    return json.dumps(
        {
            "client_id": client_id,
            "product_id": product_id,
            "client_risk_level": client_risk,
            "product_risk_level": product_risk,
            "matched": matched,
            "reason": reason,
        },
        ensure_ascii=False,
    )
