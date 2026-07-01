"""Production calculator tool for financial expressions."""

from __future__ import annotations

import ast
import operator
import re
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation

from langchain_core.tools import tool

_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}
_EXPR_CANDIDATE_PATTERN = re.compile(r"[0-9+\-*/().%\s]+")
_PERCENT_PATTERN = re.compile(r"(\d+(?:\.\d+)?)%")
_CHINESE_UNIT_PATTERN = re.compile(r"(\d+(?:\.\d+)?)(万|亿)")
_UNIT_MULTIPLIER = {
    "万": Decimal("10000"),
    "亿": Decimal("100000000"),
}


def _normalize_units(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        value = Decimal(match.group(1))
        unit = match.group(2)
        normalized = value * _UNIT_MULTIPLIER[unit]
        return format(normalized, "f")

    return _CHINESE_UNIT_PATTERN.sub(replace, text)


def _extract_expression(text: str) -> str:
    normalized = _normalize_units(text)
    candidates = []
    for match in _EXPR_CANDIDATE_PATTERN.finditer(normalized):
        candidate = match.group().strip()
        if not candidate:
            continue
        if not any(ch.isdigit() for ch in candidate):
            continue
        if not any(op in candidate for op in ("+", "-", "*", "/", "%", "(", ")")):
            continue
        candidates.append(candidate)

    if not candidates:
        raise ValueError("未找到有效的数学表达式")
    return max(candidates, key=len)


def _replace_percentages(expression: str) -> str:
    return _PERCENT_PATTERN.sub(
        lambda match: str(Decimal(match.group(1)) / Decimal("100")),
        expression,
    )


def safe_eval(expression: str) -> Decimal:
    """Safely evaluate a numeric expression with Decimal precision."""
    tree = ast.parse(_replace_percentages(expression), mode="eval")

    def evaluate(node: ast.AST) -> Decimal:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return Decimal(str(node.value))
            raise ValueError(f"不支持的常量类型: {type(node.value).__name__}")
        if isinstance(node, ast.BinOp):
            left = evaluate(node.left)
            right = evaluate(node.right)
            op = _ALLOWED_OPERATORS.get(type(node.op))
            if op is None:
                raise ValueError(f"不支持的操作符: {type(node.op).__name__}")
            if isinstance(node.op, ast.Div) and right == 0:
                raise ZeroDivisionError("除数不能为零")
            return op(left, right)
        if isinstance(node, ast.UnaryOp):
            operand = evaluate(node.operand)
            op = _ALLOWED_OPERATORS.get(type(node.op))
            if op is None:
                raise ValueError(f"不支持的单目操作符: {type(node.op).__name__}")
            return op(operand)
        raise ValueError(f"不支持的语法节点: {type(node).__name__}")

    return evaluate(tree.body)


@tool
def calculator(expression: str) -> str:
    """Precise financial calculator using Decimal and a safe AST evaluator."""
    try:
        expr = _extract_expression(expression)
        result = safe_eval(expr)
        return str(result.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))
    except (InvalidOperation, ZeroDivisionError, SyntaxError, ValueError) as exc:
        return f"计算错误: {exc}"
