"""Rerank tool for retrieval results."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import tool

from src.schemas.constants import RR_SCORE


def rerank_documents(documents: list[dict[str, Any]], top_k: int = 5) -> list[dict[str, Any]]:
    """Return top-k documents sorted by existing relevance score."""
    return sorted(
        documents,
        key=lambda doc: float(doc.get(RR_SCORE, 0) or 0),
        reverse=True,
    )[:top_k]


@tool
def rerank_tool(query: str, documents: str, top_k: int = 5) -> str:
    """Rerank JSON retrieval results; model-backed reranking can replace the score sort later."""
    try:
        docs = json.loads(documents)
        if not isinstance(docs, list):
            raise ValueError("documents 必须是 JSON 数组")
        return json.dumps(rerank_documents(docs, top_k=top_k), ensure_ascii=False)
    except (json.JSONDecodeError, ValueError) as exc:
        return f"重排序错误: {exc}"
