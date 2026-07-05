"""BGE reranker tool for retrieval results."""

from __future__ import annotations

import json
from importlib import import_module
from typing import Any

from langchain_core.tools import tool

from src.schemas.constants import RR_CONTENT, RR_SCORE

DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"


class RerankService:
    _instance: "RerankService | None" = None

    def __new__(cls) -> "RerankService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.model = None
        return cls._instance

    def _ensure_model(self) -> Any:
        if self.model is not None:
            return self.model

        try:
            flag_embedding = import_module("FlagEmbedding")
        except ImportError as exc:
            raise RuntimeError(
                "未配置 BGE reranker 模型；请安装并配置 FlagEmbedding/BAAI bge-reranker-v2-m3"
            ) from exc

        self.model = flag_embedding.FlagAutoReranker.from_finetuned(
            model_name_or_path=DEFAULT_RERANK_MODEL,
            use_fp16=True,
        )
        return self.model

    def rerank(
        self,
        query: str,
        documents: list[dict[str, Any]],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """对检索结果进行重排序。"""
        if not documents:
            return []

        pairs = [(query, str(doc.get(RR_CONTENT, ""))) for doc in documents]
        scores = self._ensure_model().compute_score(pairs)
        if len(scores) != len(documents):
            raise RuntimeError("reranker 返回分数数量与文档数量不一致")

        scored = []
        for doc, score in zip(documents, scores):
            item = dict(doc)
            item[RR_SCORE] = float(score)
            scored.append(item)

        scored.sort(key=lambda doc: doc[RR_SCORE], reverse=True)
        return scored[:top_k]


@tool
def rerank_tool(query: str, documents: str, top_k: int = 5) -> str:
    """对检索结果进行 BGE 重排序，提升精确率。"""
    try:
        docs = json.loads(documents)
        if not isinstance(docs, list):
            raise ValueError("documents 必须是 JSON 数组")
        service = RerankService()
        return json.dumps(service.rerank(query, docs, top_k=top_k), ensure_ascii=False)
    except (json.JSONDecodeError, RuntimeError, ValueError) as exc:
        return f"重排序错误: {exc}"
