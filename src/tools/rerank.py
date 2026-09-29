"""BGE reranker tool for retrieval results."""

from __future__ import annotations

import json
from importlib import import_module
from typing import Any

from langchain_core.tools import tool

from src.config import config
from src.schemas.constants import RR_CONTENT, RR_SCORE

DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"


def rerank_model_name() -> str:
    """配置的 reranker 模型名（architecture.md §5.1 RERANK_MODEL）。

    空配置回落默认模型，保证配置缺失时行为与默认一致。
    """
    return config.rerank_model or DEFAULT_RERANK_MODEL


class RerankerNotConfigured(RuntimeError):
    """FlagEmbedding 或 reranker 模型未配置。

    与运行期失败（模型崩溃、分数不一致等）区分：调用方据此走显式
    "unavailable" 降级路径，而不是把"未配置"记成执行失败（ISSUE-10）。
    """


def reranker_available() -> bool:
    """FlagEmbedding 可导入即视为 reranker 已配置。

    供工具注册表决定是否向 LLM 暴露 rerank_tool；模型加载失败在
    实际调用时经 RerankerNotConfigured/RuntimeError 显式报错。
    """
    try:
        import_module("FlagEmbedding")
    except ImportError:
        return False
    return True


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
            raise RerankerNotConfigured(
                "未配置 BGE reranker 模型；请安装并配置 FlagEmbedding/BAAI bge-reranker-v2-m3"
            ) from exc

        # 模型名来自配置（architecture.md §5.1 RERANK_MODEL），便于本地化/离线。
        # 权重不可获取（未本地化且无法联网）属于"未配置"而非运行期故障：
        # 归一到 RerankerNotConfigured，让调用方走显式 unavailable 降级
        try:
            self.model = flag_embedding.FlagAutoReranker.from_finetuned(
                model_name_or_path=rerank_model_name(),
                use_fp16=True,
            )
        except OSError as exc:
            raise RerankerNotConfigured(
                f"无法加载 BGE reranker 权重 {rerank_model_name()}；请先本地化模型: {exc}"
            ) from exc
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