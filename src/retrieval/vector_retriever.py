"""ChromaDB 向量检索器"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, cast

import chromadb

# TYPE_CHECKING 块：只在 IDE/pyright/mypy 检查时生效
# 避免在运行时导入可能很重的类型定义
if TYPE_CHECKING:
    from chromadb.api.types import QueryResult

# ⚡ 字段统一：使用常量而非裸字符串
from src.config import config
from src.retrieval.base import BaseRetriever
from src.schemas.constants import (
    CHROMA_COLLECTION_NAME,
    CHROMA_EMBEDDING_MODEL_KEY,
    CHROMA_HNSW_SPACE_KEY,
    CHROMA_SPACE,
    DEFAULT_TOP_K,
)
from src.schemas.typed_dicts import RetrievalResult


# 这是 ChromaDB 向量检索器的完整实现，负责把用户 query 转成 embedding、查向量库、把原始结果转成项目统一的 RetrievalResult 格式。
class ChromaVectorRetriever(BaseRetriever):
    def __init__(self, persist_directory: Optional[str] = None):
        persist_directory = persist_directory or config.chroma.persist_directory
        self.client = chromadb.PersistentClient(
            path=persist_directory,
        )
        expected_model = config.embedding.model
        self.collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={
                CHROMA_HNSW_SPACE_KEY: CHROMA_SPACE,
                CHROMA_EMBEDDING_MODEL_KEY: expected_model,
            },
        )
        self._verify_embedding_model(expected_model)
        self._model = None

    def _verify_embedding_model(self, expected_model: str) -> None:
        """校验 collection 记录的 embedding 模型与当前配置一致。

        Legacy 数据（无 metadata）补写并告警；不匹配则抛出 RuntimeError。
        """
        import warnings

        metadata = self.collection.metadata or {}
        stored_model = metadata.get(CHROMA_EMBEDDING_MODEL_KEY)
        # 非字符串值（如 None 或 mock 对象）视为未设置
        if not isinstance(stored_model, str):
            warnings.warn(
                f"Chroma collection '{CHROMA_COLLECTION_NAME}' 未记录 embedding_model，"
                f"补写为 '{expected_model}'。若实际入库模型不同，检索结果将不可靠。",
                stacklevel=2,
            )
            try:
                self.collection.modify(
                    metadata={
                        CHROMA_HNSW_SPACE_KEY: CHROMA_SPACE,
                        CHROMA_EMBEDDING_MODEL_KEY: expected_model,
                    }
                )
            except Exception:
                warnings.warn("无法更新 collection metadata。", stacklevel=2)
            return
        if stored_model != expected_model:
            raise RuntimeError(
                f"Embedding 模型不匹配：Chroma collection 记录为 '{stored_model}'，"
                f"当前配置为 '{expected_model}'。入库与检索必须使用同一模型，"
                f"否则向量空间不匹配将导致检索完全失效。请重新入库或修正配置。"
            )

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        filters: Optional[Dict] = None,
    ) -> List[RetrievalResult]:
        query_embedding = self._embed(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filters or None,
        )
        return self._format(results)

    def delete_by_source(self, source: str) -> int:
        """P2-7: 按 source 路径删除文档的所有 chunk，返回删除数量。

        用于文档版本管理：重新入库前先删除旧版本，避免重复 chunk。
        """
        from src.schemas.constants import META_SOURCE

        # 先查匹配的 ID
        existing = self.collection.get(where={META_SOURCE: source}, include=[])
        ids = existing.get("ids", [])
        if ids:
            self.collection.delete(ids=ids)
        return len(ids)

    def _embed(self, text: str) -> List[float]:
        """调用配置的 embedding 模型（懒加载，首次使用时加载，之后复用）"""
        if self._model is None:
            from src.ingestion.embedder import get_embedding_model

            self._model = get_embedding_model(config.embedding.model)
        return self._model.embed_query(text)

    def _format(self, results: QueryResult) -> list[RetrievalResult]:
        formatted: list[RetrievalResult] = []
        documents = results.get("documents")  # 文档内容列表的列表
        metadatas = results.get("metadatas")
        distances = results.get("distances")
        if not documents or not metadatas or not distances:
            return formatted
        for doc, meta, dist in zip(
            documents[0],
            metadatas[0],
            distances[0],
        ):
            formatted.append(
                RetrievalResult(
                    content=doc,
                    metadata=cast(dict, meta or {}),
                    # 距离转相似度：ChromaDB 返回的是 distance（距离越小越相似），项目统一用 score（相似度越大越相关），所以 score = 1 - dist
                    score=1 - dist,  # cosine distance -> similarity
                )
            )
        return formatted


# ──────────────────────────────────────────────
# 调用示例
# ──────────────────────────────────────────────
#
# 示例 1：基本使用，默认配置
#   retriever = ChromaVectorRetriever()
#   results = retriever.retrieve("什么是 RAG？", top_k=3)
#   # 内部流程：
#   #   1. _embed("什么是 RAG？") -> [0.123, -0.456, ...]  (维度由配置模型决定)
#   #   2. collection.query(query_embeddings=[...], n_results=3)
#   #   3. _format() -> [
#   #        RetrievalResult(content="RAG 是...", metadata={"source": "..."}, score=0.89),
#   #        RetrievalResult(content="...", metadata={...}, score=0.76),
#   #        ...
#   #      ]
#
# 示例 2：传入自定义持久化目录（比如测试环境）
#   test_retriever = ChromaVectorRetriever(persist_directory="/tmp/test_chroma")
#
# 示例 3：带过滤条件
#   results = retriever.retrieve(
#       "怎么退款？",
#       filters={"doc_type": "faq", "category": "售后"}
#   )
#   # 内部实际传给 ChromaDB：
#   # where={"$and": [{"doc_type": "faq"}, {"category": "售后"}]}
#
# 示例 4：懒加载行为（首次调用才加载模型）
#   retriever = ChromaVectorRetriever()
#   print(retriever._model)  # None（还没加载）
#   results = retriever.retrieve("hello")  # 首次调用，触发 _embed -> 加载模型
#   print(retriever._model)  # <EmbeddingFunction...>（已缓存）
