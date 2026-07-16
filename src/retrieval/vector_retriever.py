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
        self.collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={CHROMA_HNSW_SPACE_KEY: CHROMA_SPACE},
        )
        self._model = None

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
