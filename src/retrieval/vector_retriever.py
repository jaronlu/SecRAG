"""ChromaDB 向量检索器"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional

import chromadb

if TYPE_CHECKING:
    from chromadb.api.types import QueryResult

# ⚡ 字段统一：使用常量而非裸字符串
from src.config import config
from src.retrieval.base import BaseRetriever
from src.schemas.constants import (
    CHROMA_COLLECTION_NAME,
    CHROMA_DEFAULT_PERSIST_DIR,
    CHROMA_SPACE,
    DEFAULT_TOP_K,
    RR_CONTENT,
    RR_METADATA,
    RR_SCORE,
)


class ChromaVectorRetriever(BaseRetriever):
    def __init__(self, persist_directory: str = CHROMA_DEFAULT_PERSIST_DIR):
        self.client = chromadb.PersistentClient(
            path=persist_directory,
        )
        self.collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": CHROMA_SPACE},
        )
        self._model = None

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        filters: Optional[Dict] = None,
    ) -> List[Dict]:
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

    def _format(self, results: QueryResult) -> List[Dict]:
        formatted = []
        documents = results.get("documents")
        metadatas = results.get("metadatas")
        distances = results.get("distances")
        if not documents or not metadatas or not distances:
            return formatted
        for doc, meta, dist in zip(
            documents[0],
            metadatas[0],
            distances[0],
        ):
            formatted.append({
                RR_CONTENT: doc,
                RR_METADATA: meta,
                RR_SCORE: 1 - dist,  # cosine distance -> similarity
            })
        return formatted
