"""ChromaDB 向量检索器"""

from typing import Any, Dict, List, Optional

import chromadb

# ⚡ 字段统一：使用常量而非裸字符串
from src.config import config
from src.schemas.constants import (
    CHROMA_COLLECTION_NAME,
    CHROMA_DEFAULT_PERSIST_DIR,
    RR_CONTENT,
    RR_METADATA,
    RR_SCORE,
)

from .base import BaseRetriever


class FinancialVectorRetriever(BaseRetriever):
    def __init__(self, persist_directory: str = CHROMA_DEFAULT_PERSIST_DIR):
        self.client = chromadb.PersistentClient(
            path=persist_directory,
        )
        self.collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
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
        """调用配置的 embedding 模型"""
        from src.ingestion.embedder import get_embedding_model

        model = get_embedding_model(config.embedding.model)
        # TODO: 每次检索都重新加载模型；Phase 2 改为应用启动时单例
        return model.embed_query(text)

    def _format(self, results: Any) -> List[Dict]:
        formatted = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            formatted.append({
                RR_CONTENT: doc,
                RR_METADATA: meta,
                RR_SCORE: 1 - dist,  # cosine distance -> similarity
            })
        return formatted
