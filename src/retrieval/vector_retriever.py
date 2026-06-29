from typing import Dict, List, Optional

import chromadb

from .base import BaseRetriever


class FinancialVectorRetriever(BaseRetriever):
    def __init__(self):
        self.client = chromadb.PersistentClient(
            path="./data/chroma",
        )
        self.collection = self.client.get_or_create_collection(
            name="securities_docs",
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

        model = get_embedding_model("BAAI/bge-m3")

        # TODO:：每次检索都重新加载模型；Phase 2 改为应用启动时单例
        return model.embed_query(text)

    def _format(self, results: Dict) -> List[Dict]:
        formatted = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            formatted.append({
                "content": doc,
                "metadata": meta,
                "score": 1 - dist,  # cosine distance -> similarity
            })
        return formatted
