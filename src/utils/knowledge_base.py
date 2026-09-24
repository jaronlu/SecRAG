"""P2: 知识库管理——文档列表、详情、统计、删除。

封装 ChromaDB 和现有入库流水线，提供运行时的知识库管理能力。
用于管理界面和 API 端点。

功能：
- list_documents: 列出所有文档（按 source 分组，含 chunk 数、类型、标题）
- get_document_chunks: 查看某文档的所有 chunk
- get_stats: 知识库统计（文档数、chunk 数、按类型分布）
- delete_document: 删除文档（复用 vector_retriever.delete_by_source）
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from src.config import config
from src.schemas.constants import (
    META_DOC_ID,
    META_DOC_TYPE,
    META_SOURCE,
    META_TITLE,
)


class KnowledgeBaseManager:
    """知识库管理器——封装 ChromaDB 操作，提供文档级管理能力。"""

    def __init__(self, persist_directory: str | None = None):
        self.persist_directory = persist_directory or config.chroma_persist_directory
        self._collection = None

    def _get_collection(self):
        """懒加载 ChromaDB collection。"""
        if self._collection is None:
            import chromadb

            client = chromadb.PersistentClient(path=self.persist_directory)
            self._collection = client.get_or_create_collection(
                name="secrag_documents",
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def list_documents(
        self,
        doc_type: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        """列出知识库中的所有文档（按 source 分组）。

        Args:
            doc_type: 按文档类型筛选（可选）
            limit: 返回数量限制
            offset: 分页偏移

        Returns:
            {total, documents: [{source, doc_id, doc_type, title, chunk_count, first_indexed}]}
        """
        collection = self._get_collection()
        total = collection.count()

        if total == 0:
            return {"total": 0, "documents": []}

        # 分批获取所有 chunk 的元数据
        batch_size = 1000
        all_metadatas = []
        for i in range(0, total, batch_size):
            batch = collection.get(
                limit=batch_size,
                offset=i,
                include=["metadatas"],
            )
            all_metadatas.extend(batch.get("metadatas", []) or [])

        # 按 source 分组
        docs: dict[str, dict[str, Any]] = {}
        for meta in all_metadatas:
            if not meta:
                continue
            source = meta.get(META_SOURCE, "unknown")
            if doc_type and meta.get(META_DOC_TYPE) != doc_type:
                continue

            if source not in docs:
                docs[source] = {
                    "source": source,
                    "doc_id": meta.get(META_DOC_ID, ""),
                    "doc_type": meta.get(META_DOC_TYPE, "unknown"),
                    "title": meta.get(META_TITLE, source.split("/")[-1]),
                    "chunk_count": 0,
                }
            docs[source]["chunk_count"] += 1

        doc_list = sorted(docs.values(), key=lambda x: x["source"])
        total_docs = len(doc_list)
        paginated = doc_list[offset : offset + limit]

        return {"total": total_docs, "documents": paginated}

    def get_document_chunks(
        self,
        source: str,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """获取某文档的所有 chunk。

        Args:
            source: 文档 source 路径
            limit: 返回数量限制
            offset: 分页偏移

        Returns:
            {source, total, chunks: [{chunk_id, chunk_index, content, metadata}]}
        """
        collection = self._get_collection()

        result = collection.get(
            where={META_SOURCE: source},
            limit=limit,
            offset=offset,
            include=["documents", "metadatas"],
        )

        ids = result.get("ids", []) or []
        documents = result.get("documents", []) or []
        metadatas = result.get("metadatas", []) or []

        # 获取总数（不带 limit/offset）
        total_result = collection.get(where={META_SOURCE: source}, include=[])
        total = len(total_result.get("ids", []) or [])

        chunks = []
        for i, (cid, doc, meta) in enumerate(zip(ids, documents, metadatas)):
            chunks.append({
                "chunk_id": cid,
                "chunk_index": (meta or {}).get("chunk_index", offset + i),
                "content": doc,
                "metadata": meta or {},
            })

        # 按 chunk_index 排序
        chunks.sort(key=lambda x: x.get("chunk_index", 0))

        return {
            "source": source,
            "total": total,
            "chunks": chunks,
        }

    def get_stats(self) -> dict[str, Any]:
        """获取知识库统计信息。

        Returns:
            {total_chunks, total_documents, by_doc_type: {type: count},
             by_doc_type_chunks: {type: chunk_count}}
        """
        collection = self._get_collection()
        total_chunks = collection.count()

        if total_chunks == 0:
            return {
                "total_chunks": 0,
                "total_documents": 0,
                "by_doc_type": {},
                "by_doc_type_chunks": {},
            }

        # 获取所有元数据
        batch_size = 1000
        all_metadatas = []
        for i in range(0, total_chunks, batch_size):
            batch = collection.get(
                limit=batch_size,
                offset=i,
                include=["metadatas"],
            )
            all_metadatas.extend(batch.get("metadatas", []) or [])

        # 统计
        sources = set()
        by_type_docs: dict[str, set] = defaultdict(set)
        by_type_chunks: dict[str, int] = defaultdict(int)

        for meta in all_metadatas:
            if not meta:
                continue
            source = meta.get(META_SOURCE, "unknown")
            doc_type = meta.get(META_DOC_TYPE, "unknown")
            sources.add(source)
            by_type_docs[doc_type].add(source)
            by_type_chunks[doc_type] += 1

        return {
            "total_chunks": total_chunks,
            "total_documents": len(sources),
            "by_doc_type": {k: len(v) for k, v in by_type_docs.items()},
            "by_doc_type_chunks": dict(by_type_chunks),
        }

    def delete_document(self, source: str) -> int:
        """删除某文档的所有 chunk。

        Args:
            source: 文档 source 路径

        Returns:
            删除的 chunk 数量
        """
        from src.retrieval.vector_retriever import ChromaVectorRetriever

        engine = ChromaVectorRetriever()
        return engine.delete_by_source(source)

    def search_documents(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """在知识库中语义搜索（用于管理界面预览检索效果）。

        Args:
            query: 搜索查询
            top_k: 返回结果数

        Returns:
            [{content, metadata, score}]
        """
        from src.retrieval.vector_retriever import ChromaVectorRetriever

        engine = ChromaVectorRetriever()
        results = engine.retrieve(query=query, top_k=top_k)
        return [
            {
                "content": r["content"],
                "metadata": r["metadata"],
                "score": r["score"],
            }
            for r in results
        ]


# 全局单例
_kb_manager: KnowledgeBaseManager | None = None


def get_kb_manager() -> KnowledgeBaseManager:
    """获取知识库管理器单例。"""
    global _kb_manager
    if _kb_manager is None:
        _kb_manager = KnowledgeBaseManager()
    return _kb_manager
