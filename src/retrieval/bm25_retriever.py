"""BM25 关键词检索器：对 ChromaDB 中文档建索引，支持精确术语匹配。

与向量检索互补：向量检索擅长语义相似，BM25 擅长精确术语（法规条款号、股票代码、产品名）。
结果通过 RRF（Reciprocal Rank Fusion）与向量结果融合。
"""

from __future__ import annotations

from typing import Optional

from rank_bm25 import BM25Okapi

from src.retrieval.vector_retriever import ChromaVectorRetriever
from src.schemas.constants import (
    META_BM25_SCORE,
    META_CHUNK_ID,
    META_RRF_SCORE,
    META_SOURCE,
    META_VECTOR_SCORE,
    RR_CONTENT,
    RR_METADATA,
    RR_SCORE,
    RRF_K,
)
from src.schemas.typed_dicts import RetrievalResult

# 模块级 BM25 索引缓存：(persist_directory) -> (bm25, docs, metadatas, ids)
_bm25_cache: dict[str, tuple[BM25Okapi, list[str], list[dict], list[str]]] = {}


def invalidate_bm25_cache(persist_directory: str | None = None) -> int:
    """使 BM25 索引缓存失效；入库发布新版本后必须调用（issues.md 二.1）。

    Args:
        persist_directory: 只失效指定索引；None 表示全部失效。

    Returns:
        失效的条目数。
    """
    if persist_directory is None:
        count = len(_bm25_cache)
        _bm25_cache.clear()
        return count
    removed = 0
    for key in [key for key in _bm25_cache if key == persist_directory]:
        del _bm25_cache[key]
        removed += 1
    return removed

# 中文分词：jieba 延迟导入，避免启动开销
_tokenizer = None


def _get_tokenizer():
    """懒加载 jieba 分词器。"""
    global _tokenizer
    if _tokenizer is None:
        import jieba

        jieba.setLogLevel(60)  # 静音
        _tokenizer = jieba
    return _tokenizer


def _tokenize(text: str) -> list[str]:
    """中文分词：jieba 精确模式 + 英文/数字保留。"""
    if not text:
        return []
    jieba = _get_tokenizer()
    tokens = list(jieba.cut(text, cut_all=False))
    # 过滤空白和单字符（保留英文和数字）
    return [t.strip().lower() for t in tokens if t.strip() and len(t.strip()) > 0]


def _build_bm25_index(
    vector_engine: ChromaVectorRetriever,
) -> tuple[BM25Okapi, list[str], list[dict], list[str]]:
    """从 ChromaDB 加载所有文档并构建 BM25 索引。

    返回 (bm25, contents, metadatas, ids)。
    """
    collection = vector_engine.collection
    # 分批加载所有文档
    all_docs: list[str] = []
    all_metas: list[dict] = []
    all_ids: list[str] = []
    offset = 0
    batch_size = 5000
    while True:
        batch = collection.get(
            limit=batch_size,
            offset=offset,
            include=["documents", "metadatas"],
        )
        ids = batch.get("ids", []) or []
        docs = batch.get("documents", []) or []
        metas = batch.get("metadatas", []) or []
        if not ids:
            break
        all_ids.extend(ids)
        all_docs.extend(str(d) for d in docs)
        all_metas.extend(dict(m or {}) for m in metas)
        offset += len(ids)
        if len(ids) < batch_size:
            break

    # 构建 BM25 索引
    tokenized = [_tokenize(doc) for doc in all_docs]
    bm25 = BM25Okapi(tokenized)
    return bm25, all_docs, all_metas, all_ids


class BM25Retriever:
    """BM25 关键词检索器，与 ChromaVectorRetriever 共享数据。"""

    def __init__(self, vector_engine: ChromaVectorRetriever):
        self.vector_engine = vector_engine
        self._index: Optional[tuple[BM25Okapi, list[str], list[dict], list[str]]] = None

    def _get_index(self) -> tuple[BM25Okapi, list[str], list[dict], list[str]]:
        """懒加载 BM25 索引，带模块级缓存。

        缓存 key 用 vector_engine 的 persist_directory（稳定的索引身份），
        而不是 id(engine)——同一进程内每次检索会新建 engine 实例，
        id 作 key 既不稳定也不表达知识库版本（issues.md 二.1）。
        """
        if self._index is not None:
            return self._index
        cache_key = str(getattr(self.vector_engine, "persist_directory", ""))
        if cache_key in _bm25_cache:
            self._index = _bm25_cache[cache_key]
            return self._index
        self._index = _build_bm25_index(self.vector_engine)
        _bm25_cache[cache_key] = self._index
        return self._index

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[dict] = None,
    ) -> list[RetrievalResult]:
        """BM25 关键词检索，返回 RetrievalResult 列表。

        filters 目前仅支持精确匹配（ChromaDB where 子句），用于后过滤。
        """
        bm25, contents, metadatas, ids = self._get_index()
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        scores = bm25.get_scores(query_tokens)
        # 按分数降序排列
        ranked = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )

        results: list[RetrievalResult] = []
        for idx in ranked:
            if scores[idx] <= 0:
                break  # BM25 分数为 0 表示无匹配
            meta = metadatas[idx]
            # 简单后过滤：支持精确匹配的 filters
            if filters and not self._match_filters(meta, filters):
                continue
            results.append(
                RetrievalResult(
                    content=contents[idx],
                    metadata=meta,
                    score=float(scores[idx]),
                )
            )
            if len(results) >= top_k:
                break
        return results

    @staticmethod
    def _match_filters(metadata: dict, filters: dict) -> bool:
        """简单后过滤：支持精确匹配、$gte/$lte 范围和 $and 组合。

        与 Chroma where 契约对齐（issues.md 一.6）：时间过滤形如
        {"$and": [{"date_day": {"$gte": 20240101}}, {"date_day": {"$lte": 20241231}}]}。
        """
        for key, condition in filters.items():
            if key == "$and":
                if not isinstance(condition, list):
                    return False
                if not all(BM25Retriever._match_filters(metadata, item) for item in condition):
                    return False
                continue
            if key == "$or":
                if not isinstance(condition, list):
                    return False
                if not any(BM25Retriever._match_filters(metadata, item) for item in condition):
                    return False
                continue
            value = metadata.get(key)
            if isinstance(condition, dict):
                if "$gte" in condition and (value is None or value < condition["$gte"]):
                    return False
                if "$lte" in condition and (value is None or value > condition["$lte"]):
                    return False
            else:
                if value != condition:
                    return False
        return True


def rrf_fuse(
    vector_results: list[RetrievalResult],
    bm25_results: list[RetrievalResult],
    k: int = RRF_K,
    top_k: int = 10,
) -> list[RetrievalResult]:
    """Reciprocal Rank Fusion：融合向量检索和 BM25 检索结果。

    score = sum(1 / (k + rank)) for each result in both lists.
    以 chunk_id 或 (source, content) 作为去重 key。

    issues.md 一.5：RRF 分数写入 metadata.rrf_score，原始量纲分别保留在
    metadata.vector_score / metadata.bm25_score，返回顺序即融合排序；
    下游不得再用原始 score 重排或阈值过滤融合结果。
    """
    if not bm25_results:
        return vector_results[:top_k]
    if not vector_results:
        return bm25_results[:top_k]

    def _result_key(result: RetrievalResult) -> str:
        meta = result.get(RR_METADATA, {})
        chunk_id = meta.get(META_CHUNK_ID, "")
        if chunk_id:
            return str(chunk_id)
        return f"{meta.get(META_SOURCE, '')}:{result.get(RR_CONTENT, '')[:50]}"

    fused: dict[str, tuple[float, RetrievalResult]] = {}

    for rank, result in enumerate(vector_results):
        key = _result_key(result)
        score = 1.0 / (k + rank + 1)
        result.setdefault(RR_METADATA, {})[META_VECTOR_SCORE] = result.get(RR_SCORE, 0.0)
        if key in fused:
            existing_score, existing = fused[key]
            # 保留向量检索的 metadata（更完整），累加分数
            fused[key] = (existing_score + score, existing)
        else:
            fused[key] = (score, result)

    for rank, result in enumerate(bm25_results):
        key = _result_key(result)
        score = 1.0 / (k + rank + 1)
        result.setdefault(RR_METADATA, {})[META_BM25_SCORE] = result.get(RR_SCORE, 0.0)
        if key in fused:
            existing_score, existing = fused[key]
            fused[key] = (existing_score + score, existing)
        else:
            fused[key] = (score, result)

    # 按融合分数降序，并把融合分写进 metadata 供下游排序使用
    ranked = sorted(fused.values(), key=lambda x: x[0], reverse=True)
    results: list[RetrievalResult] = []
    for fused_score, result in ranked[:top_k]:
        result.setdefault(RR_METADATA, {})[META_RRF_SCORE] = round(fused_score, 6)
        results.append(result)
    return results
