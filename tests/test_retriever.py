"""ChromaVectorRetriever 单元测试（mock ChromaDB + embedding）"""

from typing import Dict, List
from unittest.mock import MagicMock

import pytest

from src.schemas.constants import (
    DOC_TYPE_FINANCIAL_DATA,
    META_DATE,
    META_DOC_TYPE,
    META_TITLE,
    RR_CONTENT,
    RR_METADATA,
    RR_SCORE,
)


@pytest.fixture
def mock_chromadb(monkeypatch) -> MagicMock:
    """mock ChromaDB PersistentClient 和 collection"""
    mock_collection = MagicMock()
    mock_client = MagicMock()
    mock_client.get_or_create_collection.return_value = mock_collection
    monkeypatch.setattr("chromadb.PersistentClient", lambda **_: mock_client)
    return mock_collection


@pytest.fixture
def mock_embedding(monkeypatch) -> None:
    """mock _embed 避免真实加载模型"""

    def fake_embed(self, text: str) -> List[float]:
        return [0.1] * 768  # bge-m3 输出 768 维

    monkeypatch.setattr(
        "src.retrieval.vector_retriever.ChromaVectorRetriever._embed", fake_embed
    )


@pytest.fixture
def retriever(mock_chromadb, mock_embedding):
    """返回 ChromaVectorRetriever 实例，依赖均已 mock"""
    from src.retrieval.vector_retriever import ChromaVectorRetriever

    return ChromaVectorRetriever()


# ── _format ──────────────────────────────────────────────────────────────


class TestFormat:
    def test_empty_results(self, retriever):
        results = {"documents": [[]], "metadatas": [[]], "distances": [[]]}
        formatted = retriever._format(results)
        assert formatted == []

    def test_single_result(self, retriever):
        results = {
            "documents": [["茅台净利润747亿"]],
            "metadatas": [[{"title": "年报", "date": "2024"}]],
            "distances": [[0.15]],
        }
        formatted = retriever._format(results)
        assert len(formatted) == 1
        assert formatted[0][RR_CONTENT] == "茅台净利润747亿"
        assert formatted[0][RR_METADATA] == {"title": "年报", "date": "2024"}
        assert formatted[0][RR_SCORE] == pytest.approx(0.85)

    def test_multiple_results(self, retriever):
        results = {
            "documents": [["doc1", "doc2", "doc3"]],
            "metadatas": [[{"a": 1}, {"a": 2}, {"a": 3}]],
            "distances": [[0.1, 0.2, 0.3]],
        }
        formatted = retriever._format(results)
        assert len(formatted) == 3
        for i, doc in enumerate(formatted):
            assert doc[RR_CONTENT] == f"doc{i + 1}"
            assert doc[RR_METADATA]["a"] == i + 1
            assert doc[RR_SCORE] == pytest.approx(1 - (i + 1) * 0.1)

    def test_score_upper_bound(self, retriever):
        """distance=0 → score=1（完全匹配）"""
        results = {
            "documents": [["完美匹配"]],
            "metadatas": [[{"title": "test"}]],
            "distances": [[0.0]],
        }
        formatted = retriever._format(results)
        assert formatted[0][RR_SCORE] == pytest.approx(1.0)

    def test_score_lower_bound(self, retriever):
        """distance=1 → score=0（完全不匹配）
        ChromaDB cosine distance range 是 [0, 2]，但这里只测极端值
        """
        results = {
            "documents": [["不相关"]],
            "metadatas": [[{"title": "test"}]],
            "distances": [[1.0]],
        }
        formatted = retriever._format(results)
        assert formatted[0][RR_SCORE] == pytest.approx(0.0)


# ── _embed ───────────────────────────────────────────────────────────────


class TestEmbed:
    def test_embed_returns_float_list(self, monkeypatch):
        """真实调用 _embed，mock 底层模型避免下载"""
        import chromadb

        # mock chromadb 连接
        mock_collection = MagicMock()
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        monkeypatch.setattr("chromadb.PersistentClient", lambda **_: mock_client)

        # mock embedding 模型
        from src.ingestion import embedder as embedder_module

        class FakeModel:
            def embed_query(self, text: str) -> List[float]:
                return [0.5] * 768

            def __init__(self, **kwargs): ...

        monkeypatch.setattr(embedder_module, "HuggingFaceEmbeddings", FakeModel)

        from src.retrieval.vector_retriever import ChromaVectorRetriever

        retriever = ChromaVectorRetriever()
        embedding = retriever._embed("测试查询")
        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(v, float) for v in embedding)


# ── retrieve ─────────────────────────────────────────────────────────────


class TestRetrieve:
    def test_basic_query(self, retriever, mock_chromadb):
        """retrieve 调用 _embed 和 ChromaDB query，返回格式化结果"""
        # 配置 mock collection.query 的返回值
        mock_chromadb.query.return_value = {
            "documents": [["茅台净利润747亿", "五粮液净利润302亿"]],
            "metadatas": [[{"title": "年报"}, {"title": "年报"}]],
            "distances": [[0.12, 0.25]],
        }

        results = retriever.retrieve("茅台2024净利润")

        assert len(results) == 2
        assert results[0][RR_CONTENT] == "茅台净利润747亿"
        assert results[0][RR_SCORE] == pytest.approx(0.88)

        # 验证 ChromaDB query 被正确调用
        mock_chromadb.query.assert_called_once()
        call_kwargs = mock_chromadb.query.call_args[1]
        assert call_kwargs["n_results"] == 5
        assert call_kwargs["where"] is None

    def test_top_k(self, retriever, mock_chromadb):
        """top_k 参数传递给 ChromaDB"""
        mock_chromadb.query.return_value = {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

        retriever.retrieve("查询", top_k=3)

        assert mock_chromadb.query.call_args[1]["n_results"] == 3

    def test_with_filters(self, retriever, mock_chromadb):
        """filters 以 where 参数传递给 ChromaDB"""
        mock_chromadb.query.return_value = {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

        filters = {META_DOC_TYPE: {"$eq": DOC_TYPE_FINANCIAL_DATA}}
        retriever.retrieve("查询", filters=filters)

        assert mock_chromadb.query.call_args[1]["where"] == filters

    def test_empty_filters_passed_as_none(self, retriever, mock_chromadb):
        """filters 为空 dict 时，where 传 None（ChromaDB 不会报错）"""
        mock_chromadb.query.return_value = {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

        retriever.retrieve("查询", filters={})

        assert mock_chromadb.query.call_args[1]["where"] is None

    def test_no_results(self, retriever, mock_chromadb):
        """检索无结果时返回空列表"""
        mock_chromadb.query.return_value = {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

        results = retriever.retrieve("不存在的内容")
        assert results == []
