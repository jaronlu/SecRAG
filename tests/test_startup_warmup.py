"""启动预热测试（ISSUE-16）：冷启动开销不落在首个请求上。

首请求此前同步承担 torch import、BM25 全索引构建（jieba 分词 3 万+
chunks，10-30s）与 embedding 模型加载；lifespan 启动时在后台线程预热。
"""

from __future__ import annotations

import threading

from fastapi.testclient import TestClient

from src.api import main as api_main
from src.api.main import app


def _patch_warmup_dependencies(monkeypatch, events: list):
    monkeypatch.setattr(api_main, "_get_agent_app", lambda: events.append("graph"))

    class _FakeVectorEngine:
        def __init__(self):
            events.append("vector")

    class _FakeBM25Retriever:
        def __init__(self, engine):
            events.append(("bm25", engine))

        def warmup(self):
            events.append("bm25_warm")

    monkeypatch.setattr(
        "src.retrieval.vector_retriever.ChromaVectorRetriever", _FakeVectorEngine
    )
    monkeypatch.setattr("src.retrieval.bm25_retriever.BM25Retriever", _FakeBM25Retriever)
    monkeypatch.setattr(
        "src.ingestion.embedder.get_embedding_model",
        lambda name: events.append(("embed", name)),
    )
    monkeypatch.setattr("src.tools.rerank.reranker_available", lambda: False)


def test_warmup_covers_graph_vector_bm25_and_embedding(monkeypatch):
    events: list = []
    _patch_warmup_dependencies(monkeypatch, events)

    api_main._warmup_retrieval_stack()

    assert "graph" in events
    assert "vector" in events
    assert "bm25_warm" in events
    assert ("embed", api_main.config.embedding.model) in events


def test_warmup_step_failure_does_not_block_remaining_steps(monkeypatch):
    """单步失败只记日志：预热不因图构建失败而跳过向量/BM25 预热。"""
    events: list = []
    _patch_warmup_dependencies(monkeypatch, events)

    def _boom():
        raise RuntimeError("graph build failed")

    monkeypatch.setattr(api_main, "_get_agent_app", _boom)

    api_main._warmup_retrieval_stack()

    assert "vector" in events
    assert "bm25_warm" in events


def test_warmup_skips_reranker_when_unavailable(monkeypatch):
    """reranker 未配置（ISSUE-10）时预热不得触碰 rerank 服务。"""
    events: list = []
    _patch_warmup_dependencies(monkeypatch, events)

    def _forbidden():
        raise AssertionError("reranker 未配置时不得预热 rerank")

    monkeypatch.setattr("src.tools.rerank.RerankService", _forbidden)

    api_main._warmup_retrieval_stack()

    assert "bm25_warm" in events


def test_lifespan_runs_warmup_in_background(monkeypatch):
    """lifespan 启动即触发后台预热，不阻塞服务就绪。"""
    warmed = threading.Event()
    monkeypatch.setattr(api_main, "_warmup_retrieval_stack", lambda: warmed.set())

    with TestClient(app):
        assert warmed.wait(timeout=10), "lifespan 启动应触发后台预热"
