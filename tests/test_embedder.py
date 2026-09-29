import sys
from types import SimpleNamespace
from typing import Any

from langchain_core.documents import Document

from src.ingestion import embedder
from src.schemas.constants import (
    CHROMA_EMBEDDING_MODEL_KEY,
    CHROMA_HNSW_SPACE_KEY,
    CHROMA_SPACE,
    DEFAULT_EMBEDDING_MODEL,
)


def test_detect_device_prefers_cuda(monkeypatch):
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True),
        backends=SimpleNamespace(
            mps=SimpleNamespace(is_built=lambda: True, is_available=lambda: True)
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    assert embedder._detect_device() == "cuda"


def test_detect_device_uses_mps_only_when_built_and_available(monkeypatch):
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        backends=SimpleNamespace(
            mps=SimpleNamespace(is_built=lambda: True, is_available=lambda: True)
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    assert embedder._detect_device() == "mps"


def test_detect_device_falls_back_to_cpu_when_mps_unavailable(monkeypatch):
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        backends=SimpleNamespace(
            mps=SimpleNamespace(is_built=lambda: True, is_available=lambda: False)
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    assert embedder._detect_device() == "cpu"


def test_get_embedding_model_uses_financial_defaults(monkeypatch):
    captured = {}

    class FakeHuggingFaceEmbeddings:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(embedder, "HuggingFaceEmbeddings", FakeHuggingFaceEmbeddings)
    monkeypatch.setattr(embedder, "_embedding_model_cache", {})

    model = embedder.get_embedding_model("test-model")

    assert isinstance(model, FakeHuggingFaceEmbeddings)
    assert captured == {
        "model_name": "test-model",
        "model_kwargs": {"device": captured["model_kwargs"]["device"], "local_files_only": True},
        "encode_kwargs": {"normalize_embeddings": True},
    }


def test_get_embedding_model_caches_instance_per_model_and_device(monkeypatch):
    """ISSUE-15：同一 (模型名, 设备) 进程内只加载一次权重。

    每跳/每工具调用重建 HuggingFaceEmbeddings 会重新加载
    SentenceTransformer 权重（实测 0.11-0.44s × 每请求 2-6 次）。
    """
    calls: list[tuple[str, str]] = []

    class FakeHuggingFaceEmbeddings:
        def __init__(self, **kwargs):
            calls.append((kwargs["model_name"], kwargs["model_kwargs"]["device"]))

    monkeypatch.setattr(embedder, "HuggingFaceEmbeddings", FakeHuggingFaceEmbeddings)
    monkeypatch.setattr(embedder, "_detect_device", lambda: "cpu")
    monkeypatch.setattr(embedder, "_embedding_model_cache", {})

    first = embedder.get_embedding_model("model-a")
    second = embedder.get_embedding_model("model-a")
    other = embedder.get_embedding_model("model-b")

    assert first is second, "同一模型必须复用缓存实例"
    assert other is not first, "不同模型名不得共用实例"
    assert calls == [("model-a", "cpu"), ("model-b", "cpu")]

    # 设备变化视为不同的缓存键（如测试桩切换 cuda/cpu）
    monkeypatch.setattr(embedder, "_detect_device", lambda: "cuda")
    on_cuda = embedder.get_embedding_model("model-a")
    assert on_cuda is not first
    assert calls == [("model-a", "cpu"), ("model-b", "cpu"), ("model-a", "cuda")]


def test_embed_and_store_uses_provided_embedding_model(monkeypatch, tmp_path):
    chunks = [Document(page_content="测试内容", metadata={"source": "unit-test"})]
    embedding_model: Any = object()
    expected_vectorstore = object()
    captured = {}

    def fake_from_documents(**kwargs):
        captured.update(kwargs)
        return expected_vectorstore

    monkeypatch.setattr(embedder.Chroma, "from_documents", fake_from_documents)

    vectorstore = embedder.embed_and_store(
        chunks=chunks,
        persist_directory=str(tmp_path),
        embedding_model=embedding_model,
    )

    assert vectorstore is expected_vectorstore
    assert captured == {
        "documents": chunks,
        "embedding": embedding_model,
        "persist_directory": str(tmp_path),
        "collection_name": "securities_docs",
        "collection_metadata": {
            CHROMA_HNSW_SPACE_KEY: CHROMA_SPACE,
            CHROMA_EMBEDDING_MODEL_KEY: DEFAULT_EMBEDDING_MODEL,
        },
    }


def test_upsert_chunks_uses_stable_document_ids(monkeypatch, tmp_path):
    chunks = [
        Document(page_content="测试内容", metadata={"doc_id": "doc-1"}, id="chunk-1"),
        Document(page_content="更多内容", metadata={"doc_id": "doc-1"}, id="chunk-2"),
    ]
    embedding_model: Any = object()
    captured: dict[str, Any] = {"batches": []}

    class FakeChroma:
        def __init__(self, **kwargs):
            captured["init"] = kwargs

        def add_documents(self, documents, ids):
            captured["batches"].append((documents, ids))

    monkeypatch.setattr(embedder, "Chroma", FakeChroma)

    embedder.upsert_chunks(chunks, str(tmp_path), embedding_model, batch_size=1)

    assert captured["init"]["persist_directory"] == str(tmp_path)
    assert captured["batches"] == [
        ([chunks[0]], ["chunk-1"]),
        ([chunks[1]], ["chunk-2"]),
    ]


def test_upsert_chunks_rejects_invalid_batch_size(tmp_path):
    chunks = [Document(page_content="测试内容", metadata={}, id="chunk-1")]
    embedding_model: Any = object()

    try:
        embedder.upsert_chunks(chunks, str(tmp_path), embedding_model, batch_size=0)
    except ValueError as exc:
        assert "batch_size" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_list_chunk_ids_by_doc_id_filters_on_doc_id(monkeypatch, tmp_path):
    embedding_model: Any = object()
    captured = {}

    class FakeChroma:
        def __init__(self, **kwargs):
            captured["init"] = kwargs

        def get(self, **kwargs):
            captured["get"] = kwargs
            return {"ids": ["old-1", "old-2"]}

    monkeypatch.setattr(embedder, "Chroma", FakeChroma)

    ids = embedder.list_chunk_ids_by_doc_id("doc-1", str(tmp_path), embedding_model)

    assert ids == ["old-1", "old-2"]
    assert captured["get"] == {"where": {"doc_id": "doc-1"}, "include": []}


def test_delete_chunk_ids_deletes_sorted_ids(monkeypatch, tmp_path):
    embedding_model: Any = object()
    captured = {}

    class FakeChroma:
        def __init__(self, **kwargs):
            captured["init"] = kwargs

        def delete(self, ids):
            captured["ids"] = ids

    monkeypatch.setattr(embedder, "Chroma", FakeChroma)

    embedder.delete_chunk_ids({"old-2", "old-1"}, str(tmp_path), embedding_model)

    assert captured["ids"] == ["old-1", "old-2"]
