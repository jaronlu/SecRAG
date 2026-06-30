from langchain_core.documents import Document

from src.ingestion import embedder


def test_get_embedding_model_uses_financial_defaults(monkeypatch):
    captured = {}

    class FakeHuggingFaceEmbeddings:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(embedder, "HuggingFaceEmbeddings", FakeHuggingFaceEmbeddings)

    model = embedder.get_embedding_model("test-model")

    assert isinstance(model, FakeHuggingFaceEmbeddings)
    assert captured == {
        "model_name": "test-model",
        "model_kwargs": {"device": "mps"},
        "encode_kwargs": {"normalize_embeddings": True},
    }


def test_embed_and_store_uses_provided_embedding_model(monkeypatch, tmp_path):
    chunks = [Document(page_content="测试内容", metadata={"source": "unit-test"})]
    embedding_model = object()
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
        "collection_metadata": {"hnsw:space": "cosine"},
    }
