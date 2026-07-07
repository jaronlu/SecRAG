from typing import Any

from langchain_core.documents import Document

from scripts import inspect_chunks, preview_chunks
from src.schemas.constants import (
    DOC_TYPE_FINANCIAL_DATA,
    META_CHUNK_HASH,
    META_CHUNK_ID,
    META_CHUNK_INDEX,
    META_DATE,
    META_DOC_ID,
    META_DOC_TYPE,
    META_SOURCE,
    META_STOCK_CODE,
    META_TITLE,
)


def test_build_chunk_views_and_markdown_preview():
    chunks = [
        Document(
            page_content="第一段内容。" * 20,
            metadata={
                META_DOC_ID: "doc-1",
                META_CHUNK_ID: "chunk-1",
                META_CHUNK_INDEX: 0,
                META_CHUNK_HASH: "hash-1",
                META_DOC_TYPE: DOC_TYPE_FINANCIAL_DATA,
                META_SOURCE: "source.csv",
                META_TITLE: "标题",
                META_STOCK_CODE: "600519",
                META_DATE: "2026",
            },
        )
    ]

    rows = preview_chunks.build_chunk_views(chunks, preview_chars=12)
    markdown = preview_chunks.render_markdown(rows, title="Chunk Preview")

    assert rows[0]["doc_id"] == "doc-1"
    assert rows[0]["content_length"] == len(chunks[0].page_content)
    assert rows[0]["content_preview"].endswith("...")
    assert "total_chunks: 1" in markdown
    assert "- chunk_id: `chunk-1`" in markdown


def test_render_jsonl_outputs_one_row_per_chunk():
    rows: list[preview_chunks.ChunkView] = [
        {
            "doc_id": "doc-1",
            "chunk_id": "chunk-1",
            "chunk_index": 0,
            "chunk_hash": "hash-1",
            "doc_type": DOC_TYPE_FINANCIAL_DATA,
            "source": "source.csv",
            "title": "标题",
            "stock_code": "600519",
            "date": "2026",
            "page_number": "",
            "content_length": 10,
            "content_preview": "内容",
        }
    ]

    output = preview_chunks.render_jsonl(rows)

    assert output.count("\n") == 1
    assert '"doc_id": "doc-1"' in output


def test_preview_file_uses_ingest_normalization(monkeypatch, tmp_path):
    file_path = tmp_path / "stable.csv"
    file_path.write_text("code,year\n600519,2026\n", encoding="utf-8")
    metadata = {
        META_DOC_ID: "dataset:manual:stable-id",
        META_DOC_TYPE: DOC_TYPE_FINANCIAL_DATA,
        META_TITLE: "stable",
    }

    monkeypatch.setattr(preview_chunks, "_load_sample_metadata", lambda _: metadata)
    monkeypatch.setattr(
        preview_chunks,
        "_load_documents",
        lambda _: [Document(page_content="code: 600519\nyear: 2026", metadata={})],
    )

    rows = preview_chunks.preview_file(file_path, DOC_TYPE_FINANCIAL_DATA, limit=1)

    assert rows[0]["doc_id"] == "dataset:manual:stable-id"
    assert rows[0]["chunk_index"] == 0
    assert rows[0]["chunk_hash"]


def test_inspect_doc_id_reads_chroma_without_embedding(monkeypatch):
    captured: dict[str, Any] = {}

    class FakeChroma:
        def __init__(self, **kwargs):
            captured["init"] = kwargs

        def get(self, **kwargs):
            captured["get"] = kwargs
            return {
                "ids": ["chunk-1"],
                "documents": ["stored content"],
                "metadatas": [
                    {
                        META_DOC_ID: "doc-1",
                        META_CHUNK_ID: "chunk-1",
                        META_CHUNK_INDEX: 0,
                        META_CHUNK_HASH: "hash-1",
                        META_DOC_TYPE: DOC_TYPE_FINANCIAL_DATA,
                        META_SOURCE: "source.csv",
                    }
                ],
            }

    monkeypatch.setattr(inspect_chunks, "Chroma", FakeChroma)

    rows = inspect_chunks.inspect_doc_id("doc-1", persist_directory="/tmp/chroma", limit=10)

    assert captured["init"]["embedding_function"] is None
    assert captured["init"]["create_collection_if_not_exists"] is False
    assert captured["get"] == {
        "where": {META_DOC_ID: "doc-1"},
        "limit": 10,
        "include": ["documents", "metadatas"],
    }
    assert rows[0]["chunk_id"] == "chunk-1"
    assert rows[0]["content_preview"] == "stored content"
