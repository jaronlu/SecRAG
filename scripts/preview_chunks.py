"""Preview parsed chunks before writing them into ChromaDB."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Literal, NotRequired, TypedDict

from langchain_core.documents import Document

from scripts.ingest import (
    CHUNKER_VERSION,
    PARSER_VERSION,
    _hash_metadata,
    _load_documents,
    _load_sample_metadata,
    _normalize_parsed_text,
    derive_doc_id,
    normalize_chunks,
    sha256_file,
    sha256_text,
)
from src.config import config
from src.ingestion.chunkers import chunk_documents
from src.schemas.constants import (
    ALL_VALID_DOC_TYPES,
    META_CHUNK_HASH,
    META_CHUNK_ID,
    META_CHUNK_INDEX,
    META_DATE,
    META_DOC_ID,
    META_DOC_TYPE,
    META_PAGE_NUMBER,
    META_SOURCE,
    META_STOCK_CODE,
    META_TITLE,
)

OutputFormat = Literal["markdown", "jsonl"]


class ChunkView(TypedDict):
    doc_id: str
    chunk_id: str
    chunk_index: int
    chunk_hash: str
    doc_type: str
    source: str
    title: str
    stock_code: str
    date: str
    page_number: str
    content_length: int
    content_preview: str
    content: NotRequired[str]


def _metadata_text(metadata: dict[str, Any], key: str) -> str:
    value = metadata.get(key, "")
    if value is None:
        return ""
    return str(value)


def _metadata_int(metadata: dict[str, Any], key: str) -> int:
    value = metadata.get(key, 0)
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return 0


def _preview_text(content: str, max_chars: int) -> str:
    normalized = " ".join(content.split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[:max_chars].rstrip() + "..."


def build_chunk_views(
    chunks: list[Document],
    *,
    full_content: bool = False,
    preview_chars: int = 500,
) -> list[ChunkView]:
    rows: list[ChunkView] = []
    for chunk in chunks:
        metadata = chunk.metadata
        row: ChunkView = {
            "doc_id": _metadata_text(metadata, META_DOC_ID),
            "chunk_id": _metadata_text(metadata, META_CHUNK_ID),
            "chunk_index": _metadata_int(metadata, META_CHUNK_INDEX),
            "chunk_hash": _metadata_text(metadata, META_CHUNK_HASH),
            "doc_type": _metadata_text(metadata, META_DOC_TYPE),
            "source": _metadata_text(metadata, META_SOURCE),
            "title": _metadata_text(metadata, META_TITLE),
            "stock_code": _metadata_text(metadata, META_STOCK_CODE),
            "date": _metadata_text(metadata, META_DATE),
            "page_number": _metadata_text(metadata, META_PAGE_NUMBER),
            "content_length": len(chunk.page_content),
            "content_preview": _preview_text(chunk.page_content, preview_chars),
        }
        if full_content:
            row["content"] = chunk.page_content
        rows.append(row)
    return sorted(rows, key=lambda item: item["chunk_index"])


def render_markdown(rows: list[ChunkView], *, title: str) -> str:
    lines = [f"# {title}", "", f"total_chunks: {len(rows)}", ""]
    for row in rows:
        lines.extend([
            f"## Chunk {row['chunk_index']}",
            "",
            f"- doc_id: `{row['doc_id']}`",
            f"- chunk_id: `{row['chunk_id']}`",
            f"- chunk_hash: `{row['chunk_hash']}`",
            f"- doc_type: `{row['doc_type']}`",
            f"- source: `{row['source']}`",
            f"- title: {row['title']}",
            f"- stock_code: `{row['stock_code']}`",
            f"- date: `{row['date']}`",
            f"- page_number: `{row['page_number']}`",
            f"- content_length: {row['content_length']}",
            "",
            "```text",
            row.get("content", row["content_preview"]),
            "```",
            "",
        ])
    return "\n".join(lines).rstrip() + "\n"


def render_jsonl(rows: list[ChunkView]) -> str:
    return "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else "")


def render_rows(rows: list[ChunkView], *, title: str, output_format: OutputFormat) -> str:
    if output_format == "jsonl":
        return render_jsonl(rows)
    return render_markdown(rows, title=title)


def preview_file(
    file_path: Path,
    doc_type: str,
    *,
    limit: int | None = None,
    full_content: bool = False,
    preview_chars: int = 500,
) -> list[ChunkView]:
    sample_metadata = _load_sample_metadata(file_path)
    effective_doc_type = str(sample_metadata.get(META_DOC_TYPE, doc_type))
    doc_id = derive_doc_id(file_path, effective_doc_type, sample_metadata)
    documents = _load_documents(file_path)
    if not documents:
        raise RuntimeError("文档解析结果为空")

    parse_hash = sha256_text(_normalize_parsed_text(documents))
    chunks = chunk_documents(documents=documents, doc_type=effective_doc_type)
    if not chunks:
        raise RuntimeError("文档分块结果为空")

    normalized_chunks = normalize_chunks(
        chunks,
        file_path,
        effective_doc_type,
        sample_metadata,
        doc_id=doc_id,
        file_hash=sha256_file(file_path),
        metadata_hash=_hash_metadata(sample_metadata),
        parse_hash=parse_hash,
        doc_version=1,
        parser_version=PARSER_VERSION,
        chunker_version=CHUNKER_VERSION,
        embedding_model=config.embedding.model,
    )
    if limit is not None:
        normalized_chunks = normalized_chunks[:limit]
    return build_chunk_views(
        normalized_chunks,
        full_content=full_content,
        preview_chars=preview_chars,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview chunks without writing ChromaDB.")
    parser.add_argument("file_path", type=Path)
    parser.add_argument("doc_type", choices=sorted(ALL_VALID_DOC_TYPES))
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--format", choices=["markdown", "jsonl"], default="markdown")
    parser.add_argument("--full-content", action="store_true")
    parser.add_argument("--preview-chars", type=int, default=500)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        rows = preview_file(
            args.file_path,
            args.doc_type,
            limit=args.limit,
            full_content=args.full_content,
            preview_chars=args.preview_chars,
        )
        output = render_rows(
            rows, title=f"Chunk Preview: {args.file_path}", output_format=args.format
        )
        sys.stdout.write(output)
        return 0
    except Exception as exc:
        print(f"preview failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
