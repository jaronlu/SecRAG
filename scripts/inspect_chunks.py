"""Inspect chunks already stored in ChromaDB by doc_id."""

from __future__ import annotations

import argparse
import sys
from typing import Any, cast

from langchain_chroma import Chroma
from langchain_core.documents import Document

from scripts.preview_chunks import ChunkView, OutputFormat, build_chunk_views, render_rows
from src.schemas.constants import (
    CHROMA_COLLECTION_NAME,
    CHROMA_DEFAULT_PERSIST_DIR,
    META_CHUNK_ID,
    META_DOC_ID,
)


def _open_chroma(persist_directory: str) -> Chroma:
    return Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=None,
        persist_directory=persist_directory,
        create_collection_if_not_exists=False,
    )


def load_stored_chunks(
    doc_id: str,
    *,
    persist_directory: str = CHROMA_DEFAULT_PERSIST_DIR,
    limit: int | None = None,
) -> list[Document]:
    vectorstore = _open_chroma(persist_directory)
    results = vectorstore.get(
        where={META_DOC_ID: doc_id},
        limit=limit,
        include=["documents", "metadatas"],
    )
    ids = [str(item) for item in cast(list[Any], results.get("ids", []))]
    documents = cast(list[str], results.get("documents", []))
    metadatas = cast(list[dict[str, Any] | None], results.get("metadatas", []))

    chunks: list[Document] = []
    for index, content in enumerate(documents):
        metadata = dict(metadatas[index] or {}) if index < len(metadatas) else {}
        chunk_id = ids[index] if index < len(ids) else ""
        if index < len(ids):
            metadata.setdefault(META_CHUNK_ID, chunk_id)
        chunks.append(Document(page_content=content, metadata=metadata, id=chunk_id))
    return chunks


def inspect_doc_id(
    doc_id: str,
    *,
    persist_directory: str = CHROMA_DEFAULT_PERSIST_DIR,
    limit: int | None = None,
    full_content: bool = False,
    preview_chars: int = 500,
) -> list[ChunkView]:
    chunks = load_stored_chunks(
        doc_id,
        persist_directory=persist_directory,
        limit=limit,
    )
    return build_chunk_views(
        chunks,
        full_content=full_content,
        preview_chars=preview_chars,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect stored Chroma chunks by doc_id.")
    parser.add_argument("doc_id")
    parser.add_argument("--persist-directory", default=CHROMA_DEFAULT_PERSIST_DIR)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--format", choices=["markdown", "jsonl"], default="markdown")
    parser.add_argument("--full-content", action="store_true")
    parser.add_argument("--preview-chars", type=int, default=500)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        rows = inspect_doc_id(
            args.doc_id,
            persist_directory=args.persist_directory,
            limit=args.limit,
            full_content=args.full_content,
            preview_chars=args.preview_chars,
        )
        output = render_rows(
            rows,
            title=f"Chunk Inspect: {args.doc_id}",
            output_format=cast(OutputFormat, args.format),
        )
        sys.stdout.write(output)
        return 0
    except Exception as exc:
        print(f"inspect failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
