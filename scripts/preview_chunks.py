"""CLI entrypoint for previewing stored ChromaDB chunks."""

from __future__ import annotations

import argparse
import sys

from src.ingestion.chunk_view import inspect_doc_id, render_rows
from src.schemas.constants import CHROMA_DEFAULT_PERSIST_DIR


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview stored Chroma chunks by doc_id.")
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
            title=f"Chunk Preview: {args.doc_id}",
            output_format=args.format,
        )
        sys.stdout.write(output)
        return 0
    except Exception as exc:
        print(f"preview failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
