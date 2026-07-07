"""CLI entrypoint for incremental document ingestion."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.ingestion.pipeline import ingest_directory
from src.schemas.constants import ALL_VALID_DOC_TYPES


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest documents into ChromaDB incrementally.")
    parser.add_argument("directory")
    parser.add_argument("doc_type", choices=sorted(ALL_VALID_DOC_TYPES))
    parser.add_argument("--full-scan", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    ingest_directory(Path(args.directory), args.doc_type, full_scan=args.full_scan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
