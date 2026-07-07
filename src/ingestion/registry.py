"""SQLite-backed document registry for incremental ingestion."""

from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

REGISTRY_STATUS_ACTIVE = "active"
REGISTRY_STATUS_ARCHIVED = "archived"
REGISTRY_STATUS_FAILED = "failed"

INGEST_RUN_STATUS_RUNNING = "running"
INGEST_RUN_STATUS_SUCCESS = "success"
INGEST_RUN_STATUS_FAILED = "failed"

INGEST_ACTION_CREATED = "created"
INGEST_ACTION_REPLACED = "replaced"
INGEST_ACTION_SKIPPED = "skipped"
INGEST_ACTION_ARCHIVED = "archived"
INGEST_ACTION_FAILED = "failed"


@dataclass(frozen=True)
class DocumentRegistryRecord:
    doc_id: str
    source_uri: str
    relative_path: str
    doc_type: str
    title: str
    stock_code: str
    publish_date: str
    file_hash: str
    metadata_hash: str
    parse_hash: str
    parser_version: str
    chunker_version: str
    embedding_model: str
    chunk_count: int
    doc_version: int
    status: str
    last_seen_at: str
    last_ingested_at: str
    error: str


@dataclass(frozen=True)
class DocumentRegistryUpdate:
    doc_id: str
    source_uri: str
    relative_path: str
    doc_type: str
    title: str
    stock_code: str
    publish_date: str
    file_hash: str
    metadata_hash: str
    parse_hash: str
    parser_version: str
    chunker_version: str
    embedding_model: str
    chunk_count: int
    doc_version: int
    last_seen_at: str
    last_ingested_at: str


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def create_run_id() -> str:
    return uuid.uuid4().hex


class DocumentRegistryStore:
    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_schema(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS document_registry (
                    doc_id TEXT PRIMARY KEY,
                    source_uri TEXT NOT NULL,
                    relative_path TEXT NOT NULL,
                    doc_type TEXT NOT NULL,
                    title TEXT NOT NULL DEFAULT '',
                    stock_code TEXT NOT NULL DEFAULT '',
                    publish_date TEXT NOT NULL DEFAULT '',
                    file_hash TEXT NOT NULL,
                    metadata_hash TEXT NOT NULL DEFAULT '',
                    parse_hash TEXT NOT NULL DEFAULT '',
                    parser_version TEXT NOT NULL DEFAULT '',
                    chunker_version TEXT NOT NULL DEFAULT '',
                    embedding_model TEXT NOT NULL DEFAULT '',
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    doc_version INTEGER NOT NULL DEFAULT 1,
                    status TEXT NOT NULL DEFAULT 'active',
                    last_seen_at TEXT NOT NULL,
                    last_ingested_at TEXT NOT NULL DEFAULT '',
                    error TEXT NOT NULL DEFAULT ''
                );

                CREATE TABLE IF NOT EXISTS ingest_runs (
                    run_id TEXT PRIMARY KEY,
                    root_uri TEXT NOT NULL,
                    mode TEXT NOT NULL DEFAULT 'incremental',
                    full_scan INTEGER NOT NULL DEFAULT 0,
                    started_at TEXT NOT NULL,
                    finished_at TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'running',
                    error TEXT NOT NULL DEFAULT ''
                );

                CREATE TABLE IF NOT EXISTS ingest_run_items (
                    run_id TEXT NOT NULL,
                    doc_id TEXT NOT NULL,
                    source_uri TEXT NOT NULL,
                    action TEXT NOT NULL,
                    previous_hash TEXT NOT NULL DEFAULT '',
                    new_hash TEXT NOT NULL DEFAULT '',
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    error TEXT NOT NULL DEFAULT '',
                    PRIMARY KEY (run_id, doc_id)
                );
                """
            )

    def start_run(self, run_id: str, root_uri: str, full_scan: bool, started_at: str) -> None:
        mode = "full" if full_scan else "incremental"
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO ingest_runs (run_id, root_uri, mode, full_scan, started_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (run_id, root_uri, mode, int(full_scan), started_at),
            )

    def finish_run(self, run_id: str, status: str, finished_at: str, error: str = "") -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE ingest_runs
                SET status = ?, finished_at = ?, error = ?
                WHERE run_id = ?
                """,
                (status, finished_at, error, run_id),
            )

    def record_run_item(
        self,
        run_id: str,
        doc_id: str,
        source_uri: str,
        action: str,
        previous_hash: str = "",
        new_hash: str = "",
        chunk_count: int = 0,
        error: str = "",
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO ingest_run_items (
                    run_id, doc_id, source_uri, action, previous_hash, new_hash, chunk_count, error
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    doc_id,
                    source_uri,
                    action,
                    previous_hash,
                    new_hash,
                    chunk_count,
                    error,
                ),
            )

    def get_document(self, doc_id: str) -> DocumentRegistryRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM document_registry WHERE doc_id = ?",
                (doc_id,),
            ).fetchone()
        if row is None:
            return None
        return _record_from_row(row)

    def upsert_success(self, update: DocumentRegistryUpdate) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO document_registry (
                    doc_id, source_uri, relative_path, doc_type, title, stock_code,
                    publish_date, file_hash, metadata_hash, parse_hash, parser_version,
                    chunker_version, embedding_model, chunk_count, doc_version, status,
                    last_seen_at, last_ingested_at, error
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '')
                ON CONFLICT(doc_id) DO UPDATE SET
                    source_uri = excluded.source_uri,
                    relative_path = excluded.relative_path,
                    doc_type = excluded.doc_type,
                    title = excluded.title,
                    stock_code = excluded.stock_code,
                    publish_date = excluded.publish_date,
                    file_hash = excluded.file_hash,
                    metadata_hash = excluded.metadata_hash,
                    parse_hash = excluded.parse_hash,
                    parser_version = excluded.parser_version,
                    chunker_version = excluded.chunker_version,
                    embedding_model = excluded.embedding_model,
                    chunk_count = excluded.chunk_count,
                    doc_version = excluded.doc_version,
                    status = excluded.status,
                    last_seen_at = excluded.last_seen_at,
                    last_ingested_at = excluded.last_ingested_at,
                    error = ''
                """,
                (
                    update.doc_id,
                    update.source_uri,
                    update.relative_path,
                    update.doc_type,
                    update.title,
                    update.stock_code,
                    update.publish_date,
                    update.file_hash,
                    update.metadata_hash,
                    update.parse_hash,
                    update.parser_version,
                    update.chunker_version,
                    update.embedding_model,
                    update.chunk_count,
                    update.doc_version,
                    REGISTRY_STATUS_ACTIVE,
                    update.last_seen_at,
                    update.last_ingested_at,
                ),
            )

    def mark_seen(self, doc_id: str, seen_at: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE document_registry
                SET last_seen_at = ?, status = ?, error = ''
                WHERE doc_id = ?
                """,
                (seen_at, REGISTRY_STATUS_ACTIVE, doc_id),
            )

    def mark_failed(
        self,
        doc_id: str,
        source_uri: str,
        relative_path: str,
        doc_type: str,
        title: str,
        stock_code: str,
        publish_date: str,
        file_hash: str,
        metadata_hash: str,
        parser_version: str,
        chunker_version: str,
        embedding_model: str,
        seen_at: str,
        error: str,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO document_registry (
                    doc_id, source_uri, relative_path, doc_type, title, stock_code,
                    publish_date, file_hash, metadata_hash, parser_version, chunker_version,
                    embedding_model, status, last_seen_at, error
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(doc_id) DO UPDATE SET
                    source_uri = excluded.source_uri,
                    relative_path = excluded.relative_path,
                    doc_type = excluded.doc_type,
                    title = excluded.title,
                    stock_code = excluded.stock_code,
                    publish_date = excluded.publish_date,
                    file_hash = excluded.file_hash,
                    metadata_hash = excluded.metadata_hash,
                    parser_version = excluded.parser_version,
                    chunker_version = excluded.chunker_version,
                    embedding_model = excluded.embedding_model,
                    status = excluded.status,
                    last_seen_at = excluded.last_seen_at,
                    error = excluded.error
                """,
                (
                    doc_id,
                    source_uri,
                    relative_path,
                    doc_type,
                    title,
                    stock_code,
                    publish_date,
                    file_hash,
                    metadata_hash,
                    parser_version,
                    chunker_version,
                    embedding_model,
                    REGISTRY_STATUS_FAILED,
                    seen_at,
                    error,
                ),
            )

    def active_documents(self) -> list[DocumentRegistryRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM document_registry WHERE status = ?",
                (REGISTRY_STATUS_ACTIVE,),
            ).fetchall()
        return [_record_from_row(row) for row in rows]

    def mark_archived(self, doc_id: str, seen_at: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE document_registry
                SET status = ?, last_seen_at = ?, error = ''
                WHERE doc_id = ?
                """,
                (REGISTRY_STATUS_ARCHIVED, seen_at, doc_id),
            )


def _record_from_row(row: sqlite3.Row) -> DocumentRegistryRecord:
    return DocumentRegistryRecord(
        doc_id=str(row["doc_id"]),
        source_uri=str(row["source_uri"]),
        relative_path=str(row["relative_path"]),
        doc_type=str(row["doc_type"]),
        title=str(row["title"]),
        stock_code=str(row["stock_code"]),
        publish_date=str(row["publish_date"]),
        file_hash=str(row["file_hash"]),
        metadata_hash=str(row["metadata_hash"]),
        parse_hash=str(row["parse_hash"]),
        parser_version=str(row["parser_version"]),
        chunker_version=str(row["chunker_version"]),
        embedding_model=str(row["embedding_model"]),
        chunk_count=int(row["chunk_count"]),
        doc_version=int(row["doc_version"]),
        status=str(row["status"]),
        last_seen_at=str(row["last_seen_at"]),
        last_ingested_at=str(row["last_ingested_at"]),
        error=str(row["error"]),
    )
