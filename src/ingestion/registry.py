"""SQLite-backed document registry and ingestion task state."""

from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from src.schemas.constants import (
    INGEST_ERROR_WORKER_LEASE_EXPIRED,
    INGEST_RUN_STATUS_FAILED,
    INGEST_RUN_STATUS_QUEUED,
    INGEST_RUN_STATUS_RUNNING,
)
from src.schemas.typed_dicts import IngestionRunItemView, IngestionRunSummary

REGISTRY_STATUS_ACTIVE = "active"
REGISTRY_STATUS_ARCHIVED = "archived"
REGISTRY_STATUS_FAILED = "failed"

INGEST_ACTION_CREATED = "created"
INGEST_ACTION_REPLACED = "replaced"
INGEST_ACTION_SKIPPED = "skipped"
INGEST_ACTION_ARCHIVED = "archived"
INGEST_ACTION_FAILED = "failed"


class ActiveIngestRunError(RuntimeError):
    def __init__(self, run_id: str) -> None:
        super().__init__(f"已有活动入库任务: {run_id}")
        self.run_id = run_id


class IngestWorkerLeaseLostError(RuntimeError):
    pass


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


@dataclass(frozen=True)
class IngestRunRecord:
    run_id: str
    category_id: str
    requested_by: str
    executor: str
    worker_id: str
    root_uri: str
    mode: str
    full_scan: bool
    queued_at: str
    started_at: str
    heartbeat_at: str
    lease_expires_at: str
    finished_at: str
    status: str
    error_code: str
    error: str


@dataclass(frozen=True)
class IngestRunFileRecord:
    run_id: str
    sequence: int
    relative_path: str
    file_hash: str
    metadata_hash: str
    doc_type: str


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
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @staticmethod
    def _columns(connection: sqlite3.Connection, table: str) -> set[str]:
        return {str(row["name"]) for row in connection.execute(f"PRAGMA table_info({table})")}

    @staticmethod
    def _add_missing_columns(
        connection: sqlite3.Connection,
        table: str,
        definitions: dict[str, str],
    ) -> None:
        columns = DocumentRegistryStore._columns(connection, table)
        for name, definition in definitions.items():
            if name not in columns:
                connection.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")

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
                    category_id TEXT NOT NULL DEFAULT '',
                    requested_by TEXT NOT NULL DEFAULT 'legacy',
                    executor TEXT NOT NULL DEFAULT 'legacy',
                    worker_id TEXT NOT NULL DEFAULT '',
                    root_uri TEXT NOT NULL,
                    mode TEXT NOT NULL DEFAULT 'incremental',
                    full_scan INTEGER NOT NULL DEFAULT 0,
                    queued_at TEXT NOT NULL DEFAULT '',
                    started_at TEXT NOT NULL,
                    heartbeat_at TEXT NOT NULL DEFAULT '',
                    lease_expires_at TEXT NOT NULL DEFAULT '',
                    finished_at TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'queued',
                    error_code TEXT NOT NULL DEFAULT '',
                    error TEXT NOT NULL DEFAULT ''
                );

                CREATE TABLE IF NOT EXISTS ingest_run_files (
                    run_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    relative_path TEXT NOT NULL,
                    file_hash TEXT NOT NULL,
                    metadata_hash TEXT NOT NULL,
                    doc_type TEXT NOT NULL,
                    PRIMARY KEY (run_id, sequence),
                    UNIQUE (run_id, relative_path)
                );

                CREATE TABLE IF NOT EXISTS ingest_run_items (
                    run_id TEXT NOT NULL,
                    doc_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL DEFAULT 0,
                    source_uri TEXT NOT NULL,
                    relative_path TEXT NOT NULL DEFAULT '',
                    action TEXT NOT NULL,
                    previous_hash TEXT NOT NULL DEFAULT '',
                    new_hash TEXT NOT NULL DEFAULT '',
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    processed_at TEXT NOT NULL DEFAULT '',
                    error_code TEXT NOT NULL DEFAULT '',
                    error TEXT NOT NULL DEFAULT '',
                    PRIMARY KEY (run_id, doc_id)
                );
                """
            )
            self._add_missing_columns(
                connection,
                "ingest_runs",
                {
                    "category_id": "TEXT NOT NULL DEFAULT ''",
                    "requested_by": "TEXT NOT NULL DEFAULT 'legacy'",
                    "executor": "TEXT NOT NULL DEFAULT 'legacy'",
                    "worker_id": "TEXT NOT NULL DEFAULT ''",
                    "queued_at": "TEXT NOT NULL DEFAULT ''",
                    "heartbeat_at": "TEXT NOT NULL DEFAULT ''",
                    "lease_expires_at": "TEXT NOT NULL DEFAULT ''",
                    "error_code": "TEXT NOT NULL DEFAULT ''",
                },
            )
            self._add_missing_columns(
                connection,
                "ingest_run_items",
                {
                    "sequence": "INTEGER NOT NULL DEFAULT 0",
                    "relative_path": "TEXT NOT NULL DEFAULT ''",
                    "processed_at": "TEXT NOT NULL DEFAULT ''",
                    "error_code": "TEXT NOT NULL DEFAULT ''",
                },
            )
            connection.execute("UPDATE ingest_runs SET queued_at = started_at WHERE queued_at = ''")
            connection.execute(
                """
                UPDATE ingest_runs
                SET lease_expires_at = started_at
                WHERE status IN (?, ?) AND lease_expires_at = ''
                """,
                (INGEST_RUN_STATUS_QUEUED, INGEST_RUN_STATUS_RUNNING),
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_ingest_runs_status ON ingest_runs(status)"
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_ingest_run_items_sequence
                ON ingest_run_items(run_id, sequence)
                """
            )

    @staticmethod
    def _expire_stale_in_connection(connection: sqlite3.Connection, now: str) -> None:
        connection.execute(
            """
            UPDATE ingest_runs
            SET status = ?, finished_at = ?, error_code = ?,
                error = '入库 worker 租约已过期'
            WHERE status IN (?, ?)
              AND lease_expires_at != ''
              AND lease_expires_at <= ?
            """,
            (
                INGEST_RUN_STATUS_FAILED,
                now,
                INGEST_ERROR_WORKER_LEASE_EXPIRED,
                INGEST_RUN_STATUS_QUEUED,
                INGEST_RUN_STATUS_RUNNING,
                now,
            ),
        )

    def expire_stale_runs(self, now: str) -> None:
        with self._connect() as connection:
            self._expire_stale_in_connection(connection, now)

    def create_queued_run(
        self,
        *,
        run_id: str,
        category_id: str,
        requested_by: str,
        executor: str,
        root_uri: str,
        full_scan: bool,
        queued_at: str,
        lease_expires_at: str,
        files: list[IngestRunFileRecord],
    ) -> None:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            self._expire_stale_in_connection(connection, queued_at)
            active = connection.execute(
                """
                SELECT run_id FROM ingest_runs
                WHERE status IN (?, ?)
                ORDER BY queued_at LIMIT 1
                """,
                (INGEST_RUN_STATUS_QUEUED, INGEST_RUN_STATUS_RUNNING),
            ).fetchone()
            if active is not None:
                raise ActiveIngestRunError(str(active["run_id"]))
            connection.execute(
                """
                INSERT INTO ingest_runs (
                    run_id, category_id, requested_by, executor, root_uri, mode,
                    full_scan, queued_at, started_at, lease_expires_at, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, '', ?, ?)
                """,
                (
                    run_id,
                    category_id,
                    requested_by,
                    executor,
                    root_uri,
                    "full" if full_scan else "incremental",
                    int(full_scan),
                    queued_at,
                    lease_expires_at,
                    INGEST_RUN_STATUS_QUEUED,
                ),
            )
            connection.executemany(
                """
                INSERT INTO ingest_run_files (
                    run_id, sequence, relative_path, file_hash, metadata_hash, doc_type
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        item.run_id,
                        item.sequence,
                        item.relative_path,
                        item.file_hash,
                        item.metadata_hash,
                        item.doc_type,
                    )
                    for item in files
                ],
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def start_run(self, run_id: str, root_uri: str, full_scan: bool, started_at: str) -> None:
        """Backward-compatible test/helper entry for an already-running legacy run."""
        mode = "full" if full_scan else "incremental"
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO ingest_runs (
                    run_id, root_uri, mode, full_scan, queued_at, started_at,
                    heartbeat_at, lease_expires_at, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    root_uri,
                    mode,
                    int(full_scan),
                    started_at,
                    started_at,
                    started_at,
                    started_at,
                    INGEST_RUN_STATUS_RUNNING,
                ),
            )

    def claim_run(
        self,
        run_id: str,
        worker_id: str,
        started_at: str,
        lease_expires_at: str,
    ) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE ingest_runs
                SET status = ?, worker_id = ?, started_at = ?, heartbeat_at = ?,
                    lease_expires_at = ?
                WHERE run_id = ? AND status = ? AND lease_expires_at > ?
                """,
                (
                    INGEST_RUN_STATUS_RUNNING,
                    worker_id,
                    started_at,
                    started_at,
                    lease_expires_at,
                    run_id,
                    INGEST_RUN_STATUS_QUEUED,
                    started_at,
                ),
            )
        return cursor.rowcount == 1

    def heartbeat(
        self,
        run_id: str,
        worker_id: str,
        now: str,
        lease_expires_at: str,
    ) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE ingest_runs
                SET heartbeat_at = ?, lease_expires_at = ?
                WHERE run_id = ? AND worker_id = ? AND status = ?
                  AND lease_expires_at > ?
                """,
                (
                    now,
                    lease_expires_at,
                    run_id,
                    worker_id,
                    INGEST_RUN_STATUS_RUNNING,
                    now,
                ),
            )
        return cursor.rowcount == 1

    def owns_run(self, run_id: str, worker_id: str, now: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT 1 FROM ingest_runs
                WHERE run_id = ? AND worker_id = ? AND status = ?
                  AND lease_expires_at > ?
                """,
                (run_id, worker_id, INGEST_RUN_STATUS_RUNNING, now),
            ).fetchone()
        return row is not None

    def finish_owned_run(
        self,
        *,
        run_id: str,
        worker_id: str,
        status: str,
        finished_at: str,
        error_code: str = "",
        error: str = "",
    ) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE ingest_runs
                SET status = ?, finished_at = ?, error_code = ?, error = ?
                WHERE run_id = ? AND worker_id = ? AND status = ?
                  AND lease_expires_at > ?
                """,
                (
                    status,
                    finished_at,
                    error_code,
                    error,
                    run_id,
                    worker_id,
                    INGEST_RUN_STATUS_RUNNING,
                    finished_at,
                ),
            )
        return cursor.rowcount == 1

    def finish_run(
        self,
        run_id: str,
        status: str,
        finished_at: str,
        error: str = "",
        error_code: str = "",
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE ingest_runs
                SET status = ?, finished_at = ?, error_code = ?, error = ?
                WHERE run_id = ?
                """,
                (status, finished_at, error_code, error, run_id),
            )

    def active_run_id(self, now: str | None = None) -> str | None:
        if now is not None:
            self.expire_stale_runs(now)
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT run_id FROM ingest_runs
                WHERE status IN (?, ?)
                ORDER BY queued_at LIMIT 1
                """,
                (INGEST_RUN_STATUS_QUEUED, INGEST_RUN_STATUS_RUNNING),
            ).fetchone()
        return None if row is None else str(row["run_id"])

    def get_run(self, run_id: str) -> IngestRunRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM ingest_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        return None if row is None else _run_from_row(row)

    def list_recent_runs(self, limit: int) -> list[IngestRunRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM ingest_runs ORDER BY queued_at DESC, run_id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [_run_from_row(row) for row in rows]

    def list_run_files(self, run_id: str) -> list[IngestRunFileRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM ingest_run_files
                WHERE run_id = ? ORDER BY sequence
                """,
                (run_id,),
            ).fetchall()
        return [
            IngestRunFileRecord(
                run_id=str(row["run_id"]),
                sequence=int(row["sequence"]),
                relative_path=str(row["relative_path"]),
                file_hash=str(row["file_hash"]),
                metadata_hash=str(row["metadata_hash"]),
                doc_type=str(row["doc_type"]),
            )
            for row in rows
        ]

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
        *,
        sequence: int = 0,
        relative_path: str = "",
        processed_at: str = "",
        error_code: str = "",
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO ingest_run_items (
                    run_id, doc_id, sequence, source_uri, relative_path, action,
                    previous_hash, new_hash, chunk_count, processed_at, error_code, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    doc_id,
                    sequence,
                    source_uri,
                    relative_path,
                    action,
                    previous_hash,
                    new_hash,
                    chunk_count,
                    processed_at or utc_now(),
                    error_code,
                    error,
                ),
            )

    def list_run_items(self, run_id: str) -> list[IngestionRunItemView]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT doc_id, sequence, relative_path, action, chunk_count,
                       processed_at, error_code, error
                FROM ingest_run_items
                WHERE run_id = ?
                ORDER BY CASE WHEN sequence = 0 THEN 1 ELSE 0 END, sequence, rowid
                """,
                (run_id,),
            ).fetchall()
        return [
            {
                "doc_id": str(row["doc_id"]),
                "sequence": int(row["sequence"]),
                "relative_path": str(row["relative_path"]),
                "action": str(row["action"]),
                "chunk_count": int(row["chunk_count"]),
                "processed_at": str(row["processed_at"]),
                "error_code": str(row["error_code"]),
                "error": str(row["error"]),
            }
            for row in rows
        ]

    def summarize_run(self, run_id: str) -> IngestionRunSummary | None:
        run = self.get_run(run_id)
        if run is None:
            return None
        with self._connect() as connection:
            total_files = int(
                connection.execute(
                    "SELECT COUNT(*) FROM ingest_run_files WHERE run_id = ?", (run_id,)
                ).fetchone()[0]
            )
            row = connection.execute(
                """
                SELECT COUNT(*) AS processed_files,
                       SUM(CASE WHEN action = ? THEN 1 ELSE 0 END) AS created,
                       SUM(CASE WHEN action = ? THEN 1 ELSE 0 END) AS replaced,
                       SUM(CASE WHEN action = ? THEN 1 ELSE 0 END) AS skipped,
                       SUM(CASE WHEN action = ? THEN 1 ELSE 0 END) AS archived,
                       SUM(CASE WHEN action = ? THEN 1 ELSE 0 END) AS failed
                FROM ingest_run_items WHERE run_id = ?
                """,
                (
                    INGEST_ACTION_CREATED,
                    INGEST_ACTION_REPLACED,
                    INGEST_ACTION_SKIPPED,
                    INGEST_ACTION_ARCHIVED,
                    INGEST_ACTION_FAILED,
                    run_id,
                ),
            ).fetchone()
        return {
            "run_id": run.run_id,
            "category_id": run.category_id,
            "status": run.status,
            "queued_at": run.queued_at,
            "started_at": run.started_at or None,
            "finished_at": run.finished_at or None,
            "total_files": total_files,
            "processed_files": int(row["processed_files"] or 0),
            "created": int(row["created"] or 0),
            "replaced": int(row["replaced"] or 0),
            "skipped": int(row["skipped"] or 0),
            "archived": int(row["archived"] or 0),
            "failed": int(row["failed"] or 0),
            "error_code": run.error_code,
            "error": run.error,
        }

    def get_document(self, doc_id: str) -> DocumentRegistryRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM document_registry WHERE doc_id = ?",
                (doc_id,),
            ).fetchone()
        return None if row is None else _record_from_row(row)

    def upsert_success(self, update: DocumentRegistryUpdate) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO document_registry (
                    doc_id, source_uri, relative_path, doc_type, title, stock_code,
                    publish_date, file_hash, metadata_hash, parse_hash, parser_version,
                    chunker_version, embedding_model, chunk_count, doc_version, status,
                    last_seen_at, last_ingested_at, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '')
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
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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


def _run_from_row(row: sqlite3.Row) -> IngestRunRecord:
    return IngestRunRecord(
        run_id=str(row["run_id"]),
        category_id=str(row["category_id"]),
        requested_by=str(row["requested_by"]),
        executor=str(row["executor"]),
        worker_id=str(row["worker_id"]),
        root_uri=str(row["root_uri"]),
        mode=str(row["mode"]),
        full_scan=bool(row["full_scan"]),
        queued_at=str(row["queued_at"]),
        started_at=str(row["started_at"]),
        heartbeat_at=str(row["heartbeat_at"]),
        lease_expires_at=str(row["lease_expires_at"]),
        finished_at=str(row["finished_at"]),
        status=str(row["status"]),
        error_code=str(row["error_code"]),
        error=str(row["error"]),
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
