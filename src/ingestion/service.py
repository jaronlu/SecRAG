"""Shared ingestion orchestration for CLI and API entry points."""

from __future__ import annotations

import threading
import uuid
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import unquote, urlparse

from langchain_huggingface import HuggingFaceEmbeddings

from src.config import config
from src.ingestion import pipeline
from src.ingestion.catalog import (
    CategoryPreflight,
    UnknownIngestionCategoryError,
    UnsafeIngestionPathError,
    ensure_safe_source_path,
    get_category_config,
    get_ingestion_catalog,
    preflight_category,
)
from src.ingestion.chunk_view import inspect_doc_id
from src.ingestion.embedder import delete_chunk_ids, get_embedding_model, list_chunk_ids_by_doc_id
from src.ingestion.identity import (
    derive_doc_id,
    file_uri_in_directory,
    hash_metadata,
    load_sample_metadata,
    sha256_file,
    sha256_text,
)
from src.ingestion.registry import (
    INGEST_ACTION_ARCHIVED,
    INGEST_ACTION_FAILED,
    ActiveIngestRunError,
    DocumentRegistryStore,
    IngestRunFileRecord,
    IngestRunRecord,
    IngestWorkerLeaseLostError,
    create_run_id,
)
from src.schemas.constants import (
    ALL_VALID_DOC_TYPES,
    CHROMA_DEFAULT_PERSIST_DIR,
    DATA_RAW_ROOT,
    INGEST_ERROR_DOCUMENT_PROCESSING_FAILED,
    INGEST_ERROR_EMBEDDING_UNAVAILABLE,
    INGEST_ERROR_SOURCE_CHANGED_AFTER_ENQUEUE,
    INGEST_ERROR_UNSAFE_SOURCE_PATH,
    INGEST_REGISTRY_DB_PATH,
    INGEST_RUN_STATUS_FAILED,
    INGEST_RUN_STATUS_SUCCESS,
    INGEST_WORKER_HEARTBEAT_SECONDS,
    INGEST_WORKER_LEASE_SECONDS,
    META_DOC_TYPE,
)
from src.schemas.typed_dicts import (
    IngestionCategoryConfig,
    IngestionCategorySummary,
    IngestionChunkView,
    IngestionFile,
    IngestionRunItemView,
    IngestionRunSummary,
)


class CategoryPreflightError(ValueError):
    def __init__(self, message: str, files: list[IngestionFile] | None = None) -> None:
        super().__init__(message)
        self.files = files or []


class IngestRunNotFoundError(LookupError):
    pass


class IngestDocumentNotFoundError(LookupError):
    pass


def _format_time(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="seconds")


class IngestionService:
    def __init__(
        self,
        *,
        project_root: Path | None = None,
        registry_path: Path | str = INGEST_REGISTRY_DB_PATH,
        persist_directory: str = CHROMA_DEFAULT_PERSIST_DIR,
        catalog: tuple[IngestionCategoryConfig, ...] | None = None,
        embedding_model_factory: Callable[[str], HuggingFaceEmbeddings] = get_embedding_model,
        now_factory: Callable[[], datetime] | None = None,
        heartbeat_seconds: int = INGEST_WORKER_HEARTBEAT_SECONDS,
        lease_seconds: int = INGEST_WORKER_LEASE_SECONDS,
    ) -> None:
        self.project_root = (project_root or Path.cwd()).absolute()
        self.data_raw_root = self.project_root / DATA_RAW_ROOT
        self.registry = DocumentRegistryStore(registry_path)
        self.persist_directory = persist_directory
        self.catalog = catalog or get_ingestion_catalog()
        self.embedding_model_factory = embedding_model_factory
        self.now_factory = now_factory or (lambda: datetime.now(UTC))
        self.heartbeat_seconds = heartbeat_seconds
        self.lease_seconds = lease_seconds

    def _now(self) -> datetime:
        value = self.now_factory()
        return value if value.tzinfo is not None else value.replace(tzinfo=UTC)

    def _preflight(self, category_id: str) -> CategoryPreflight:
        category = get_category_config(category_id, self.catalog)
        return preflight_category(
            category,
            project_root=self.project_root,
            data_raw_root=self.data_raw_root,
        )

    def list_categories(self) -> tuple[list[IngestionCategorySummary], str | None]:
        summaries: list[IngestionCategorySummary] = []
        for category in self.catalog:
            try:
                summaries.append(
                    preflight_category(
                        category,
                        project_root=self.project_root,
                        data_raw_root=self.data_raw_root,
                    ).summary
                )
            except (FileNotFoundError, UnsafeIngestionPathError) as exc:
                summaries.append({
                    **category,
                    "file_count": 0,
                    "manifest_count": 0,
                    "invalid_manifest_count": 1,
                    "ready": False,
                    "error_code": INGEST_ERROR_UNSAFE_SOURCE_PATH
                    if isinstance(exc, UnsafeIngestionPathError)
                    else "",
                    "error": "分类目录不可安全访问",
                })
        return summaries, self.registry.active_run_id(_format_time(self._now()))

    def list_category_files(self, category_id: str) -> list[IngestionFile]:
        return self._preflight(category_id).files

    def list_document_chunks(
        self,
        doc_id: str,
        *,
        offset: int,
        limit: int,
    ) -> tuple[list[IngestionChunkView], int]:
        rows = inspect_doc_id(
            doc_id,
            persist_directory=self.persist_directory,
            full_content=True,
        )
        if not rows:
            raise IngestDocumentNotFoundError(doc_id)
        chunks: list[IngestionChunkView] = [
            {
                "chunk_id": row["chunk_id"],
                "chunk_index": row["chunk_index"],
                "chunk_hash": row["chunk_hash"],
                "doc_type": row["doc_type"],
                "title": row["title"],
                "stock_code": row["stock_code"],
                "date": row["date"],
                "page_number": row["page_number"],
                "content_length": row["content_length"],
                "content": row.get("content", ""),
                "permission_level": row["permission_level"],
                "allowed_roles": row["allowed_roles"],
                "parser_version": row["parser_version"],
                "chunker_version": row["chunker_version"],
                "embedding_model": row["embedding_model"],
            }
            for row in rows[offset : offset + limit]
        ]
        return chunks, len(rows)

    def _create_from_preflight(
        self,
        preflight: CategoryPreflight,
        *,
        category_id: str,
        requested_by: str,
        executor: str,
        full_scan: bool,
    ) -> IngestionRunSummary:
        if not preflight.summary["ready"]:
            if preflight.summary["file_count"] == 0:
                raise CategoryPreflightError("分类中没有可入库业务文件", preflight.files)
            raise CategoryPreflightError("分类预检失败", preflight.files)

        run_id = create_run_id()
        queued_at_dt = self._now()
        queued_at = _format_time(queued_at_dt)
        lease_expires_at = _format_time(queued_at_dt + timedelta(seconds=self.lease_seconds))
        files = [
            IngestRunFileRecord(
                run_id=run_id,
                sequence=sequence,
                relative_path=item.relative_path,
                file_hash=item.file_hash,
                metadata_hash=item.metadata_hash,
                doc_type=item.doc_type,
            )
            for sequence, item in enumerate(preflight.snapshots, start=1)
        ]
        self.registry.create_queued_run(
            run_id=run_id,
            category_id=category_id,
            requested_by=requested_by,
            executor=executor,
            root_uri=preflight.category_root.resolve().as_uri(),
            full_scan=full_scan,
            queued_at=queued_at,
            lease_expires_at=lease_expires_at,
            files=files,
        )
        summary = self.registry.summarize_run(run_id)
        assert summary is not None
        return summary

    def create_run(
        self,
        category_id: str,
        *,
        requested_by: str,
        executor: str = "api",
    ) -> IngestionRunSummary:
        return self._create_from_preflight(
            self._preflight(category_id),
            category_id=category_id,
            requested_by=requested_by,
            executor=executor,
            full_scan=False,
        )

    def create_directory_run(
        self,
        directory: Path,
        doc_type: str,
        *,
        full_scan: bool,
        requested_by: str = "cli",
    ) -> IngestionRunSummary:
        if doc_type not in ALL_VALID_DOC_TYPES:
            raise CategoryPreflightError(f"不支持的文档类型: {doc_type}")
        try:
            relative_path = str(directory.absolute().relative_to(self.project_root))
        except ValueError as exc:
            raise UnsafeIngestionPathError("CLI 入库目录必须位于项目目录内") from exc
        category: IngestionCategoryConfig = {
            "category_id": "",
            "label": "CLI",
            "group": "CLI",
            "relative_path": relative_path,
            "default_doc_type": doc_type,
            "allowed_doc_types": sorted(ALL_VALID_DOC_TYPES),
        }
        preflight = preflight_category(
            category,
            project_root=self.project_root,
            data_raw_root=self.data_raw_root,
        )
        return self._create_from_preflight(
            preflight,
            category_id="",
            requested_by=requested_by,
            executor="cli",
            full_scan=full_scan,
        )

    def _lease_expiry(self, now: datetime) -> str:
        return _format_time(now + timedelta(seconds=self.lease_seconds))

    def _run_root(self, run: IngestRunRecord) -> Path:
        parsed = urlparse(run.root_uri)
        if parsed.scheme != "file":
            raise UnsafeIngestionPathError("入库根路径不是本地文件 URI")
        return Path(unquote(parsed.path))

    def _check_owner(self, run_id: str, worker_id: str) -> None:
        if not self.registry.owns_run(run_id, worker_id, _format_time(self._now())):
            raise IngestWorkerLeaseLostError(run_id)

    def _record_snapshot_failure(
        self,
        item: IngestRunFileRecord,
        file_path: Path,
        error_code: str,
        error: str,
    ) -> None:
        self.registry.record_run_item(
            run_id=item.run_id,
            doc_id=f"snapshot:{sha256_text(item.relative_path)[:16]}",
            source_uri=file_path.absolute().as_uri(),
            action=INGEST_ACTION_FAILED,
            sequence=item.sequence,
            relative_path=item.relative_path,
            processed_at=_format_time(self._now()),
            error_code=error_code,
            error=error,
        )

    def _validate_snapshot(
        self,
        item: IngestRunFileRecord,
        category_root: Path,
    ) -> tuple[Path, dict]:
        file_path = self.project_root / item.relative_path
        manifest_path = file_path.with_name(f"{file_path.name}.meta.json")
        try:
            ensure_safe_source_path(file_path, category_root, self.data_raw_root)
            ensure_safe_source_path(manifest_path, category_root, self.data_raw_root)
        except UnsafeIngestionPathError:
            raise
        except FileNotFoundError as exc:
            raise CategoryPreflightError("文件或 manifest 在排队后被删除") from exc
        metadata = load_sample_metadata(file_path)
        if (
            sha256_file(file_path) != item.file_hash
            or hash_metadata(metadata) != item.metadata_hash
            or metadata.get(META_DOC_TYPE) != item.doc_type
        ):
            raise CategoryPreflightError("文件或 manifest 在排队后发生变化")
        return file_path, metadata

    def _archive_missing(
        self,
        *,
        run: IngestRunRecord,
        run_id: str,
        worker_id: str,
        seen_doc_ids: set[str],
        embedding_model: HuggingFaceEmbeddings,
        next_sequence: int,
    ) -> int:
        if not run.full_scan:
            return next_sequence
        root = self._run_root(run)
        for record in self.registry.active_documents():
            if record.doc_id in seen_doc_ids or not file_uri_in_directory(record.source_uri, root):
                continue
            self._check_owner(run_id, worker_id)
            old_ids = list_chunk_ids_by_doc_id(
                doc_id=record.doc_id,
                persist_directory=self.persist_directory,
                embedding_model=embedding_model,
            )
            self._check_owner(run_id, worker_id)
            delete_chunk_ids(
                chunk_ids=old_ids,
                persist_directory=self.persist_directory,
                embedding_model=embedding_model,
            )
            processed_at = _format_time(self._now())
            self.registry.mark_archived(record.doc_id, processed_at)
            self.registry.record_run_item(
                run_id=run_id,
                doc_id=record.doc_id,
                source_uri=record.source_uri,
                action=INGEST_ACTION_ARCHIVED,
                previous_hash=record.file_hash,
                sequence=next_sequence,
                relative_path=record.relative_path,
                processed_at=processed_at,
            )
            next_sequence += 1
        return next_sequence

    def execute_run(self, run_id: str) -> IngestionRunSummary:
        run = self.registry.get_run(run_id)
        if run is None:
            raise IngestRunNotFoundError(run_id)
        worker_id = uuid.uuid4().hex
        started_at_dt = self._now()
        started_at = _format_time(started_at_dt)
        if not self.registry.claim_run(
            run_id,
            worker_id,
            started_at,
            self._lease_expiry(started_at_dt),
        ):
            summary = self.registry.summarize_run(run_id)
            if summary is None:
                raise IngestRunNotFoundError(run_id)
            return summary

        stop_heartbeat = threading.Event()
        lost_lease = threading.Event()

        def heartbeat_loop() -> None:
            while not stop_heartbeat.wait(self.heartbeat_seconds):
                now = self._now()
                if not self.registry.heartbeat(
                    run_id,
                    worker_id,
                    _format_time(now),
                    self._lease_expiry(now),
                ):
                    lost_lease.set()
                    return

        heartbeat_thread = threading.Thread(
            target=heartbeat_loop,
            name=f"ingest-heartbeat-{run_id[:8]}",
            daemon=True,
        )
        heartbeat_thread.start()
        try:
            try:
                embedding_model = self.embedding_model_factory(config.embedding.model)
            except Exception:
                finished_at = _format_time(self._now())
                self.registry.finish_owned_run(
                    run_id=run_id,
                    worker_id=worker_id,
                    status=INGEST_RUN_STATUS_FAILED,
                    finished_at=finished_at,
                    error_code=INGEST_ERROR_EMBEDDING_UNAVAILABLE,
                    error="Embedding 模型不可用",
                )
                summary = self.registry.summarize_run(run_id)
                assert summary is not None
                return summary

            category_root = self._run_root(run)
            snapshots = self.registry.list_run_files(run_id)
            seen_doc_ids: set[str] = set()
            failed = False
            for item in snapshots:
                if lost_lease.is_set():
                    raise IngestWorkerLeaseLostError(run_id)
                self._check_owner(run_id, worker_id)
                file_path = self.project_root / item.relative_path
                try:
                    file_path, metadata = self._validate_snapshot(item, category_root)
                except UnsafeIngestionPathError:
                    self._record_snapshot_failure(
                        item,
                        file_path,
                        INGEST_ERROR_UNSAFE_SOURCE_PATH,
                        "文件或 manifest 路径不安全",
                    )
                    failed = True
                    continue
                except (CategoryPreflightError, OSError, ValueError, TypeError):
                    self._record_snapshot_failure(
                        item,
                        file_path,
                        INGEST_ERROR_SOURCE_CHANGED_AFTER_ENQUEUE,
                        "文件或 manifest 在排队后发生变化",
                    )
                    failed = True
                    continue

                doc_id = derive_doc_id(file_path, item.doc_type, metadata)
                seen_doc_ids.add(doc_id)
                action = pipeline.ingest_document(
                    file_path=file_path,
                    doc_type=item.doc_type,
                    registry_store=self.registry,
                    run_id=run_id,
                    root_dir=self.project_root,
                    embedding_model=embedding_model,
                    persist_directory=self.persist_directory,
                    sequence=item.sequence,
                    relative_path_override=item.relative_path,
                    ownership_check=lambda: self._check_owner(run_id, worker_id),
                )
                failed = failed or action == INGEST_ACTION_FAILED

            if not failed:
                self._archive_missing(
                    run=run,
                    run_id=run_id,
                    worker_id=worker_id,
                    seen_doc_ids=seen_doc_ids,
                    embedding_model=embedding_model,
                    next_sequence=len(snapshots) + 1,
                )

            items = self.registry.list_run_items(run_id)
            failed_items = [item for item in items if item["action"] == INGEST_ACTION_FAILED]
            error_code = failed_items[0]["error_code"] if failed_items else ""
            error = f"{len(failed_items)} 个文件处理失败" if failed_items else ""
            finished_at = _format_time(self._now())
            self.registry.finish_owned_run(
                run_id=run_id,
                worker_id=worker_id,
                status=INGEST_RUN_STATUS_FAILED if failed_items else INGEST_RUN_STATUS_SUCCESS,
                finished_at=finished_at,
                error_code=error_code,
                error=error,
            )
        except IngestWorkerLeaseLostError:
            pass
        except Exception:
            finished_at = _format_time(self._now())
            self.registry.finish_owned_run(
                run_id=run_id,
                worker_id=worker_id,
                status=INGEST_RUN_STATUS_FAILED,
                finished_at=finished_at,
                error_code=INGEST_ERROR_DOCUMENT_PROCESSING_FAILED,
                error="入库任务执行失败",
            )
        finally:
            stop_heartbeat.set()
            heartbeat_thread.join(timeout=1)

        summary = self.registry.summarize_run(run_id)
        if summary is None:
            raise IngestRunNotFoundError(run_id)
        return summary

    def get_run(self, run_id: str) -> IngestionRunSummary:
        self.registry.expire_stale_runs(_format_time(self._now()))
        summary = self.registry.summarize_run(run_id)
        if summary is None:
            raise IngestRunNotFoundError(run_id)
        return summary

    def list_run_items(self, run_id: str) -> list[IngestionRunItemView]:
        if self.registry.get_run(run_id) is None:
            raise IngestRunNotFoundError(run_id)
        return self.registry.list_run_items(run_id)

    def list_recent_runs(self, limit: int) -> list[IngestionRunSummary]:
        self.registry.expire_stale_runs(_format_time(self._now()))
        summaries: list[IngestionRunSummary] = []
        for run in self.registry.list_recent_runs(limit):
            summary = self.registry.summarize_run(run.run_id)
            if summary is not None:
                summaries.append(summary)
        return summaries


__all__ = [
    "ActiveIngestRunError",
    "CategoryPreflightError",
    "IngestRunNotFoundError",
    "IngestionService",
    "UnknownIngestionCategoryError",
    "UnsafeIngestionPathError",
]
