import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import pytest
from langchain_huggingface import HuggingFaceEmbeddings

from src.ingestion import service as service_module
from src.ingestion.registry import (
    INGEST_ACTION_CREATED,
    ActiveIngestRunError,
    DocumentRegistryStore,
)
from src.ingestion.service import IngestionService
from src.schemas.constants import (
    INGEST_ERROR_EMBEDDING_UNAVAILABLE,
    INGEST_ERROR_SOURCE_CHANGED_AFTER_ENQUEUE,
    INGEST_RUN_STATUS_FAILED,
    INGEST_RUN_STATUS_SUCCESS,
)
from src.schemas.typed_dicts import IngestionCategoryConfig


def _fake_embedding() -> HuggingFaceEmbeddings:
    return cast(HuggingFaceEmbeddings, object())


def _catalog() -> tuple[IngestionCategoryConfig, ...]:
    return (
        {
            "category_id": "financials",
            "label": "财务数据",
            "group": "证券数据",
            "relative_path": "data/raw/financials",
            "default_doc_type": "financial_data",
            "allowed_doc_types": ["financial_data"],
        },
    )


def _write_source(project_root: Path, name: str = "sample.csv") -> Path:
    root = project_root / "data/raw/financials"
    root.mkdir(parents=True, exist_ok=True)
    file_path = root / name
    file_path.write_text("code,year\n600519,2026\n", encoding="utf-8")
    file_path.with_name(f"{file_path.name}.meta.json").write_text(
        json.dumps({
            "doc_id": f"dataset:manual:{name}",
            "doc_type": "financial_data",
            "permission_level": "internal",
            "allowed_roles": ["technical"],
        }),
        encoding="utf-8",
    )
    return file_path


def _service(tmp_path: Path, **kwargs) -> IngestionService:
    return IngestionService(
        project_root=tmp_path,
        registry_path=tmp_path / "registry.db",
        persist_directory=str(tmp_path / "chroma"),
        catalog=_catalog(),
        embedding_model_factory=lambda _: _fake_embedding(),
        **kwargs,
    )


def test_registry_migrates_current_schema_without_rebuild(tmp_path):
    db_path = tmp_path / "registry.db"
    with sqlite3.connect(db_path) as connection:
        connection.executescript(
            """
            CREATE TABLE ingest_runs (
                run_id TEXT PRIMARY KEY,
                root_uri TEXT NOT NULL,
                mode TEXT NOT NULL DEFAULT 'incremental',
                full_scan INTEGER NOT NULL DEFAULT 0,
                started_at TEXT NOT NULL,
                finished_at TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'running',
                error TEXT NOT NULL DEFAULT ''
            );
            CREATE TABLE ingest_run_items (
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
            INSERT INTO ingest_runs (run_id, root_uri, started_at, status)
            VALUES ('legacy', 'file:///tmp', '2026-07-01T00:00:00+00:00', 'success');
            """
        )

    DocumentRegistryStore(db_path)
    DocumentRegistryStore(db_path)

    with sqlite3.connect(db_path) as connection:
        run_columns = {row[1] for row in connection.execute("PRAGMA table_info(ingest_runs)")}
        item_columns = {row[1] for row in connection.execute("PRAGMA table_info(ingest_run_items)")}
        queued_at = connection.execute(
            "SELECT queued_at FROM ingest_runs WHERE run_id = 'legacy'"
        ).fetchone()[0]
    assert {
        "category_id",
        "queued_at",
        "worker_id",
        "lease_expires_at",
        "error_code",
    } <= run_columns
    assert {"sequence", "relative_path", "processed_at", "error_code"} <= item_columns
    assert queued_at == "2026-07-01T00:00:00+00:00"


def test_create_run_is_atomic_and_rejects_second_active_run(tmp_path):
    _write_source(tmp_path)
    service = _service(tmp_path)

    first = service.create_run("financials", requested_by="user_tech")

    assert first["status"] == "queued"
    assert first["started_at"] is None
    assert first["total_files"] == 1
    with pytest.raises(ActiveIngestRunError) as exc:
        service.create_run("financials", requested_by="user_tech")
    assert exc.value.run_id == first["run_id"]


def test_execute_run_processes_only_snapshot(monkeypatch, tmp_path):
    _write_source(tmp_path)
    service = _service(tmp_path)
    queued = service.create_run("financials", requested_by="user_tech")
    _write_source(tmp_path, "added-after-queue.csv")
    processed: list[str] = []

    def fake_ingest_document(**kwargs):
        processed.append(kwargs["relative_path_override"])
        kwargs["registry_store"].record_run_item(
            run_id=kwargs["run_id"],
            doc_id="dataset:manual:sample.csv",
            source_uri=kwargs["file_path"].as_uri(),
            action=INGEST_ACTION_CREATED,
            sequence=kwargs["sequence"],
            relative_path=kwargs["relative_path_override"],
            chunk_count=1,
        )
        return INGEST_ACTION_CREATED

    monkeypatch.setattr(service_module.pipeline, "ingest_document", fake_ingest_document)

    summary = service.execute_run(queued["run_id"])

    assert summary["status"] == INGEST_RUN_STATUS_SUCCESS
    assert summary["total_files"] == 1
    assert processed == ["data/raw/financials/sample.csv"]


def test_source_change_after_enqueue_fails_without_processing(monkeypatch, tmp_path):
    file_path = _write_source(tmp_path)
    service = _service(tmp_path)
    queued = service.create_run("financials", requested_by="user_tech")
    file_path.write_text("changed", encoding="utf-8")
    called = False

    def fake_ingest_document(**kwargs):
        nonlocal called
        called = True
        return INGEST_ACTION_CREATED

    monkeypatch.setattr(service_module.pipeline, "ingest_document", fake_ingest_document)

    summary = service.execute_run(queued["run_id"])

    assert called is False
    assert summary["status"] == INGEST_RUN_STATUS_FAILED
    assert summary["error_code"] == INGEST_ERROR_SOURCE_CHANGED_AFTER_ENQUEUE
    assert summary["failed"] == 1


def test_embedding_initialization_failure_is_asynchronous_run_failure(tmp_path):
    _write_source(tmp_path)

    def fail_embedding(_: str):
        raise RuntimeError("device unavailable")

    service = IngestionService(
        project_root=tmp_path,
        registry_path=tmp_path / "registry.db",
        persist_directory=str(tmp_path / "chroma"),
        catalog=_catalog(),
        embedding_model_factory=fail_embedding,
    )
    queued = service.create_run("financials", requested_by="user_tech")

    summary = service.execute_run(queued["run_id"])

    assert queued["status"] == "queued"
    assert summary["status"] == INGEST_RUN_STATUS_FAILED
    assert summary["error_code"] == INGEST_ERROR_EMBEDDING_UNAVAILABLE


def test_expired_lease_is_reclaimed_before_new_run(tmp_path):
    _write_source(tmp_path)
    current = datetime(2026, 7, 14, 8, 0, tzinfo=UTC)
    clock = {"now": current}
    service = _service(tmp_path, now_factory=lambda: clock["now"], lease_seconds=2)
    first = service.create_run("financials", requested_by="user_tech")
    clock["now"] = current + timedelta(seconds=3)

    second = service.create_run("financials", requested_by="user_tech")

    assert second["run_id"] != first["run_id"]
    assert service.get_run(first["run_id"])["error_code"] == "worker_lease_expired"
