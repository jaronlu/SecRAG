from pathlib import Path

from langchain_core.documents import Document

from scripts import ingest
from scripts.ingest import (
    _load_sample_metadata,
    derive_doc_id,
    ingest_document,
    normalize_chunks,
)
from src.ingestion.registry import DocumentRegistryStore
from src.schemas.constants import (
    DOC_TYPE_ANNOUNCEMENT,
    DOC_TYPE_FINANCIAL_DATA,
    DOC_TYPE_FAQ,
    DOC_TYPE_RESEARCH_REPORT,
    META_ALLOWED_ROLES,
    META_DOC_ID,
    META_DOC_TYPE,
    META_PERMISSION_LEVEL,
    META_RETRIEVAL_SOURCE,
    META_TITLE,
    ROLE_ADVISOR,
    ROLE_COMPLIANCE,
    ROLE_INSTITUTIONAL_SALES,
    ROLE_OPERATIONS,
    ROLE_TECHNICAL,
    SOURCE_FAQ,
    SOURCE_REPORT,
)


def test_loads_sample_metadata_manifest():
    metadata = _load_sample_metadata(
        Path("data/raw/demo_knowledge_base/samples/faq/sample_project_technical_faq.html")
    )

    assert metadata[META_DOC_TYPE] == DOC_TYPE_FAQ
    assert metadata[META_RETRIEVAL_SOURCE] == SOURCE_FAQ
    assert metadata[META_PERMISSION_LEVEL] == "internal"
    assert metadata[META_ALLOWED_ROLES] == ["technical"]


def test_loads_announcements_metadata_manifest():
    metadata = _load_sample_metadata(
        Path("data/raw/demo_knowledge_base/announcements/local-source-repos.pdf")
    )

    assert metadata[META_DOC_TYPE] == DOC_TYPE_FAQ
    assert metadata[META_RETRIEVAL_SOURCE] == SOURCE_FAQ
    assert metadata[META_PERMISSION_LEVEL] == "internal"
    assert metadata[META_ALLOWED_ROLES] == ["technical"]


def test_loads_financial_csv_metadata_manifest():
    metadata = _load_sample_metadata(
        Path("data/raw/demo_knowledge_base/announcements/sample-financial.csv")
    )

    assert metadata[META_DOC_TYPE] == DOC_TYPE_FINANCIAL_DATA
    assert metadata[META_RETRIEVAL_SOURCE] == "sql_query"
    assert metadata[META_PERMISSION_LEVEL] == "internal"
    assert metadata[META_ALLOWED_ROLES] == ["advisor", "institutional_sales", "compliance"]


def test_loads_real_securities_data_metadata_manifest():
    metadata = _load_sample_metadata(
        Path("data/raw/real_securities_data/reports/000001_2025_2026.pdf")
    )

    assert metadata[META_DOC_TYPE] == DOC_TYPE_RESEARCH_REPORT
    assert metadata[META_RETRIEVAL_SOURCE] == SOURCE_REPORT
    assert metadata[META_PERMISSION_LEVEL] == "internal"
    assert metadata[META_ALLOWED_ROLES] == ["advisor", "institutional_sales", "compliance"]


def test_real_securities_data_files_exist():
    data_root = Path("data/raw/real_securities_data")
    annual_reports = sorted((data_root / "announcements").glob("*.pdf"))
    research_reports = sorted((data_root / "reports").glob("*.pdf"))
    csv_files = sorted((data_root / "financials").glob("*.csv"))

    assert len(annual_reports) >= 2
    assert len(research_reports) >= 2
    assert len(csv_files) >= 2

    for pdf_path in [*annual_reports, *research_reports]:
        assert pdf_path.read_bytes().startswith(b"%PDF-")

    for csv_path in csv_files:
        assert len(csv_path.read_text(encoding="utf-8").splitlines()) >= 2


def test_normalize_chunks_embeds_permission_metadata():
    file_path = Path("data/raw/demo_knowledge_base/samples/faq/sample_project_technical_faq.html")
    chunks = [Document(page_content="LangGraph test", metadata={})]

    normalized = normalize_chunks(
        chunks=chunks,
        file_path=file_path,
        doc_type=DOC_TYPE_FAQ,
        sample_metadata=_load_sample_metadata(file_path),
    )

    metadata = normalized[0].metadata
    assert metadata[META_DOC_TYPE] == DOC_TYPE_FAQ
    assert metadata[META_RETRIEVAL_SOURCE] == SOURCE_FAQ
    assert metadata[META_PERMISSION_LEVEL] == "internal"
    assert metadata[META_ALLOWED_ROLES] == "technical"


def test_derives_real_data_doc_ids_from_stable_sources():
    annual_metadata = _load_sample_metadata(
        Path("data/raw/real_securities_data/announcements/000001_2025.pdf")
    )
    research_metadata = _load_sample_metadata(
        Path("data/raw/real_securities_data/reports/000001_2025_2026.pdf")
    )
    csv_metadata = _load_sample_metadata(
        Path("data/raw/real_securities_data/financials/baostock_600519_valuation_202501.csv")
    )
    efinance_metadata = _load_sample_metadata(
        Path("data/raw/real_securities_data/financials/efinance_600519_base_info.csv")
    )

    assert (
        derive_doc_id(
            Path("data/raw/real_securities_data/announcements/000001_2025.pdf"),
            DOC_TYPE_ANNOUNCEMENT,
            annual_metadata,
        )
        == "cninfo:announcement:1225022887"
    )
    assert (
        derive_doc_id(
            Path("data/raw/real_securities_data/reports/000001_2025_2026.pdf"),
            DOC_TYPE_RESEARCH_REPORT,
            research_metadata,
        )
        == "eastmoney:research:AP202604261821586763"
    )
    assert (
        derive_doc_id(
            Path("data/raw/real_securities_data/financials/baostock_600519_valuation_202501.csv"),
            DOC_TYPE_FINANCIAL_DATA,
            csv_metadata,
        )
        == "dataset:baostock:query_history_k_data_plus:sh.600519:2025-01-02:2025-01-27"
    )
    assert (
        derive_doc_id(
            Path("data/raw/real_securities_data/financials/efinance_600519_base_info.csv"),
            DOC_TYPE_FINANCIAL_DATA,
            efinance_metadata,
        )
        == "dataset:efinance:eastmoney:get_base_info:600519"
    )


def test_manifest_doc_id_survives_file_rename(tmp_path):
    old_path = tmp_path / "old-name.csv"
    new_path = tmp_path / "new-name.csv"
    old_path.write_text("code,year\n600519,2026\n", encoding="utf-8")
    new_path.write_text("code,year\n600519,2026\n", encoding="utf-8")
    metadata = {META_DOC_ID: "dataset:manual:stable-id"}

    assert derive_doc_id(old_path, DOC_TYPE_FINANCIAL_DATA, metadata) == derive_doc_id(
        new_path,
        DOC_TYPE_FINANCIAL_DATA,
        metadata,
    )


def test_normalize_chunks_overwrites_loader_path_doc_id():
    file_path = Path("data/raw/real_securities_data/financials/efinance_600519_base_info.csv")
    chunks = [Document(page_content="content", metadata={META_DOC_ID: "path-derived"})]

    normalized = normalize_chunks(
        chunks=chunks,
        file_path=file_path,
        doc_type=DOC_TYPE_FINANCIAL_DATA,
        sample_metadata=_load_sample_metadata(file_path),
        doc_id="dataset:efinance:eastmoney:get_base_info:600519",
    )

    assert normalized[0].metadata[META_DOC_ID] == "dataset:efinance:eastmoney:get_base_info:600519"


def test_ingest_document_skips_unchanged_file(monkeypatch, tmp_path):
    file_path = tmp_path / "stable.csv"
    file_path.write_text("code,year\n600519,2026\n", encoding="utf-8")
    registry = DocumentRegistryStore(tmp_path / "registry.db")
    embedding_model = object()
    run_id = "run-1"
    metadata = {
        META_DOC_TYPE: DOC_TYPE_FINANCIAL_DATA,
        META_DOC_ID: "dataset:manual:stable-id",
        META_TITLE: "stable",
    }

    monkeypatch.setattr(ingest, "_load_sample_metadata", lambda _: metadata)
    monkeypatch.setattr(ingest, "upsert_chunks", lambda **_: None)
    monkeypatch.setattr(ingest, "list_chunk_ids_by_doc_id", lambda **_: [])
    monkeypatch.setattr(ingest, "delete_chunk_ids", lambda **_: None)
    monkeypatch.setattr(
        ingest,
        "_load_documents",
        lambda _: [Document(page_content="code: 600519", metadata={})],
    )
    registry.start_run(run_id, tmp_path.as_uri(), full_scan=False, started_at="2026-07-07T00:00:00Z")

    first_action = ingest_document(
        file_path,
        DOC_TYPE_FINANCIAL_DATA,
        registry_store=registry,
        run_id=run_id,
        root_dir=tmp_path,
        embedding_model=embedding_model,
        persist_directory=str(tmp_path / "chroma"),
    )
    second_action = ingest_document(
        file_path,
        DOC_TYPE_FINANCIAL_DATA,
        registry_store=registry,
        run_id=run_id,
        root_dir=tmp_path,
        embedding_model=embedding_model,
        persist_directory=str(tmp_path / "chroma"),
    )

    assert first_action == "created"
    assert second_action == "skipped"


def test_ingest_document_deletes_stale_chunks_after_upsert(monkeypatch, tmp_path):
    file_path = tmp_path / "stable.csv"
    file_path.write_text("code,year\n600519,2026\n", encoding="utf-8")
    registry = DocumentRegistryStore(tmp_path / "registry.db")
    embedding_model = object()
    run_id = "run-1"
    metadata = {
        META_DOC_TYPE: DOC_TYPE_FINANCIAL_DATA,
        META_DOC_ID: "dataset:manual:stable-id",
        META_TITLE: "stable",
    }
    calls: list[tuple[str, list[str]]] = []

    monkeypatch.setattr(ingest, "_load_sample_metadata", lambda _: metadata)
    monkeypatch.setattr(ingest, "list_chunk_ids_by_doc_id", lambda **_: ["old-1", "old-2"])

    def fake_upsert_chunks(**kwargs):
        calls.append(("upsert", [str(chunk.id) for chunk in kwargs["chunks"]]))

    def fake_delete_chunk_ids(**kwargs):
        calls.append(("delete", sorted(kwargs["chunk_ids"])))

    monkeypatch.setattr(ingest, "upsert_chunks", fake_upsert_chunks)
    monkeypatch.setattr(ingest, "delete_chunk_ids", fake_delete_chunk_ids)
    monkeypatch.setattr(
        ingest,
        "_load_documents",
        lambda _: [Document(page_content="new content", metadata={})],
    )
    registry.start_run(run_id, tmp_path.as_uri(), full_scan=False, started_at="2026-07-07T00:00:00Z")

    action = ingest_document(
        file_path,
        DOC_TYPE_FINANCIAL_DATA,
        registry_store=registry,
        run_id=run_id,
        root_dir=tmp_path,
        embedding_model=embedding_model,
        persist_directory=str(tmp_path / "chroma"),
    )

    assert action == "created"
    assert calls[0][0] == "upsert"
    assert calls[1] == ("delete", ["old-1", "old-2"])


def test_ingest_document_failure_keeps_existing_chunks(monkeypatch, tmp_path):
    file_path = tmp_path / "stable.csv"
    file_path.write_text("code,year\n600519,2026\n", encoding="utf-8")
    registry = DocumentRegistryStore(tmp_path / "registry.db")
    embedding_model = object()
    run_id = "run-1"
    metadata = {
        META_DOC_TYPE: DOC_TYPE_FINANCIAL_DATA,
        META_DOC_ID: "dataset:manual:stable-id",
        META_TITLE: "stable",
    }
    calls: list[str] = []

    monkeypatch.setattr(ingest, "_load_sample_metadata", lambda _: metadata)
    monkeypatch.setattr(ingest, "_load_documents", lambda _: [])
    monkeypatch.setattr(ingest, "list_chunk_ids_by_doc_id", lambda **_: ["old-1"])
    monkeypatch.setattr(ingest, "upsert_chunks", lambda **_: calls.append("upsert"))
    monkeypatch.setattr(ingest, "delete_chunk_ids", lambda **_: calls.append("delete"))
    registry.start_run(run_id, tmp_path.as_uri(), full_scan=False, started_at="2026-07-07T00:00:00Z")

    action = ingest_document(
        file_path,
        DOC_TYPE_FINANCIAL_DATA,
        registry_store=registry,
        run_id=run_id,
        root_dir=tmp_path,
        embedding_model=embedding_model,
        persist_directory=str(tmp_path / "chroma"),
    )

    record = registry.get_document("dataset:manual:stable-id")
    assert action == "failed"
    assert calls == []
    assert record is not None
    assert record.status == "failed"
    assert record.error == "文档解析结果为空"


def test_sample_allowed_roles_match_role_constants():
    import json

    valid_roles = {
        ROLE_ADVISOR,
        ROLE_INSTITUTIONAL_SALES,
        ROLE_COMPLIANCE,
        ROLE_OPERATIONS,
        ROLE_TECHNICAL,
    }
    manifests = [
        Path("data/raw/demo_knowledge_base/samples/metadata.json"),
        Path("data/raw/demo_knowledge_base/announcements/metadata.json"),
        Path("data/raw/real_securities_data/metadata.json"),
    ]

    for manifest in manifests:
        entries = json.loads(manifest.read_text(encoding="utf-8"))
        for metadata in entries.values():
            assert set(metadata[META_ALLOWED_ROLES]) <= valid_roles


def test_real_data_manifest_doc_types():
    import json

    entries = json.loads(
        Path("data/raw/real_securities_data/metadata.json").read_text(encoding="utf-8")
    )
    doc_types = {metadata[META_DOC_TYPE] for metadata in entries.values()}

    assert DOC_TYPE_ANNOUNCEMENT in doc_types
    assert DOC_TYPE_RESEARCH_REPORT in doc_types
    assert DOC_TYPE_FINANCIAL_DATA in doc_types
