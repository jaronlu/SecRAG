from pathlib import Path

from langchain_core.documents import Document

from scripts.ingest import _load_sample_metadata, normalize_chunks
from src.schemas.constants import (
    DOC_TYPE_ANNOUNCEMENT,
    DOC_TYPE_FINANCIAL_DATA,
    DOC_TYPE_FAQ,
    DOC_TYPE_RESEARCH_REPORT,
    META_ALLOWED_ROLES,
    META_DOC_TYPE,
    META_PERMISSION_LEVEL,
    META_RETRIEVAL_SOURCE,
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
