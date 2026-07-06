from pathlib import Path

from langchain_core.documents import Document

from scripts.ingest import _load_sample_metadata, normalize_chunks
from src.schemas.constants import (
    DOC_TYPE_FINANCIAL_DATA,
    DOC_TYPE_FAQ,
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
)


def test_loads_sample_metadata_manifest():
    metadata = _load_sample_metadata(Path("src/data/samples/faq/sample_project_technical_faq.html"))

    assert metadata[META_DOC_TYPE] == DOC_TYPE_FAQ
    assert metadata[META_RETRIEVAL_SOURCE] == SOURCE_FAQ
    assert metadata[META_PERMISSION_LEVEL] == "internal"
    assert metadata[META_ALLOWED_ROLES] == ["technical"]


def test_loads_announcements_metadata_manifest():
    metadata = _load_sample_metadata(Path("src/data/announcements/local-source-repos.pdf"))

    assert metadata[META_DOC_TYPE] == DOC_TYPE_FAQ
    assert metadata[META_RETRIEVAL_SOURCE] == SOURCE_FAQ
    assert metadata[META_PERMISSION_LEVEL] == "internal"
    assert metadata[META_ALLOWED_ROLES] == ["technical"]


def test_loads_financial_csv_metadata_manifest():
    metadata = _load_sample_metadata(Path("src/data/announcements/sample-financial.csv"))

    assert metadata[META_DOC_TYPE] == DOC_TYPE_FINANCIAL_DATA
    assert metadata[META_RETRIEVAL_SOURCE] == "sql_query"
    assert metadata[META_PERMISSION_LEVEL] == "internal"
    assert metadata[META_ALLOWED_ROLES] == ["advisor", "institutional_sales", "compliance"]


def test_normalize_chunks_embeds_permission_metadata():
    file_path = Path("src/data/samples/faq/sample_project_technical_faq.html")
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
        Path("src/data/samples/metadata.json"),
        Path("src/data/announcements/metadata.json"),
    ]

    for manifest in manifests:
        entries = json.loads(manifest.read_text(encoding="utf-8"))
        for metadata in entries.values():
            assert set(metadata[META_ALLOWED_ROLES]) <= valid_roles
