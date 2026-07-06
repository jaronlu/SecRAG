from pathlib import Path

from langchain_core.documents import Document

from scripts.ingest import _load_sample_metadata, normalize_chunks
from src.schemas.constants import (
    DOC_TYPE_FAQ,
    META_ALLOWED_ROLES,
    META_DOC_TYPE,
    META_PERMISSION_LEVEL,
    META_RETRIEVAL_SOURCE,
    SOURCE_FAQ,
)


def test_loads_sample_metadata_manifest():
    metadata = _load_sample_metadata(Path("src/data/samples/faq/sample_project_technical_faq.html"))

    assert metadata[META_DOC_TYPE] == DOC_TYPE_FAQ
    assert metadata[META_RETRIEVAL_SOURCE] == SOURCE_FAQ
    assert metadata[META_PERMISSION_LEVEL] == "internal"
    assert metadata[META_ALLOWED_ROLES] == ["technical"]


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
