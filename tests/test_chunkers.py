from langchain_core.documents import Document

from src.ingestion.chunkers import create_financial_splitter, chunk_documents
from src.schemas.constants import (
    DOC_TYPE_ANNOUNCEMENT,
    DOC_TYPE_FINANCIAL_DATA,
    DOC_TYPE_RESEARCH_REPORT,
)


def _sample_document(
    content: str = "这是一段测试文本。这是第二句。", metadata: dict | None = None
) -> Document:
    return Document(page_content=content, metadata=metadata or {"source": "test"})


# --- create_financial_splitter ---


def test_create_financial_splitter_returns_splitter():
    splitter = create_financial_splitter()
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    assert isinstance(splitter, RecursiveCharacterTextSplitter)


def test_create_financial_splitter_default_params():
    splitter = create_financial_splitter()
    assert splitter._chunk_size == 500
    assert splitter._chunk_overlap == 100


def test_create_financial_splitter_custom_params():
    splitter = create_financial_splitter(chunk_size=300, chunk_overlap=50)
    assert splitter._chunk_size == 300
    assert splitter._chunk_overlap == 50


def test_create_financial_splitter_chinese_aware():
    splitter = create_financial_splitter(chunk_size=30, chunk_overlap=5)
    doc = _sample_document(
        "第一句很长很长很长。第二句也很长很长很长。第三句同样很长。第四句也很长。第五句结束。"
    )
    chunks = splitter.split_documents([doc])
    assert len(chunks) > 1


# --- chunk_documents ---


def test_chunk_documents_research_report():
    doc = _sample_document("研究内容。")
    chunks = chunk_documents([doc], DOC_TYPE_RESEARCH_REPORT)
    assert len(chunks) >= 1
    assert all(c.page_content for c in chunks)


def test_chunk_documents_announcement_smaller_chunks():
    content = "公告正文。段落二。段落三。段落四。段落五。段落六。"
    doc = _sample_document(content)
    chunks = chunk_documents([doc], DOC_TYPE_ANNOUNCEMENT)
    assert len(chunks) >= 1


def test_chunk_documents_financial_report_larger_chunks():
    content = "财务报告" * 200
    doc = _sample_document(content)
    chunks_ann = chunk_documents([doc], DOC_TYPE_ANNOUNCEMENT)
    chunks_fin = chunk_documents([doc], DOC_TYPE_FINANCIAL_DATA)
    assert len(chunks_fin) < len(chunks_ann)


def test_chunk_documents_preserves_metadata():
    meta = {"source": "annual_report.pdf", "year": "2024"}
    doc = _sample_document("年报内容。", metadata=meta)
    chunks = chunk_documents([doc], DOC_TYPE_FINANCIAL_DATA)
    for chunk in chunks:
        assert chunk.metadata["source"] == "annual_report.pdf"
        assert chunk.metadata["year"] == "2024"


def test_chunk_documents_unknown_type_fallback():
    doc = _sample_document("未知类型。")
    chunks = chunk_documents([doc], "nonexistent_type")
    assert len(chunks) >= 1
