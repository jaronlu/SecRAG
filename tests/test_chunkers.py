import statistics
from pathlib import Path

from langchain_core.documents import Document

from src.ingestion.chunkers import (
    aggregate_document_elements,
    chunk_documents,
    create_financial_splitter,
)
from src.schemas.constants import (
    DOC_TYPE_ANNOUNCEMENT,
    DOC_TYPE_FINANCIAL_DATA,
    DOC_TYPE_RESEARCH_REPORT,
    META_PAGE_NUMBER,
)


def _sample_document(
    content: str = "这是一段测试文本。这是第二句。", metadata: dict | None = None
) -> Document:
    return Document(page_content=content, metadata=metadata or {"source": "test"})


def _element(text: str, category: str, page: int | None = None) -> Document:
    """构造一个 UnstructuredLoader 风格的解析元素（一个元素一个 Document）。"""
    metadata: dict = {"category": category}
    if page is not None:
        metadata[META_PAGE_NUMBER] = page
    return Document(page_content=text, metadata=metadata)


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


# --- ISSUE-21: 解析元素聚合 / 设计分块尺寸 ---


def test_aggregate_document_elements_drops_header_footer_and_email():
    documents = [
        _element("宁德时代新能源科技股份有限公司 2026 年半年度报告全文", "Header", 1),
        _element("公司上半年实现营业收入 90,703,260,964.48 元。", "NarrativeText", 1),
        _element("ir@catl.com", "EmailAddress", 1),
        _element("第 1 页 共 200 页", "Footer", 1),
    ]

    aggregated = aggregate_document_elements(documents)

    assert len(aggregated) == 1
    content = aggregated[0].page_content
    assert "90,703,260,964.48" in content
    assert "半年度报告全文" not in content
    assert "ir@catl.com" not in content
    assert "第 1 页" not in content


def test_aggregate_document_elements_merges_fragments_in_order():
    documents = [_element(f"碎片{i}", "UncategorizedText") for i in range(5)]

    aggregated = aggregate_document_elements(documents)

    assert len(aggregated) == 1
    assert aggregated[0].page_content == "碎片0\n\n碎片1\n\n碎片2\n\n碎片3\n\n碎片4"


def test_aggregate_document_elements_keeps_table_blocks_separate():
    table = "<table><tr><td>营业收入</td><td>90,703,260,964.48</td></tr></table>"
    documents = [
        _element("主要会计数据", "Title"),
        _element(table, "Table"),
        _element("营业收入同比减少。", "NarrativeText"),
    ]

    aggregated = aggregate_document_elements(documents)

    assert [doc.metadata["block_type"] for doc in aggregated] == ["text", "table", "text"]
    assert aggregated[1].page_content == table


def test_aggregate_document_elements_passes_through_uncategorized_documents():
    documents = [
        Document(page_content="code: 600519\nyear: 2026", metadata={"source": "x.csv"})
    ]

    assert aggregate_document_elements(documents) == documents


def test_chunk_documents_binds_design_chunk_size_to_element_fragments():
    """ISSUE-21 根因：碎片元素直接送切分器时 chunk_size 永不生效。"""
    documents = [
        _element(f"指标{i}：数值 {i},000.00，同比增长 {i}.3%。", "UncategorizedText", 1)
        for i in range(40)
    ]

    chunks = chunk_documents(documents, DOC_TYPE_FINANCIAL_DATA)

    lengths = [len(chunk.page_content) for chunk in chunks]
    assert max(lengths) <= 800
    assert max(lengths) >= 600
    assert len(chunks) < 10


def test_chunk_documents_keeps_table_header_with_split_rows():
    rows = "".join(f"<tr><td>指标{i}</td><td>{i},000.00</td></tr>" for i in range(200))
    table = f"<table><tr><td>项目</td><td>金额</td></tr>{rows}</table>"

    chunks = chunk_documents([_element(table, "Table")], DOC_TYPE_FINANCIAL_DATA)

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.page_content) <= 800
        assert "<tr><td>项目</td><td>金额</td></tr>" in chunk.page_content


def test_chunk_documents_drops_identical_repeated_blocks():
    table = (
        "<table><tr><td>项目</td><td>本期金额</td></tr>"
        "<tr><td>营业收入</td><td>90,703,260,964.48</td></tr>"
        "<tr><td>归母净利润</td><td>44,517,000,000.00</td></tr></table>"
    )
    documents = [_element(table, "Table", 1), _element(table, "Table", 2)]

    chunks = chunk_documents(documents, DOC_TYPE_FINANCIAL_DATA)

    assert len(chunks) == 1


def test_chunk_documents_assigns_page_number_from_element_offset():
    documents = [
        _element(f"第{page}页正文内容。" * 75, "NarrativeText", page) for page in (1, 2, 3)
    ]

    chunks = chunk_documents(documents, DOC_TYPE_RESEARCH_REPORT)

    pages = [chunk.metadata.get(META_PAGE_NUMBER) for chunk in chunks]
    assert pages[0] == 1
    assert max(pages) == 3
    assert pages == sorted(pages)


def test_chunk_documents_leaves_uncategorized_documents_unsplit_unchanged():
    document = Document(
        page_content="code: 600519\nyear: 2026\nrevenue: 907.03",
        metadata={"source": "x.csv"},
    )

    chunks = chunk_documents([document], DOC_TYPE_FINANCIAL_DATA)

    assert [chunk.page_content for chunk in chunks] == [document.page_content]


def test_real_report_chunk_length_distribution_meets_design():
    """ISSUE-21 验收：真实研报分块长度落在设计区间（500/100）。"""
    from src.ingestion.loaders import load_pdf

    report = Path("data/raw/real_securities_data/reports/600519_.pdf")
    documents = load_pdf(report)

    chunks = chunk_documents(documents, DOC_TYPE_RESEARCH_REPORT)

    lengths = [len(chunk.page_content) for chunk in chunks]
    short_share = sum(1 for length in lengths if length <= 30) / len(lengths)
    assert max(lengths) <= 500
    assert statistics.median(lengths) >= 200
    assert short_share < 0.1
