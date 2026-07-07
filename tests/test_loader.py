from pathlib import Path

from src.ingestion.loaders import load_pdf, load_word


def test_load_pdf():
    pdf_path = Path(__file__).parent.parent / "data/raw/demo_knowledge_base/announcements/local-source-repos.pdf"
    documents = load_pdf(pdf_path)

    assert len(documents) > 0
    assert all(document.page_content for document in documents)
    assert all(isinstance(document.metadata, dict) for document in documents)


def test_word():
    word_path = Path(__file__).parent.parent / "data/raw/demo_knowledge_base/announcements/local-word.docx"
    documents = load_word(word_path)

    assert len(documents) > 0
    assert all(document.page_content for document in documents)
