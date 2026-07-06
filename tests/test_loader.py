from importlib.resources import files
from pathlib import Path

from src.ingestion.loaders import load_pdf, load_word


def test_load_pdf():
    pdf_path = files("src.data.announcements") / "local-source-repos.pdf"
    documents = load_pdf(Path(str(pdf_path)))

    assert len(documents) > 0
    assert all(document.page_content for document in documents)
    assert all(isinstance(document.metadata, dict) for document in documents)


def test_word():
    word_path = Path(__file__).parent.parent / "src/data/announcements/local-word.docx"
    documents = load_word(Path(str(word_path)))

    assert len(documents) > 0
    assert all(document.page_content for document in documents)
