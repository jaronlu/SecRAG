from pathlib import Path
from importlib.resources import files
from src.ingestion.loaders import load_pdf

# 测试 load_pdf
def test_load_pdf():
    pdf_path = files("src.data.announcements") / "local-source-repos.pdf"
    print(f"pdf_path: {pdf_path}")
    documents = load_pdf(Path(str(pdf_path)))
    assert len(documents) > 0
    for document in documents:
        print(f"document: {document}")
        print(f"document.page_content: {document.page_content}")
        print(f"document.metadata: {document.metadata}")
        print("-" * 100)

if __name__ == "__main__":
    test_load_pdf()