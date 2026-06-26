from importlib import metadata
from pathlib import Path
from typing import List
from httpx import get
from langchain_core import documents
import pandas as pd
from langchain_unstructured import UnstructuredLoader
from langchain_core.documents import Document

def load_pdf(file_path: Path) -> List[Document]:
    """加载 PDF 文档，优先使用 UnstructuredLoader，失败时回退到 pypdf"""
    try:
        loader = UnstructuredLoader(file_path=str(file_path))
        print(f'loader: {loader}')
        return loader.load()
    except Exception as e:
        print(f'error: {e}')
        return []

def load_word(file_path: Path) -> List[Document]:
    """加载 Word 文档"""
    loader = UnstructuredLoader(file_path=str(file_path))
    return loader.load()

def load_directory(directory: Path) -> List[Document]:
    """批量加载目录下的文档"""
    documents = []

    for file_path in directory.rglob("*.pdf"):
        documents.extend(load_pdf(file_path))

    for file_path in directory.rglob("*.docx"):
        documents.extend(load_word(file_path))

    for file_path in directory.rglob("*.doc"):
        documents.extend(load_word(file_path))

    return documents

def load_announcement(url: str) -> List[Document]:
    """加载公告文档 HTML"""
    loader = UnstructuredLoader(web_url=url)
    return loader.load()


def load_financial_csv(file_path: Path) -> List[Document]:
    """加载财务数据 CSV"""
    df = pd.read_csv(file_path)
    documents = []
    for _, row in df.iterrows():
        content = "\n".join([f"{col}: {val}" for col, val in row.items()])
        documents.append(Document(
            page_content=content,
            metadata={
                "doc_type":"financial_data",
                "source": str(file_path),
                "stock_code": row.get("code",""),
                "year": row.get("year",""),
                "net_profit" : row.get("net_profit", ""),
                "total_assets": row.get("total_assets", "")
            }
        ))
    return documents

if __name__ == "__main__":
    # print(f"Path(__file__): {Path(__file__)}") #/Users/ryan/Desktop/SecRag/src/ingestion/loaders.py
    # print(f"Path(__file__).parent: {Path(__file__).parent}") #/Users/ryan/Desktop/SecRag/src/ingestion
    # print(f"Path(__file__).parent.parent: {Path(__file__).parent.parent}") #/Users/ryan/Desktop/SecRag/src
    # for parent in Path(__file__).parents:
    #     print(f"parent: {parent}")
    
    # documents = load_pdf(Path(__file__).parent.parent / "data/announcements/local-source-repos.pdf")
    # for document in documents:
    #     print(f"document: {document}")

    documents = load_financial_csv(Path(__file__).parent.parent / "data/announcements/sample-financial.csv")
    print(f"共加载 {len(documents)} 条财务数据：")
    for i, doc in enumerate(documents, 1):
        print(f"\n--- 文档 {i} ---")
        print(f"  内容:\n{doc.page_content}")
        print(f"  元数据: {doc.metadata}")
