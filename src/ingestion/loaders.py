from pathlib import Path
from typing import List

from langchain_unstructured import UnstructuredLoader
from langchain_core.documents import Document

'''
  load_pdf 踩坑记录

  1. FileNotFoundError: 相对路径 data/announcements 找不到文件。
    - 原因：脚本 CWD 是项目根目录，数据文件在 app/ingestion/data/announcements/ 下。
    - 解决：调用方用 Path(__file__).parent / "data/..." 定位。
  2. ImportError: partition_pdf() is not available。
    - 原因：unstructured 包缺少 [pdf] extras（pikepdf、pdf2image 等）。
    - 解决：uv pip install "unstructured[pdf]"。
    - 网络不通时：系统代理 127.0.0.1:16780 未运行，uv 会报 Connection refused。
    - 启动代理服务后重试，或 NO_PROXY="*" uv pip install ... 绕过。
  3. UnstructuredLoader 的 metadata 无 page 字段，fallback (pypdf) 有 page。
    - __main__ 中打印 metadata.get('page') 在原生模式下返回 None。
'''
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

def load_directory(directory: Path, pattern: str = "**/*.{pdf,docx}") -> List[Document]:
    """加载目录下所有文档"""
    documents = []
    for file_path in directory.glob(pattern):
        if file_path.suffix.lower() == ".pdf":
            documents.extend(load_pdf(file_path))
        elif file_path.suffix.lower() in [".docx", ".doc"]:
            documents.extend(load_word(file_path))
    return documents

def load_announcement(url: str) -> List[Document]:
    """加载公告文档"""
    loader = UnstructuredLoader(web_url=url)
    return loader.load()

if __name__ == "__main__":
    print(f"Path(__file__): {Path(__file__)}") #/Users/ryan/Desktop/SecRag/src/ingestion/loaders.py
    print(f"Path(__file__).parent: {Path(__file__).parent}") #/Users/ryan/Desktop/SecRag/src/ingestion
    print(f"Path(__file__).parent.parent: {Path(__file__).parent.parent}") #/Users/ryan/Desktop/SecRag/src
    for parent in Path(__file__).parents:
        print(f"parent: {parent}")
    
    documents = load_pdf(Path(__file__).parent.parent / "data/announcements/local-source-repos.pdf")
    for document in documents:
        print(f"document: {document}")