"""完整入库流程"""

import hashlib
import sys
from pathlib import Path

from src.ingestion.chunkers import chunk_documents
from src.ingestion.embedder import embed_and_store, get_embedding_model
from src.schemas.constants import (
    META_CHUNK_ID,
    META_DOC_ID,
    META_DOC_TYPE,
    META_SOURCE,
    META_TITLE,
    META_DATE,
    META_PERMISSION_LEVEL,
    PERMISSION_INTERNAL,
    ALL_VALID_DOC_TYPES,
    CHROMA_DEFAULT_PERSIST_DIR,
)

SUPPORTED_SUFFIXES = {".pdf", ".docx", ".doc", ".html", ".htm"}


def build_chunk_id(source: str, index: int, content: str) -> str:
    """构造稳定 chunk id，避免重复运行时反复写入随机 UUID。"""
    digest = hashlib.sha1(f"{source}:{index}:{content}".encode("utf-8")).hexdigest()
    return digest[:24]


def normalize_chunks(chunks, file_path: Path, doc_type: str):
    """补齐 Chroma 过滤和去重所需的基础元数据。"""
    source = str(file_path)
    doc_id = hashlib.sha1(source.encode("utf-8")).hexdigest()[:16]

    # 从 loader 输出或文件路径推断 title / date
    title = next(
        (chunk.metadata.get(META_TITLE) for chunk in chunks if chunk.metadata.get(META_TITLE)),
        file_path.stem,
    )
    date = next(
        (chunk.metadata.get(META_DATE) for chunk in chunks if chunk.metadata.get(META_DATE)),
        "",
    )

    for index, chunk in enumerate(chunks):
        chunk_id = build_chunk_id(source, index, chunk.page_content)
        chunk.id = chunk_id
        chunk.metadata.setdefault(META_CHUNK_ID, chunk_id)
        chunk.metadata.setdefault(META_DOC_ID, doc_id)
        chunk.metadata.setdefault(META_DOC_TYPE, doc_type)
        chunk.metadata.setdefault(META_SOURCE, source)
        chunk.metadata.setdefault(META_TITLE, title)
        chunk.metadata.setdefault(META_DATE, date)
        chunk.metadata.setdefault(META_PERMISSION_LEVEL, PERMISSION_INTERNAL)
    return chunks


def ingest_document(file_path: Path, doc_type: str):
    """单文档入库流程"""
    print(f"处理: {file_path}")

    documents = []
    # 1. 加载
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        from src.ingestion.loaders import load_pdf

        documents = load_pdf(file_path=file_path)
    elif suffix in [".docx", ".doc"]:
        from src.ingestion.loaders import load_word

        documents = load_word(file_path=file_path)
    elif suffix in [".html", ".htm"]:
        from src.ingestion.loaders import load_html

        documents = load_html(file_path)
    else:
        print(f"不支持的文件格式: {suffix}")
        return

    # 2. 分块
    chunks = chunk_documents(documents=documents, doc_type=doc_type)
    chunks = normalize_chunks(chunks, file_path, doc_type)
    print(f"  分块数: {len(chunks)}")

    # 3. 向量化并存储
    embedding_model = get_embedding_model()
    embed_and_store(
        chunks=chunks,
        persist_directory=CHROMA_DEFAULT_PERSIST_DIR,
        embedding_model=embedding_model,
    )
    print(f"  入库完成: {len(chunks)} chunks")


def ingest_directory(directory: Path, doc_type: str):
    """批量入库"""
    for file_path in directory.rglob("*"):
        if not file_path.is_file() or file_path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        try:
            ingest_document(file_path=file_path, doc_type=doc_type)
        except Exception as e:
            print(f"失败: {file_path}, 错误: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python scripts/ingest.py <目录路径> <文档类型>")
        print("文档类型: research_report / announcement / regulation / financial_data / meeting_minutes")
        sys.exit(1)

    directory = Path(sys.argv[1])
    doc_type = sys.argv[2]
    if doc_type not in ALL_VALID_DOC_TYPES:
        print(f"不支持的文档类型: {doc_type}")
        sys.exit(1)
    ingest_directory(directory=directory, doc_type=doc_type)
