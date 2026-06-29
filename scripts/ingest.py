"""完整入库流程"""

import sys
from pathlib import Path

from src.ingestion.chunkers import chunk_documents
from src.ingestion.embedder import embed_and_store, get_embedding_model


def ingest_document(file_path: Path, doc_type: str):
    """单文档入库流程"""
    print(f"处理：{file_path}")

    documents = []
    # 1. 加载
    if file_path.suffix.lower() == ".pdf":
        from src.ingestion.loaders import load_pdf

        documents = load_pdf(file_path=file_path)
    elif file_path.suffix.lower in [".doc", ".doc"]:
        from src.ingestion.loaders import load_word

        documents = load_word(file_path=file_path)
    else:
        print(f"不支持的文件格式{file_path.suffix}")
        return

    # 2. 分块
    chunks = chunk_documents(documents=documents, doc_type=doc_type)
    print(f"  分块数: {len(chunks)}")

    # 3. 向量化并存储
    embedding_model = get_embedding_model()
    embed_and_store(
        chunks=chunks, persist_directory="./data/chroma", embedding_model=embedding_model
    )
    print(f"  入库完成: {len(chunks)} chunks")


def ingest_directory(directory: Path, doct_pype: str):
    """批量入库"""

    for file_path in (
        list(directory.rglob("*.pdf"))
        + list(directory.rglob("*.docx"))
        + list(directory.rglob("*.doc"))
    ):
        try:
            ingest_document(file_path=file_path, doc_type=doct_pype)
        except Exception as e:
            print(f"失败: {file_path}, 错误: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python scripts/ingest.py <目录路径> <文档类型>")
        print("文档类型: research_report / announcement / regulation / meeting_minutes")
        sys.exit(1)

    directory = Path(sys.argv[1])
    doc_type = sys.argv[2]
    ingest_directory(directory=directory, doct_pype=doc_type)
