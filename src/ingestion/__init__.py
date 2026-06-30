# =============================================================================
#   src/ingestion/
#  ├── __init__.py      ← 包的"门面"，告诉 Python 这是个包
#  ├── loaders.py       ← 具体实现
#  ├── chunkers.py      ← 具体实现
#  └── embedder.py      ← 具体实现
# =============================================================================

from src.ingestion.loaders import (
    load_directory,     # 批量加载目录下所有文档
    load_financial_csv, # 从 CSV 加载财务数据
    load_html,          # 加载公告 HTML
    load_pdf,           # 加载 PDF 文档
    load_word,          # 加载 Word 文档
)

from src.ingestion.chunkers import (
    chunk_documents,            # 按文档类型分块
    create_financial_splitter, # 创建默认金融分块器
)

from src.ingestion.embedder import (
    embed_and_store,      # 向量化并存入 ChromaDB
    get_embedding_model,  # 获取 Embedding 模型
)

__all__ = [
    "load_pdf",
    "load_word",
    "load_directory",
    "load_html",
    "load_financial_csv",
    "create_financial_splitter",
    "chunk_documents",
    "get_embedding_model",
    "embed_and_store",
]
