# =============================================================================
#   src/ingestion/
#  ├── __init__.py      ← 包的"门面"，告诉 Python 这是个包
#  ├── loaders.py       ← 具体实现
#  ├── chunkers.py      ← 具体实现
#  └── embedder.py      ← 具体实现
# =============================================================================

# 用户只需 from src.ingestion import load_pdf, chunk_documents
# 不用关心内部文件结构（loaders.py / chunkers.py / embedder.py）

# from src.ingestion.loaders import (
#     load_pdf,           # 加载 PDF 文档 → List[Document]
#     load_word,          # 加载 Word 文档 → List[Document]
#     load_directory,     # 批量加载目录下所有文档 → List[Document]
#     load_announcement,  # 从 URL 加载公告 HTML → List[Document]
#     load_financial_csv, # 从 CSV 加载财务数据 → List[Document]
# )
#
# from src.ingestion.chunkers import (
#     create_financial_splitter,  # 创建默认金融分块器 → RecursiveCharacterTextSplitter
#     chunk_documents,            # 按文档类型分块 → List[Document]
# )
#
# from src.ingestion.embedder import (
#     get_embedding_model,  # 获取 Embedding 模型 → HuggingFaceEmbeddings
#     embed_and_store,      # 向量化并存入 ChromaDB → Chroma
# )
