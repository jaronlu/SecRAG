from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


def get_embedding_model(model_name: str = "BAAI/bge-m3") -> HuggingFaceEmbeddings:
    """
    金融场景推荐 Embedding 模型：
    - BAAI/bge-m3：中文效果好，支持多语言
    - moka-ai/m3e-base：轻量，适合快速原型
    """
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},  # 或 "cuda" / "mps"
        encode_kwargs={"normalize_embeddings": True},
    )

def embed_and_store(
    chunks: list[Document],
    persist_directory: str,
    embedding_model: HuggingFaceEmbeddings,
) -> Chroma:
    """将 chunks 向量化并存入 ChromaDB"""
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_name="securities_docs",
        collection_metadata={"hnsw:space": "cosine"},
    )
    return vectorstore
