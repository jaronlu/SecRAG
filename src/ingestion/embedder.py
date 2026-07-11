import platform
import warnings

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

# ⚡ 字段统一：配置常量见 src/schemas/constants.py
from src.schemas.constants import (
    CHROMA_COLLECTION_NAME,
    CHROMA_HNSW_SPACE_KEY,
    CHROMA_SPACE,
    CHROMA_UPSERT_BATCH_SIZE,
    DEFAULT_EMBEDDING_MODEL,
    META_DOC_ID,
)


def _detect_device() -> str:
    """自动检测可用的 embedding 推理设备"""
    system = platform.system()
    if system == "Darwin":
        return "mps"
    if system == "Linux":
        try:
            import torch  # noqa: F811

            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
    return "cpu"


def get_embedding_model(
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> HuggingFaceEmbeddings:
    """
    返回一个 Embedding 转换器实例。

    优先从本地 HuggingFace 缓存加载（秒级），缓存未命中时自动回退到在线下载。
    业务场景推荐：
      - BAAI/bge-m3：中文效果好，支持多语言（≈ 主力模型）
      - moka-ai/m3e-base：轻量，适合快速原型（≈ 轻量替代）
    """
    device = _detect_device()
    try:
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device, "local_files_only": True},
            encode_kwargs={"normalize_embeddings": True},
        )
    except OSError:
        warnings.warn(f"本地缓存未命中，尝试在线下载模型: {model_name}", stacklevel=1)
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )


def embed_and_store(
    chunks: list[
        Document
    ],  # 待入库的文档块列表；每个 Document 有 .page_content（文本）和 .metadata（来源等）
    persist_directory: str,  # 向量库持久化目录；≈ Core Data 的 SQLite 文件路径，目录不存在会自动创建
    embedding_model: HuggingFaceEmbeddings,  # 上面 get_embedding_model() 返回的转换器实例；≈ 传给 NSPersistentContainer 的 NSValueTransformer
) -> Chroma:  # 返回 Chroma 向量库实例；≈ NSPersistentContainer，之后可用来做 similarity_search（≈ fetch request）
    """将 chunks 向量化并存入 ChromaDB"""
    # Chroma.from_documents() 是一个类工厂方法
    # 类比：[NSPersistentContainer performFetch:request withTransformer:transformer]
    # 1. 遍历 chunks，用 embedding_model 把每条 page_content 转成 float 向量
    # 2. 向量 + 原文 + metadata 写入 Chroma 集合
    # 3. 持久化到 persist_directory
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_name=CHROMA_COLLECTION_NAME,
        collection_metadata={
            CHROMA_HNSW_SPACE_KEY: CHROMA_SPACE,
        },
    )
    return vectorstore  # 返回 Chroma 实例；≈ 返回 NSPersistentContainer，后续可用 .similarity_search() 做检索


def get_vectorstore(
    persist_directory: str,
    embedding_model: HuggingFaceEmbeddings,
) -> Chroma:
    """打开既有 Chroma 集合，不隐式写入文档。"""
    return Chroma(
        embedding_function=embedding_model,
        persist_directory=persist_directory,
        collection_name=CHROMA_COLLECTION_NAME,
        collection_metadata={
            CHROMA_HNSW_SPACE_KEY: CHROMA_SPACE,
        },
    )


def upsert_chunks(
    chunks: list[Document],
    persist_directory: str,
    embedding_model: HuggingFaceEmbeddings,
    batch_size: int = CHROMA_UPSERT_BATCH_SIZE,
) -> None:
    """按稳定 chunk.id 分批 upsert，避免超过 Chroma 单批上限。"""
    if not chunks:
        return
    if batch_size <= 0:
        raise ValueError("batch_size 必须大于 0")
    vectorstore = get_vectorstore(
        persist_directory=persist_directory,
        embedding_model=embedding_model,
    )
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        vectorstore.add_documents(batch, ids=[str(chunk.id) for chunk in batch])


def list_chunk_ids_by_doc_id(
    doc_id: str,
    persist_directory: str,
    embedding_model: HuggingFaceEmbeddings,
) -> list[str]:
    """列出某个 doc_id 当前在 Chroma 中的 chunk IDs。"""
    vectorstore = get_vectorstore(
        persist_directory=persist_directory,
        embedding_model=embedding_model,
    )
    results = vectorstore.get(where={META_DOC_ID: doc_id}, include=[])
    ids = results.get("ids", [])
    return [str(chunk_id) for chunk_id in ids]


def delete_chunk_ids(
    chunk_ids: set[str] | list[str],
    persist_directory: str,
    embedding_model: HuggingFaceEmbeddings,
) -> None:
    """按显式 chunk IDs 删除 stale chunks。"""
    ids = sorted(chunk_ids)
    if not ids:
        return
    vectorstore = get_vectorstore(
        persist_directory=persist_directory,
        embedding_model=embedding_model,
    )
    vectorstore.delete(ids=ids)
