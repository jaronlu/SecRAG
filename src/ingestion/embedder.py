import warnings

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

# ⚡ 字段统一：配置常量见 src/schemas/constants.py
from src.schemas.constants import (
    CHROMA_COLLECTION_NAME,
    CHROMA_EMBEDDING_MODEL_KEY,
    CHROMA_HNSW_SPACE_KEY,
    CHROMA_SPACE,
    CHROMA_UPSERT_BATCH_SIZE,
    DEFAULT_EMBEDDING_MODEL,
    META_DOC_ID,
)


def _collection_metadata(embedding_model_name: str) -> dict[str, str]:
    """构建 Chroma collection metadata，记录 embedding 模型名用于一致性校验。"""
    return {
        CHROMA_HNSW_SPACE_KEY: CHROMA_SPACE,
        CHROMA_EMBEDDING_MODEL_KEY: embedding_model_name,
    }


def verify_embedding_model_consistency(vectorstore: Chroma, expected_model: str) -> None:
    """校验 Chroma collection 中记录的 embedding 模型与当前配置一致。

    - 若 collection metadata 无 embedding_model（legacy 数据）：写入当前模型名并告警。
    - 若记录的模型与预期不一致：抛出 RuntimeError，阻止向量空间不匹配导致的检索失效。
    - 若 vectorstore 无 _collection 属性（如 mock 对象）：跳过校验。
    """
    collection = getattr(vectorstore, "_collection", None)
    if collection is None:
        return
    stored_model = (
        collection.metadata.get(CHROMA_EMBEDDING_MODEL_KEY) if collection.metadata else None
    )
    # 非字符串值（如 None 或 mock 对象）视为未设置
    if not isinstance(stored_model, str):
        # Legacy 数据：补写 metadata，不阻断
        warnings.warn(
            f"Chroma collection '{CHROMA_COLLECTION_NAME}' 未记录 embedding_model，"
            f"补写为 '{expected_model}'。若实际入库模型不同，检索结果将不可靠。",
            stacklevel=2,
        )
        try:
            collection.modify(metadata=_collection_metadata(expected_model))
        except Exception:
            warnings.warn(
                "无法更新 collection metadata（Chroma 版本可能不支持 modify）。", stacklevel=2
            )
        return
    if stored_model != expected_model:
        raise RuntimeError(
            f"Embedding 模型不匹配：Chroma collection 记录为 '{stored_model}'，"
            f"当前配置为 '{expected_model}'。入库与检索必须使用同一模型，"
            f"否则向量空间不匹配将导致检索完全失效。请重新入库或修正配置。"
        )


def _detect_device() -> str:
    """自动检测可用的 embedding 推理设备"""
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_built() and mps.is_available():
        return "mps"
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


def _model_name(embedding_model: HuggingFaceEmbeddings) -> str:
    """从 HuggingFaceEmbeddings 实例提取模型名。"""
    return getattr(embedding_model, "model_name", None) or DEFAULT_EMBEDDING_MODEL


def embed_and_store(
    chunks: list[
        Document
    ],  # 待入库的文档块列表；每个 Document 有 .page_content（文本）和 .metadata（来源等）
    persist_directory: str,  # 向量库持久化目录；≈ Core Data 的 SQLite 文件路径，目录不存在会自动创建
    embedding_model: HuggingFaceEmbeddings,  # 上面 get_embedding_model() 返回的转换器实例；≈ 传给 NSPersistentContainer 的 NSValueTransformer
) -> Chroma:  # 返回 Chroma 向量库实例；≈ NSPersistentContainer，之后可用来做 similarity_search（≈ fetch request）
    """将 chunks 向量化并存入 ChromaDB"""
    model_name = _model_name(embedding_model)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_name=CHROMA_COLLECTION_NAME,
        collection_metadata=_collection_metadata(model_name),
    )
    return vectorstore


def get_vectorstore(
    persist_directory: str,
    embedding_model: HuggingFaceEmbeddings,
) -> Chroma:
    """打开既有 Chroma 集合，不隐式写入文档，并校验 embedding 模型一致性。"""
    model_name = _model_name(embedding_model)
    vectorstore = Chroma(
        embedding_function=embedding_model,
        persist_directory=persist_directory,
        collection_name=CHROMA_COLLECTION_NAME,
        collection_metadata=_collection_metadata(model_name),
    )
    verify_embedding_model_consistency(vectorstore, model_name)
    return vectorstore


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
