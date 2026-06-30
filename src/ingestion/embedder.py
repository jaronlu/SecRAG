from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

# ⚡ 字段统一：配置常量见 src/schemas/constants.py
from src.schemas.constants import CHROMA_COLLECTION_NAME, CHROMA_SPACE


def get_embedding_model(
    model_name: str = "BAAI/bge-m3",
) -> HuggingFaceEmbeddings:
    """
    返回一个 Embedding 转换器实例。

    ObjC 类比：
      相当于创建并配置一个 NSValueTransformer，
      指定用哪个模型（model_name）做文本→向量的转换。

    金融场景推荐：
      - BAAI/bge-m3：中文效果好，支持多语言（≈ 主力模型）
      - moka-ai/m3e-base：轻量，适合快速原型（≈ 轻量替代）
    """
    # HuggingFaceEmbeddings(...) 初始化 embedding 模型
    # 类似 [MyTransformer setModel:@"bge-m3"]
    return HuggingFaceEmbeddings(
        model_name=model_name,  # 模型名称，对应 HuggingFace Hub 上的 repo id
        model_kwargs={
            "device": "mps"
        },  # 推理设备；"cuda" 用 NVIDIA GPU，"mps" 用 Apple Silicon，默认 CPU
        encode_kwargs={
            "normalize_embeddings": True
        },  # 对输出向量做 L2 归一化（转成单位向量），cosine 相似度必备前提
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
            "hnsw:space": CHROMA_SPACE,
        },
    )
    return vectorstore  # 返回 Chroma 实例；≈ 返回 NSPersistentContainer，后续可用 .similarity_search() 做检索
