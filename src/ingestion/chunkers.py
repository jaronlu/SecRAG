"""金融文档分块策略"""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List

def create_financial_splitter(
    chunk_size:int = 500,
    chunk_overlap: int = 100
) -> RecursiveCharacterTextSplitter: 
    """
    金融文档专用分块器

    设计原则：
    1. 金融文档信息密度高，chunk_size 可以小一些（500字符）
    2. 保留章节标题作为上下文
    3. 中文优先：按段落/句子切分
    """
    return RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        separators=["\n\n", "\n", "。", "；", " ", ""],
        length_function=len,
    )

def chunk_documents(documents: List[Document], doc_type: str) -> List[Document]:
    """根据文档类型选择分块策略"""
    splitters = {
        "research_report": RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=100,
            separators=["\n\n", "\n", "。", "；", " ", ""],
        ),
        "announcement": RecursiveCharacterTextSplitter(
            chunk_size=300, chunk_overlap=50,
            separators=["\n\n", "\n", "。", "；", " ", ""],
        ),
        "financial_report": RecursiveCharacterTextSplitter(
            chunk_size=800, chunk_overlap=200,
            separators=["\n\n", "\n", "。", "；", " ", ""],
        ),
        "regulation": RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=100,
            separators=["\n\n", "\n", "。", "；", " ", ""],
        ),
        "meeting_minutes": RecursiveCharacterTextSplitter(
            chunk_size=400, chunk_overlap=80,
            separators=["\n\n", "\n", "。", "；", " ", ""],
        ),
    }

    splitters = splitters.get(doc_type, create_financial_splitter())
    return splitters.split_documents(documents)