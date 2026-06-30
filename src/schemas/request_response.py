"""
SecRAG 请求/响应模型

⚡ 字段统一：Citation 模型以 src.rag.formatter 和 SCHEMA-REFERENCE §3.1 为权威定义。
此处仅做 API 序列化适配（Pydantic），全部字段与 impl-06 @dataclass Citation 对齐。
"""

from typing import Optional

from pydantic import BaseModel, Field


class QARequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=500,
    )
    top_k: int = Field(default=5, ge=1, le=20, description="检索返回的条数")
    doc_type: Optional[str] = Field(default=None, description="文档类型过滤")
    stream: bool = Field(default=False, description="是否流式返回")


class QAResponse(BaseModel):
    answer: str
    citations: list[dict]  # 序列化后的 Citation（SCHEMA-REFERENCE §3.1）
    confidence: str  # high / medium / low
    retrieval_path: list[str]
