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


class Citation(BaseModel):
    doc_title: str
    source: str
    page: Optional[int] = None
    quote: str
    score: float


class QAResponse(BaseModel):
    answer: str
    citations: list[Citation]
    confidence: str  # high / medium / low
    retrieval_path: list[str]
