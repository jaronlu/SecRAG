"""
SecRAG 请求/响应模型

⚡ 字段统一：Citation 模型以 src.rag.formatter 和 SCHEMA-REFERENCE §3.1 为权威定义。
此处仅做 API 序列化适配（Pydantic），全部字段与 impl-06 @dataclass Citation 对齐。
常量值引用 src.schemas.constants（DEFAULT_TOP_K, CONFIDENCE_* 等），不做硬编码。
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from src.schemas.constants import DEFAULT_TOP_K
from src.schemas.typed_dicts import AuditTrail, CitationDict, ComplianceResult


class QARequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=500,
    )
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=20, description="检索返回的条数")
    doc_type: Optional[str] = Field(default=None, description="文档类型过滤")
    stream: bool = Field(default=False, description="是否流式返回")


class QAResponse(BaseModel):
    answer: str
    citations: list[CitationDict]  # 序列化后的 Citation（SCHEMA-REFERENCE §3.1）
    confidence: str  # CONFIDENCE_HIGH / CONFIDENCE_MEDIUM / CONFIDENCE_LOW（SCHEMA-REFERENCE §2.3）
    retrieval_path: list[str]


# ══════════════════════════════════════════════════════════════════════
# Agent 接口（impl-03 §7）
# ══════════════════════════════════════════════════════════════════════


class AssistantQARequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1, max_length=500)
    client_id: Optional[str] = None
    thread_id: Optional[str] = None


class AssistantQAResponse(BaseModel):
    answer: str
    citations: list[CitationDict]  # 序列化 Citation（SCHEMA-REFERENCE §3.1）
    confidence: str  # SCHEMA-REFERENCE §2.3
    compliance: ComplianceResult
    audit_trail: AuditTrail
