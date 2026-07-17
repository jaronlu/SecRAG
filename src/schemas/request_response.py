"""
SecRAG 请求/响应模型

⚡ 字段统一：Citation 模型以 src.rag.formatter 和 SCHEMA-REFERENCE §3.1 为权威定义。
此处仅做 API 序列化适配（Pydantic），全部字段与 impl-06 @dataclass Citation 对齐。
常量值引用 src.schemas.constants，不做硬编码。
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from src.schemas.typed_dicts import CitationDict, ComplianceResult


# ══════════════════════════════════════════════════════════════════════
# Agent 接口（impl-03 §7）
# ══════════════════════════════════════════════════════════════════════


class AssistantQARequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1, max_length=500)
    client_id: Optional[str] = None
    thread_id: Optional[str] = None


class AssistantQAResponse(BaseModel):
    thread_id: str
    turn_id: str
    answer: str
    citations: list[CitationDict]  # 序列化 Citation（SCHEMA-REFERENCE §3.1）
    confidence: str  # SCHEMA-REFERENCE §2.3
    compliance: ComplianceResult


class ConversationThreadCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    client_id: Optional[str] = None
    title: str = Field(default="新会话", min_length=1, max_length=100)


class ConversationThreadResponse(BaseModel):
    thread_id: str
    title: str
    created_at: str


class ConversationMessageResponse(BaseModel):
    message_id: str
    thread_id: str
    turn_id: str
    role: str
    content: str
    sequence: int
    created_at: str
    request_id: Optional[str] = None


class ConversationMessagesResponse(BaseModel):
    thread_id: str
    messages: list[ConversationMessageResponse]


class IngestionRunCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category_id: str = Field(..., min_length=1, max_length=100)


class IngestionRunCreateResponse(BaseModel):
    run_id: str
    category_id: str
    status: str
    queued_at: str


class IngestionCategoryResponse(BaseModel):
    category_id: str
    label: str
    group: str
    relative_path: str
    default_doc_type: str
    allowed_doc_types: list[str]
    file_count: int
    manifest_count: int
    invalid_manifest_count: int
    ready: bool
    error_code: str
    error: str


class IngestionCategoriesResponse(BaseModel):
    categories: list[IngestionCategoryResponse]
    active_run_id: Optional[str] = None


class IngestionFileResponse(BaseModel):
    relative_path: str
    extension: str
    doc_type: str
    permission_level: str
    allowed_roles: list[str]
    manifest_status: str
    error: str


class IngestionFilesResponse(BaseModel):
    category_id: str
    files: list[IngestionFileResponse]


class IngestionRunResponse(BaseModel):
    run_id: str
    category_id: str
    status: str
    queued_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    total_files: int
    processed_files: int
    created: int
    replaced: int
    skipped: int
    archived: int
    failed: int
    error_code: str
    error: str


class IngestionRunItemResponse(BaseModel):
    doc_id: str
    sequence: int
    relative_path: str
    action: str
    chunk_count: int
    processed_at: str
    error_code: str
    error: str


class IngestionRunItemsResponse(BaseModel):
    run_id: str
    items: list[IngestionRunItemResponse]


class IngestionChunkResponse(BaseModel):
    chunk_id: str
    chunk_index: int
    chunk_hash: str
    doc_type: str
    title: str
    stock_code: str
    date: str
    page_number: str
    content_length: int
    content: str
    permission_level: str
    allowed_roles: list[str]
    parser_version: str
    chunker_version: str
    embedding_model: str


class IngestionDocumentChunksResponse(BaseModel):
    doc_id: str
    total_chunks: int
    offset: int
    limit: int
    chunks: list[IngestionChunkResponse]


class IngestionRunsResponse(BaseModel):
    runs: list[IngestionRunResponse]
