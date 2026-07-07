"""
SecRAG — 字段名常量、枚举值、配置常量

本文件是项目中所有硬编码字符串字段名的唯一权威来源。
任何模块不得直接使用裸字符串作为 metadata key / State key / 枚举值。
定义见 SCHEMA-REFERENCE.md 中的对应章节。
"""

from typing import Final, Literal

# ══════════════════════════════════════════════════════════════════════
# metadata 字段键名  (SCHEMA-REFERENCE §1.1)
# ══════════════════════════════════════════════════════════════════════

META_CHUNK_ID: Final = "chunk_id"
META_DOC_ID: Final = "doc_id"
META_DOC_TYPE: Final = "doc_type"
META_SOURCE: Final = "source"
META_TITLE: Final = "title"
META_DATE: Final = "date"
META_STOCK_CODE: Final = "stock_code"
META_PERMISSION_LEVEL: Final = "permission_level"
META_PAGE_NUMBER: Final = "page_number"
META_PRODUCT_TYPE: Final = "product_type"
META_ERROR: Final = "error"
META_ALLOWED_ROLES: Final = "allowed_roles"
META_RETRIEVAL_SOURCE: Final = "retrieval_source"
META_FILE_HASH: Final = "file_hash"
META_METADATA_HASH: Final = "metadata_hash"
META_PARSE_HASH: Final = "parse_hash"
META_CHUNK_HASH: Final = "chunk_hash"
META_CHUNK_INDEX: Final = "chunk_index"
META_DOC_VERSION: Final = "doc_version"
META_INGESTED_AT: Final = "ingested_at"
META_PARSER_VERSION: Final = "parser_version"
META_CHUNKER_VERSION: Final = "chunker_version"
META_EMBEDDING_MODEL: Final = "embedding_model"

# ══════════════════════════════════════════════════════════════════════
# retrieval_results dict 键名  (SCHEMA-REFERENCE §1.2)
# ══════════════════════════════════════════════════════════════════════

RR_CONTENT: Final = "content"
RR_METADATA: Final = "metadata"
RR_SCORE: Final = "score"
RR_DENIED: Final = "denied"
RR_REASON: Final = "reason"

# ══════════════════════════════════════════════════════════════════════
# retrieval_plan step 键名  (SCHEMA-REFERENCE §1.3)
# ══════════════════════════════════════════════════════════════════════

PLAN_SOURCE: Final = "source"
PLAN_QUERY: Final = "query"
PLAN_TOP_K: Final = "top_k"
PLAN_FILTERS: Final = "filters"
PLAN_DENIED: Final = "denied"
PLAN_REASON: Final = "reason"

# ══════════════════════════════════════════════════════════════════════
# AssistantState 字段键名  (SCHEMA-REFERENCE §3.7)
# ══════════════════════════════════════════════════════════════════════

STATE_USER_ID: Final = "user_id"
STATE_USER_ROLE: Final = "user_role"
STATE_DEPARTMENT: Final = "department"
STATE_DATA_PERMISSIONS: Final = "data_permissions"
STATE_CLIENT_ID: Final = "client_id"
STATE_ORIGINAL_QUERY: Final = "original_query"
STATE_REWRITTEN_QUERY: Final = "rewritten_query"
STATE_INTENT: Final = "intent"
STATE_ENTITIES: Final = "entities"
STATE_AMBIGUITY: Final = "ambiguity"
STATE_QUERY_TYPE: Final = "query_type"
STATE_RETRIEVAL_ATTEMPTS: Final = "retrieval_attempts"
STATE_REASON_ATTEMPTS: Final = "reason_attempts"
STATE_RETRIEVAL_PLAN: Final = "retrieval_plan"
STATE_RETRIEVAL_RESULTS: Final = "retrieval_results"
STATE_MESSAGES: Final = "messages"
STATE_TOOL_CALLS: Final = "tool_calls"
STATE_INTERMEDIATE_STEPS: Final = "intermediate_steps"
STATE_VERIFICATION: Final = "verification"
STATE_COMPLIANCE: Final = "compliance"
STATE_FINAL_ANSWER: Final = "final_answer"
STATE_CITATIONS: Final = "citations"
STATE_CONFIDENCE: Final = "confidence"
STATE_RISK_DISCLOSURE: Final = "risk_disclosure"
STATE_AUDIT_TRAIL: Final = "audit_trail"

# audit_trail 字段键名（SCHEMA-REFERENCE §3.5）
AUDIT_REQUEST_ID: Final = "request_id"
AUDIT_TIMESTAMP: Final = "timestamp"
AUDIT_STARTED_PERF_COUNTER: Final = "_started_perf_counter"
AUDIT_QUERY: Final = "query"
AUDIT_RETRIEVAL: Final = "retrieval"
AUDIT_REASONING: Final = "reasoning"
AUDIT_VERIFICATION: Final = "verification"
AUDIT_COMPLIANCE: Final = "compliance"
AUDIT_RESPONSE: Final = "response"
AUDIT_QUERY_ORIGINAL: Final = "original"
AUDIT_QUERY_REWRITTEN: Final = "rewritten"
AUDIT_QUERY_INTENT: Final = "intent"
AUDIT_QUERY_TYPE: Final = "query_type"
AUDIT_QUERY_ENTITIES: Final = "entities"
AUDIT_RETRIEVAL_PLAN: Final = "plan"
AUDIT_RETRIEVAL_SOURCES: Final = "sources"
AUDIT_RETRIEVAL_TOTAL_CHUNKS: Final = "total_chunks"
AUDIT_RETRIEVAL_FILTERED_CHUNKS: Final = "filtered_chunks"
AUDIT_REASONING_TOOL_CALLS: Final = "tool_calls"
AUDIT_REASONING_ITERATIONS: Final = "iterations"
AUDIT_REASONING_DURATION_MS: Final = "duration_ms"
AUDIT_RESPONSE_CITATIONS: Final = "citations"
AUDIT_RESPONSE_CONFIDENCE: Final = "confidence"
AUDIT_RESPONSE_RISK_DISCLOSURE: Final = "risk_disclosure"

# ══════════════════════════════════════════════════════════════════════
# doc_type 枚举  (SCHEMA-REFERENCE §2.1)
# ══════════════════════════════════════════════════════════════════════

DOC_TYPE_RESEARCH_REPORT: Final = "research_report"
DOC_TYPE_ANNOUNCEMENT: Final = "announcement"
DOC_TYPE_REGULATION: Final = "regulation"
DOC_TYPE_FINANCIAL_DATA: Final = "financial_data"
DOC_TYPE_MEETING_MINUTES: Final = "meeting_minutes"
DOC_TYPE_PRODUCT: Final = "product"
DOC_TYPE_FAQ: Final = "faq"

ALL_VALID_DOC_TYPES: set[str] = {
    DOC_TYPE_RESEARCH_REPORT,
    DOC_TYPE_ANNOUNCEMENT,
    DOC_TYPE_REGULATION,
    DOC_TYPE_FINANCIAL_DATA,
    DOC_TYPE_MEETING_MINUTES,
    DOC_TYPE_PRODUCT,
    DOC_TYPE_FAQ,
}

# ══════════════════════════════════════════════════════════════════════
# 用户角色枚举  (SCHEMA-REFERENCE §2.2)
# ══════════════════════════════════════════════════════════════════════

ROLE_ADVISOR: Final = "advisor"
ROLE_INSTITUTIONAL_SALES: Final = "institutional_sales"
ROLE_COMPLIANCE: Final = "compliance"
ROLE_OPERATIONS: Final = "operations"
ROLE_TECHNICAL: Final = "technical"

UserRole = Literal[
    ROLE_ADVISOR,
    ROLE_INSTITUTIONAL_SALES,
    ROLE_COMPLIANCE,
    ROLE_OPERATIONS,
    ROLE_TECHNICAL,
]

# ══════════════════════════════════════════════════════════════════════
# confidence 枚举  (SCHEMA-REFERENCE §2.3)
# ══════════════════════════════════════════════════════════════════════

CONFIDENCE_HIGH: Final = "high"
CONFIDENCE_MEDIUM: Final = "medium"
CONFIDENCE_LOW: Final = "low"

Confidence = Literal[CONFIDENCE_HIGH, CONFIDENCE_MEDIUM, CONFIDENCE_LOW]

# ══════════════════════════════════════════════════════════════════════
# permission_level 枚举  (SCHEMA-REFERENCE §2.4)
# ══════════════════════════════════════════════════════════════════════

PERMISSION_PUBLIC: Final = "public"
PERMISSION_INTERNAL: Final = "internal"
PERMISSION_CONFIDENTIAL: Final = "confidential"

# ══════════════════════════════════════════════════════════════════════
# retrieval_source 枚举  (SCHEMA-REFERENCE §2.6)
# ══════════════════════════════════════════════════════════════════════

SOURCE_PRODUCT: Final = "product_search"
SOURCE_REGULATION: Final = "regulation_search"
SOURCE_REPORT: Final = "report_search"
SOURCE_FAQ: Final = "faq_search"

# ══════════════════════════════════════════════════════════════════════
# query_type 枚举  (SCHEMA-REFERENCE §2.5)
# ══════════════════════════════════════════════════════════════════════

QT_PRODUCT_INQUIRY: Final = "product_inquiry"
QT_RULE_INQUIRY: Final = "rule_inquiry"
QT_REGULATION_INQUIRY: Final = "regulation_inquiry"
QT_REPORT_INQUIRY: Final = "report_inquiry"
QT_FAQ_INQUIRY: Final = "faq_inquiry"
QT_TECHNICAL_INQUIRY: Final = "technical_inquiry"

QueryType = Literal[
    QT_PRODUCT_INQUIRY,
    QT_RULE_INQUIRY,
    QT_REGULATION_INQUIRY,
    QT_REPORT_INQUIRY,
    QT_FAQ_INQUIRY,
    QT_TECHNICAL_INQUIRY,
]

# ══════════════════════════════════════════════════════════════════════
# ChromaDB 配置  (SCHEMA-REFERENCE §4)
# ══════════════════════════════════════════════════════════════════════

CHROMA_COLLECTION_NAME = "securities_docs"
CHROMA_DEFAULT_PERSIST_DIR = "./data/chroma"
CHROMA_HNSW_SPACE_KEY = "hnsw:space"
CHROMA_SPACE = "cosine"
FINANCIAL_DB_PATH = "data/financial.db"
AUDIT_DB_PATH = "data/audit.db"
INGEST_REGISTRY_DB_PATH = "data/ingest_registry.db"
SAMPLE_METADATA_FILENAME = "metadata.json"
API_ROUTE_QA = "/v1/qa"
API_ROUTE_ASSISTANT_QA = "/v1/assistant/qa"

# ══════════════════════════════════════════════════════════════════════
# 角色 → 可用的检索源映射  (SCHEMA-REFERENCE §5.1)
# ══════════════════════════════════════════════════════════════════════

ROLE_ALLOWED_SOURCES: dict[str, list[str]] = {
    ROLE_ADVISOR: [SOURCE_PRODUCT, SOURCE_FAQ, SOURCE_REPORT],
    ROLE_INSTITUTIONAL_SALES: [SOURCE_PRODUCT, SOURCE_REPORT, SOURCE_REGULATION],
    ROLE_COMPLIANCE: [SOURCE_REGULATION, SOURCE_PRODUCT, SOURCE_REPORT],
    ROLE_OPERATIONS: [SOURCE_PRODUCT, SOURCE_FAQ, SOURCE_REGULATION],
    ROLE_TECHNICAL: [SOURCE_FAQ],
}

# ══════════════════════════════════════════════════════════════════════
# 角色 → 数据权限级别映射  (SCHEMA-REFERENCE §5.2)
# ══════════════════════════════════════════════════════════════════════

ROLE_DATA_PERMISSIONS: dict[str, list[str]] = {
    ROLE_ADVISOR: [PERMISSION_PUBLIC, PERMISSION_INTERNAL],
    ROLE_INSTITUTIONAL_SALES: [PERMISSION_PUBLIC, PERMISSION_INTERNAL],
    ROLE_COMPLIANCE: [PERMISSION_PUBLIC, PERMISSION_INTERNAL, PERMISSION_CONFIDENTIAL],
    ROLE_OPERATIONS: [PERMISSION_PUBLIC, PERMISSION_INTERNAL],
    ROLE_TECHNICAL: [PERMISSION_PUBLIC, PERMISSION_INTERNAL, PERMISSION_CONFIDENTIAL],
}

# ══════════════════════════════════════════════════════════════════════
# 默认值 / 阈值  (SCHEMA-REFERENCE §4)
# ══════════════════════════════════════════════════════════════════════

DEFAULT_TOP_K = 5
DEFAULT_MAX_HOPS = 3
MAX_REASON_ATTEMPTS = 2
GRADE_TOP_K = 10

CONFIDENCE_HIGH_THRESHOLD = 0.75
CONFIDENCE_MEDIUM_THRESHOLD = 0.5
CONFIDENCE_HIGH_MIN_RESULTS = 3
RETRIEVAL_MIN_SCORE = 0.6

# ══════════════════════════════════════════════════════════════════════
# Embedding 默认值  (SCHEMA-REFERENCE §4)
# ══════════════════════════════════════════════════════════════════════

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"

# ══════════════════════════════════════════════════════════════════════
# LLM 提供方默认值  (SCHEMA-REFERENCE §4)
# ══════════════════════════════════════════════════════════════════════

OPENAI_DEFAULT_API_BASE = "https://api.stepfun.com/step_plan/v1"
OPENAI_DEFAULT_MODEL = "step-3.7-flash"
OLLAMA_DEFAULT_BASE_URL = "http://localhost:11434"
OLLAMA_DEFAULT_MODEL = "llama3.1:8b"
LLM_DEFAULT_TEMPERATURE = 0.1

# ══════════════════════════════════════════════════════════════════════
# LLM 提供方标识  (SCHEMA-REFERENCE §2.7)
# ══════════════════════════════════════════════════════════════════════

LLM_PROVIDER_OPENAI: Final = "openai"
LLM_PROVIDER_OLLAMA: Final = "ollama"
