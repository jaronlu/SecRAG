"""
SecRAG — 字段名常量、枚举值、配置常量

本文件是项目中所有硬编码字符串字段名的唯一权威来源。
任何模块不得直接使用裸字符串作为 metadata key / State key / 枚举值。
定义见 SCHEMA-REFERENCE.md 中的对应章节。
"""

from typing import Literal

# ══════════════════════════════════════════════════════════════════════
# metadata 字段键名  (SCHEMA-REFERENCE §1.1)
# ══════════════════════════════════════════════════════════════════════

META_CHUNK_ID = "chunk_id"
META_DOC_ID = "doc_id"
META_DOC_TYPE = "doc_type"
META_SOURCE = "source"
META_TITLE = "title"
META_DATE = "date"
META_STOCK_CODE = "stock_code"
META_COMPANY_NAME = "company_name"
META_ANALYST = "analyst"
META_INSTITUTION = "institution"
META_PERMISSION_LEVEL = "permission_level"
META_PAGE_NUMBER = "page_number"
META_RETRIEVAL_PATH = "retrieval_path"
META_ERROR = "error"

# ══════════════════════════════════════════════════════════════════════
# retrieval_results dict 键名  (SCHEMA-REFERENCE §1.2)
# ══════════════════════════════════════════════════════════════════════

RR_CONTENT = "content"
RR_METADATA = "metadata"
RR_SCORE = "score"
RR_DENIED = "denied"
RR_REASON = "reason"

# ══════════════════════════════════════════════════════════════════════
# doc_type 枚举  (SCHEMA-REFERENCE §2.1)
# ══════════════════════════════════════════════════════════════════════

DOC_TYPE_RESEARCH_REPORT = "research_report"
DOC_TYPE_ANNOUNCEMENT = "announcement"
DOC_TYPE_REGULATION = "regulation"
DOC_TYPE_FINANCIAL_DATA = "financial_data"
DOC_TYPE_MEETING_MINUTES = "meeting_minutes"

DocType = Literal[
    DOC_TYPE_RESEARCH_REPORT,
    DOC_TYPE_ANNOUNCEMENT,
    DOC_TYPE_REGULATION,
    DOC_TYPE_FINANCIAL_DATA,
    DOC_TYPE_MEETING_MINUTES,
]

ALL_VALID_DOC_TYPES: set[str] = {
    DOC_TYPE_RESEARCH_REPORT,
    DOC_TYPE_ANNOUNCEMENT,
    DOC_TYPE_REGULATION,
    DOC_TYPE_FINANCIAL_DATA,
    DOC_TYPE_MEETING_MINUTES,
}

# ══════════════════════════════════════════════════════════════════════
# 用户角色枚举  (SCHEMA-REFERENCE §2.2)
# ══════════════════════════════════════════════════════════════════════

ROLE_ADVISOR = "advisor"
ROLE_INSTITUTIONAL_SALES = "institutional_sales"
ROLE_COMPLIANCE = "compliance"
ROLE_OPERATIONS = "operations"
ROLE_TECHNICAL = "technical"

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

CONFIDENCE_HIGH = "high"
CONFIDENCE_MEDIUM = "medium"
CONFIDENCE_LOW = "low"

Confidence = Literal[CONFIDENCE_HIGH, CONFIDENCE_MEDIUM, CONFIDENCE_LOW]

# ══════════════════════════════════════════════════════════════════════
# permission_level 枚举  (SCHEMA-REFERENCE §2.4)
# ══════════════════════════════════════════════════════════════════════

PERMISSION_PUBLIC = "public"
PERMISSION_INTERNAL = "internal"
PERMISSION_CONFIDENTIAL = "confidential"

PermissionLevel = Literal[PERMISSION_PUBLIC, PERMISSION_INTERNAL, PERMISSION_CONFIDENTIAL]

# ══════════════════════════════════════════════════════════════════════
# retrieval_source 枚举  (SCHEMA-REFERENCE §2.6)
# ══════════════════════════════════════════════════════════════════════

SOURCE_PRODUCT = "product_search"
SOURCE_REGULATION = "regulation_search"
SOURCE_REPORT = "report_search"
SOURCE_FAQ = "faq_search"

# ══════════════════════════════════════════════════════════════════════
# ChromaDB 配置  (SCHEMA-REFERENCE §4)
# ══════════════════════════════════════════════════════════════════════

CHROMA_COLLECTION_NAME = "securities_docs"
CHROMA_DEFAULT_PERSIST_DIR = "./data/chroma"
CHROMA_SPACE = "cosine"

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
    ROLE_ADVISOR: ["public", PERMISSION_INTERNAL],
    ROLE_INSTITUTIONAL_SALES: ["public", PERMISSION_INTERNAL],
    ROLE_COMPLIANCE: ["public", PERMISSION_INTERNAL, PERMISSION_CONFIDENTIAL],
    ROLE_OPERATIONS: ["public", PERMISSION_INTERNAL],
    ROLE_TECHNICAL: ["public", PERMISSION_INTERNAL, PERMISSION_CONFIDENTIAL],
}

# ══════════════════════════════════════════════════════════════════════
# 默认值 / 阈值  (SCHEMA-REFERENCE §4)
# ══════════════════════════════════════════════════════════════════════

DEFAULT_TOP_K = 5
DEFAULT_MAX_HOPS = 3
DEFAULT_RERANK_TOP_K = 5

CONFIDENCE_HIGH_THRESHOLD = 0.75
CONFIDENCE_MEDIUM_THRESHOLD = 0.5
CONFIDENCE_HIGH_MIN_RESULTS = 3
HALLUCINATION_THRESHOLD = 0.3

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

LLM_PROVIDER_OPENAI = "openai"
LLM_PROVIDER_OLLAMA = "ollama"
