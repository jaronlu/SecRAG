from typing import Annotated, Optional, Sequence, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages

from src.schemas.typed_dicts import (
    AuditTrail,
    CitationDict,
    ComplianceResult,
    IntermediateStep,
    QueryEntities,
    RetrievalPlanStep,
    RetrievalResult,
    ToolCallDict,
    VerificationResult,
)

# ⚡ 字段统一：每个 key 的字符串值必须与 src/schemas/constants.py 中的 STATE_* 常量一致。
#   TypedDict 要求键名为字面量（Python 限制），无法直接引用常量作为键名；
#   作为补偿，每个字段注释标注对应的 STATE_* 常量，新增/重命名时务必同步更新 constants.py。


class AssistantState(TypedDict):
    # 用户上下文 — STATE_USER_ID / STATE_USER_ROLE / STATE_DEPARTMENT / STATE_DATA_PERMISSIONS / STATE_CLIENT_ID
    user_id: str
    user_role: str  # 值域见 src.schemas.constants.UserRole
    department: str
    data_permissions: list[str]  # 值域见 src.schemas.constants.ROLE_DATA_PERMISSIONS
    client_id: Optional[str]  # 投顾/销售场景关联客户

    # 查询理解 — STATE_ORIGINAL_QUERY / STATE_REWRITTEN_QUERY / STATE_INTENT / STATE_ENTITIES / STATE_AMBIGUITY / STATE_QUERY_TYPE
    original_query: str
    rewritten_query: str
    intent: str
    entities: QueryEntities
    ambiguity: list[str]
    query_type: str  # 值域见 src.schemas.constants.QueryType

    # 检索计划 — STATE_RETRIEVAL_PLAN / STATE_RETRIEVAL_ATTEMPTS
    retrieval_plan: list[RetrievalPlanStep]
    retrieval_attempts: int  # 多跳检索计数器（impl-04 使用）

    # 检索结果 — STATE_RETRIEVAL_RESULTS（reducer: concatenate）
    retrieval_results: Annotated[list[RetrievalResult], "concatenate"]

    # 推理过程 — STATE_MESSAGES / STATE_TOOL_CALLS / STATE_INTERMEDIATE_STEPS
    messages: Annotated[Sequence[BaseMessage], add_messages]
    tool_calls: list[ToolCallDict]
    intermediate_steps: list[IntermediateStep]

    # 验证结果 — STATE_VERIFICATION
    verification: VerificationResult

    # 合规检查 — STATE_COMPLIANCE
    compliance: ComplianceResult

    # 最终回答 — STATE_FINAL_ANSWER / STATE_CITATIONS / STATE_CONFIDENCE / STATE_RISK_DISCLOSURE
    final_answer: str
    citations: list[CitationDict]
    confidence: str  # 值域见 src.schemas.constants.Confidence
    risk_disclosure: str

    # 追踪 — STATE_AUDIT_TRAIL
    audit_trail: AuditTrail
