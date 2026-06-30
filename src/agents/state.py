from typing import Annotated, Optional, Sequence, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages


class AssistantState(TypedDict):
    # 用户上下文
    user_id: str
    user_role: str  # src.schemas.constants.ROLE_ALLOWED_SOURCES
    department: str
    data_permissions: list[str]  # src.schemas.constants.ROLE_DATA_PERMISSIONS
    client_id: Optional[str]  # 投顾/销售场景关联客户

    # 查询理解
    original_query: str
    rewritten_query: str
    intent: str
    entities: dict  # 键名见 SCHEMA-REFERENCE §1.1 metadata 常量
    ambiguity: list[str]
    query_type: str  # SCHEMA-REFERENCE §2.5

    # 检索计划
    retrieval_plan: list[dict]

    # 检索结果（reducer: concatenate）
    retrieval_results: Annotated[list[dict], "concatenate"]  # 元素键名见 SCHEMA-REFERENCE §1.2

    # 推理过程
    messages: Annotated[Sequence[BaseMessage], add_messages]
    tool_calls: list[dict]  # 见 SCHEMA-REFERENCE §3.5 ToolCall
    intermediate_steps: list[dict]

    # 验证结果
    verification: dict  # {passed, issues, confidence}

    # 合规检查
    compliance: dict  # {passed, flags, risk_disclosure}

    # 最终回答
    final_answer: str
    citations: list[dict]  # 序列化 Citation（SCHEMA-REFERENCE §3.1）
    confidence: str  # SCHEMA-REFERENCE §2.3
    risk_disclosure: str

    # 审计
    audit_trail: dict  # 序列化 AuditEntry（SCHEMA-REFERENCE §3.3）
