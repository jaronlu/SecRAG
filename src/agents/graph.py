"""Agent Graph 构建——节点编排、条件路由、Checkpointer"""

import time
from typing import Literal, Protocol

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.agents.nodes import (
    audit_log,
    compliance_check,
    compose,
    extract_citations,
    grade_and_filter,
    load_conversation_context,
    permission_denied_response,
    persist_conversation_turn,
    planner,
    query_understand,
    reason,
    resolve_followup_query,
    retrieve,
    verify,
)
from src.agents.state import AssistantState
from src.schemas.constants import (
    CONFIDENCE_HIGH_MIN_RESULTS,
    DEFAULT_MAX_HOPS,
    MAX_REASON_ATTEMPTS,
    STATE_COMPLIANCE,
    STATE_INTERMEDIATE_STEPS,
    STATE_REASON_ATTEMPTS,
    STATE_RETRIEVAL_ATTEMPTS,
    STATE_RETRIEVAL_RESULTS,
    STATE_VERIFICATION,
)
from src.schemas.typed_dicts import IntermediateStep


class _AgentNode(Protocol):
    def __call__(self, state: AssistantState) -> AssistantState: ...


def _traced_node(
    name: str,
    node: _AgentNode,
) -> _AgentNode:
    """Record the actual node path and elapsed time in state."""

    def wrapped(state: AssistantState) -> AssistantState:
        started = time.perf_counter()
        result = node(state)
        step: IntermediateStep = {
            "step": name,
            "duration_ms": max((time.perf_counter() - started) * 1000, 0.0),
            "success": True,
        }
        return {
            **result,
            STATE_INTERMEDIATE_STEPS: result.get(STATE_INTERMEDIATE_STEPS, []) + [step],
        }

    return wrapped


# ══════════════════════════════════════════════════════════════════════
# 5.1 条件路由函数
# ══════════════════════════════════════════════════════════════════════


def should_retry_retrieval(
    state: AssistantState,
) -> Literal["denied", "continue", "retrieve"]:
    """判断是否需要补充检索（最多 DEFAULT_MAX_HOPS 次，计数器由 retrieve 节点维护）"""
    attempts = state.get(STATE_RETRIEVAL_ATTEMPTS, 0)
    results = state.get(STATE_RETRIEVAL_RESULTS, [])
    usable = [result for result in results if not result.get("denied")]
    if results and not usable:
        return "denied"
    if attempts >= DEFAULT_MAX_HOPS:
        return "continue"

    if not results:
        return "retrieve"
    if len(usable) < CONFIDENCE_HIGH_MIN_RESULTS:
        return "retrieve"
    return "continue"


def should_reason_again(state: AssistantState) -> Literal["retry", "continue"]:
    """判断验证是否通过；失败重推受 MAX_REASON_ATTEMPTS 显式限制。"""
    verification = state.get(STATE_VERIFICATION, {})
    attempts = state.get(STATE_REASON_ATTEMPTS, 0)
    if not verification.get("passed", False) and attempts < MAX_REASON_ATTEMPTS:
        return "retry"
    return "continue"


def is_compliant(state: AssistantState) -> Literal["pass", "block"]:
    """判断是否通过合规检查"""
    compliance = state.get(STATE_COMPLIANCE, {})
    if compliance.get("passed", False):
        return "pass"
    return "block"


# ══════════════════════════════════════════════════════════════════════
# 5.2 Graph 定义
# ══════════════════════════════════════════════════════════════════════

"""
认证身份
→ 加载会话
→ 消解追问
→ 查询理解
→ 生成检索计划
→ 执行检索
→ 过滤结果
→ ReAct 推理与工具调用
→ 提取引用
→ 四层验证
→ 合规检查
→ 组织回答
→ 保存会话
→ 写审计日志
"""


def build_agent_graph() -> StateGraph[AssistantState]:
    """构建 fail-closed Agent Graph。

    流程：START → query_understand → planner → retrieve → grade_and_filter
             → reason → verify → compliance_check → compose → audit_log → END
    条件路由：检索不足则重新检索，验证失败则重新推理，合规拦截仍走 compose。
    """
    graph = StateGraph(AssistantState)

    # 加载会话
    graph.add_node(
        "load_conversation_context",
        _traced_node("load_conversation_context", load_conversation_context),
    )
    # 消解追问
    graph.add_node(
        "resolve_followup_query",
        _traced_node("resolve_followup_query", resolve_followup_query),
    )
    # 查询理解
    graph.add_node("query_understand", _traced_node("query_understand", query_understand))
    # 生成检索计划
    graph.add_node("planner", _traced_node("planner", planner))
    # 执行检索
    graph.add_node("retrieve", _traced_node("retrieve", retrieve))
    # 过滤结果
    graph.add_node("grade_and_filter", _traced_node("grade_and_filter", grade_and_filter))
    # ReAct 推理与工具调用
    graph.add_node(
        "permission_denied_response",
        _traced_node("permission_denied_response", permission_denied_response),
    )
    graph.add_node("reason", _traced_node("reason", reason))
    # 提取引用
    graph.add_node("extract_citations", _traced_node("extract_citations", extract_citations))
    # 四层验证
    graph.add_node("verify", _traced_node("verify", verify))
    # 合规检查
    graph.add_node("compliance_check", _traced_node("compliance_check", compliance_check))
    # 组织回答
    graph.add_node("compose", _traced_node("compose", compose))
    # 保存会话
    graph.add_node(
        "persist_conversation_turn",
        _traced_node("persist_conversation_turn", persist_conversation_turn),
    )
    # 写审计日志
    graph.add_node("audit_log", audit_log)

    # Phase 1：auth_check 由 API Gateway / FastAPI Middleware 承担；
    # Phase 2 可下沉为图内节点。
    graph.add_edge(START, "load_conversation_context")
    graph.add_edge("load_conversation_context", "resolve_followup_query")
    graph.add_edge("resolve_followup_query", "query_understand")
    graph.add_edge("query_understand", "planner")
    graph.add_edge("planner", "retrieve")
    graph.add_edge("retrieve", "grade_and_filter")

    # 条件路由：检索不足则重新规划并补充检索（最多 DEFAULT_MAX_HOPS 次）
    graph.add_conditional_edges(
        "grade_and_filter",
        should_retry_retrieval,
        {
            "continue": "reason",
            "retrieve": "planner",
            "denied": "permission_denied_response",
        },
    )

    graph.add_edge("permission_denied_response", "persist_conversation_turn")
    graph.add_edge("reason", "extract_citations")
    graph.add_edge("extract_citations", "verify")

    # 条件路由：验证失败则重新推理
    graph.add_conditional_edges(
        "verify",
        should_reason_again,
        {
            "continue": "compliance_check",
            "retry": "reason",
        },
    )

    # 条件路由：合规拦截（block 也走 compose，附合规提示）
    graph.add_conditional_edges(
        "compliance_check",
        is_compliant,
        {
            "pass": "compose",
            "block": "compose",
        },
    )

    graph.add_edge("compose", "persist_conversation_turn")
    graph.add_edge("persist_conversation_turn", "audit_log")
    graph.add_edge("audit_log", END)

    return graph


# ══════════════════════════════════════════════════════════════════════
# 6.1 Checkpointer
# ══════════════════════════════════════════════════════════════════════


def build_agent_with_checkpoint() -> CompiledStateGraph[AssistantState]:
    """构建带 Checkpointer 的 Agent Graph"""
    from langgraph.checkpoint.memory import InMemorySaver

    graph = build_agent_graph()
    checkpointer = InMemorySaver()
    return graph.compile(checkpointer=checkpointer)
