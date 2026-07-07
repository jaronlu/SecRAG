"""Agent Graph 构建——节点编排、条件路由、Checkpointer"""

from langgraph.graph import END, START, StateGraph

from src.agents.nodes import (
    audit_log,
    compliance_check,
    compose,
    grade_and_filter,
    planner,
    query_understand,
    reason,
    retrieve,
    verify,
)
from src.agents.state import AssistantState
from src.schemas.constants import (
    DEFAULT_MAX_HOPS,
    MAX_REASON_ATTEMPTS,
    RR_SCORE,
    STATE_COMPLIANCE,
    STATE_REASON_ATTEMPTS,
    STATE_RETRIEVAL_ATTEMPTS,
    STATE_RETRIEVAL_RESULTS,
    STATE_VERIFICATION,
)


# ══════════════════════════════════════════════════════════════════════
# 5.1 条件路由函数
# ══════════════════════════════════════════════════════════════════════


def should_retry_retrieval(state: AssistantState) -> str:
    """判断是否需要补充检索（最多 DEFAULT_MAX_HOPS 次，计数器由 retrieve 节点维护）"""
    attempts = state.get(STATE_RETRIEVAL_ATTEMPTS, 0)
    if attempts >= DEFAULT_MAX_HOPS:
        return "continue"

    results = state.get(STATE_RETRIEVAL_RESULTS, [])
    if not results:
        return "retrieve"
    if results[0].get(RR_SCORE, 0) < 0.5:
        return "retrieve"
    return "continue"


def should_reason_again(state: AssistantState) -> str:
    """判断验证是否通过；失败重推受 MAX_REASON_ATTEMPTS 显式限制。"""
    verification = state.get(STATE_VERIFICATION, {})
    attempts = state.get(STATE_REASON_ATTEMPTS, 0)
    if not verification.get("passed", False) and attempts < MAX_REASON_ATTEMPTS:
        return "retry"
    return "continue"


def is_compliant(state: AssistantState) -> str:
    """判断是否通过合规检查"""
    compliance = state.get(STATE_COMPLIANCE, {})
    if compliance.get("passed", False):
        return "pass"
    return "block"


# ══════════════════════════════════════════════════════════════════════
# 5.2 Graph 定义
# ══════════════════════════════════════════════════════════════════════


def build_agent_graph() -> StateGraph:
    """构建 9 节点 Agent Graph

    流程：START → query_understand → planner → retrieve → grade_and_filter
             → reason → verify → compliance_check → compose → audit_log → END
    条件路由：检索不足则重新检索，验证失败则重新推理，合规拦截仍走 compose。
    """
    graph = StateGraph(AssistantState)

    # 添加节点
    graph.add_node("query_understand", query_understand)
    graph.add_node("planner", planner)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade_and_filter", grade_and_filter)
    graph.add_node("reason", reason)
    graph.add_node("verify", verify)
    graph.add_node("compliance_check", compliance_check)
    graph.add_node("compose", compose)
    graph.add_node("audit_log", audit_log)

    # Phase 1：auth_check 由 API Gateway / FastAPI Middleware 承担；
    # Phase 2 可下沉为图内节点。
    graph.add_edge(START, "query_understand")
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
        },
    )

    graph.add_edge("reason", "verify")

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

    graph.add_edge("compose", "audit_log")
    graph.add_edge("audit_log", END)

    return graph


# ══════════════════════════════════════════════════════════════════════
# 6.1 Checkpointer
# ══════════════════════════════════════════════════════════════════════


def build_agent_with_checkpoint():
    """构建带 Checkpointer 的 Agent Graph"""
    from langgraph.checkpoint.memory import InMemorySaver

    graph = build_agent_graph()
    checkpointer = InMemorySaver()
    return graph.compile(checkpointer=checkpointer)
