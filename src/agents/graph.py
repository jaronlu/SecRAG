"""Agent Graph 构建——节点编排、条件路由、Checkpointer"""

import logging
import time
from typing import Any, Final, Literal, Protocol

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from src.agents.nodes import (
    audit_log,
    authorize_reason_tool_call,
    call_reason_model,
    clarify,
    compliance_check,
    compose,
    extract_citations,
    finalize_reason,
    grade_and_filter,
    load_conversation_context,
    no_results_response,
    permission_denied_response,
    persist_conversation_turn,
    planner,
    prepare_reason,
    query_understand,
    record_tool_results,
    resolve_followup_query,
    retrieve,
    route_reason_model,
    tool_limit_response,
    verify,
)
from src.agents.state import AssistantState
from src.agents.tools import tools
from src.config import config
from src.schemas.constants import (
    CONFIDENCE_HIGH_MIN_RESULTS,
    DEFAULT_MAX_HOPS,
    MAX_REASON_ATTEMPTS,
    STATE_AMBIGUITY,
    STATE_COMPLIANCE,
    STATE_INTERMEDIATE_STEPS,
    STATE_REASON_ATTEMPTS,
    STATE_RETRIEVAL_ATTEMPTS,
    STATE_RETRIEVAL_FILTERED_CHUNKS,
    STATE_RETRIEVAL_RESULTS,
    STATE_RERANKER_STATUS,
    STATE_VERIFICATION,
)
from src.schemas.typed_dicts import IntermediateStep
from src.utils.langfuse_adapter import start_node_span


logger = logging.getLogger(__name__)


class _AgentNode(Protocol):
    def __call__(self, state: AssistantState) -> dict[str, Any]: ...


# SSE progress 事件面向客户端的节点集合。由图模块声明：新增或重命名节点
# 只需改这里；传输层只转发，不解释节点语义，更不据此推断业务终态——
# 终态以节点产出 final_answer 为准。
CLIENT_PROGRESS_NODES: Final[frozenset[str]] = frozenset(
    {
        "query_understand",
        "planner",
        "retrieve",
        "grade_and_filter",
        "reason",
        "verify",
        "compose",
    }
)


def _node_execution_succeeded(result: dict[str, Any]) -> bool:
    """节点正常返回只说明没有抛异常，不代表节点职责执行成功；
    返回字典中的显式执行失败标记（重排 error 降级）视为失败，
    而 "unavailable" 是环境未配置重排器，不算执行失败。"""
    reranker_status = result.get(STATE_RERANKER_STATUS)
    if isinstance(reranker_status, str) and reranker_status.startswith("error:"):
        return False
    return True


def _node_span_metadata(
    name: str,
    state: AssistantState,
    result: dict[str, Any],
) -> dict[str, Any]:
    """按节点提取 Langfuse span 的白名单 metadata（adapter 过滤兜底）。

    只放标量业务结果：模型名、检索数量、重试次数、验证/合规结果；
    查询词、文档内容、引用原文、回答原文一律不入 payload。
    """
    metadata: dict[str, Any] = {"node_name": name}
    if name == "retrieve":
        # 本轮新增 chunk 数 = 累计结果 - 进入节点前的结果
        before = len(state.get(STATE_RETRIEVAL_RESULTS, []) or [])
        after = len(result.get(STATE_RETRIEVAL_RESULTS, []) or [])
        metadata["retrieval_count"] = max(after - before, 0)
        # 多跳重试次数：本轮结束后的检索轮次（1 = 首轮）
        metadata["retry_count"] = result.get(STATE_RETRIEVAL_ATTEMPTS, 0)
    elif name == "grade_and_filter":
        metadata["retrieval_count"] = result.get(STATE_RETRIEVAL_FILTERED_CHUNKS, 0)
    elif name in ("verify", "permission_denied_response"):
        verification = result.get(STATE_VERIFICATION, {})
        metadata["verification_status"] = (
            "passed" if verification.get("passed") else "failed"
        )
    elif name == "compliance_check":
        compliance = result.get(STATE_COMPLIANCE, {})
        metadata["compliance_status"] = (
            "passed" if compliance.get("passed") else "blocked"
        )
    elif name == "compose":
        # 终态编排：验证/合规结果取自 state（含被安全文案降级的终局）
        metadata["verification_status"] = (
            "passed" if state.get(STATE_VERIFICATION, {}).get("passed") else "failed"
        )
        metadata["compliance_status"] = (
            "passed" if state.get(STATE_COMPLIANCE, {}).get("passed") else "blocked"
        )
    elif name in ("query_understand", "planner"):
        # LLM 调用节点：模型名来自全局配置（节点共享同一 LLM 实例）
        metadata["model_name"] = config.llm.model
    return metadata


def _traced_node(
    name: str,
    node: _AgentNode,
) -> _AgentNode:
    """Record the actual node path and elapsed time in state.

    节点抛异常时 state 更新会被 LangGraph 丢弃，无法在 intermediate_steps
    里留下 success=False 记录，只能记入应用日志后原样抛出。

    同一包装器内创建 Langfuse 节点 span（未启用/未采样时为 no-op），
    避免逐节点重复插桩；ReAct 子图节点在 nodes.py 内手动补 span。
    """

    def wrapped(state: AssistantState) -> dict[str, Any]:
        started = time.perf_counter()
        span = start_node_span(name, metadata={"node_name": name})
        try:
            result = node(state)
        except Exception as exc:
            span.finish(
                status="error",
                error_type=type(exc).__name__,
                metadata={"duration_ms": max((time.perf_counter() - started) * 1000, 0.0)},
            )
            logger.exception(
                "Node %s raised after %.1f ms",
                name,
                max((time.perf_counter() - started) * 1000, 0.0),
            )
            raise
        duration_ms = max((time.perf_counter() - started) * 1000, 0.0)
        succeeded = _node_execution_succeeded(result)
        span_metadata = _node_span_metadata(name, state, result)
        span_metadata["duration_ms"] = duration_ms
        # 重排 error 降级是显式执行失败（见 _node_execution_succeeded），
        # span 状态与之对齐；固定标签，不透出异常消息内容
        span.finish(
            status="ok" if succeeded else "error",
            error_type=None if succeeded else "reranker_error",
            metadata=span_metadata,
        )
        step: IntermediateStep = {
            "step": name,
            "duration_ms": duration_ms,
            "success": succeeded,
        }
        return {
            **result,
            STATE_INTERMEDIATE_STEPS: state.get(STATE_INTERMEDIATE_STEPS, []) + [step],
        }

    return wrapped


# ══════════════════════════════════════════════════════════════════════
# 5.1 条件路由函数
# ══════════════════════════════════════════════════════════════════════


def should_retry_retrieval(
    state: AssistantState,
) -> Literal["denied", "continue", "retrieve", "no_results"]:
    """判断是否需要补充检索（最多 DEFAULT_MAX_HOPS 次，计数器由 retrieve 节点维护）。

    重试轮次耗尽仍无任何可用结果时走 no_results 短路（ISSUE-3）：
    不再进入 reason 让模型无证据作答，直接返回"未找到资料"。
    """
    attempts = state.get(STATE_RETRIEVAL_ATTEMPTS, 0)
    results = state.get(STATE_RETRIEVAL_RESULTS, [])
    usable = [result for result in results if not result.get("denied")]
    if results and not usable:
        return "denied"
    if attempts >= DEFAULT_MAX_HOPS:
        return "continue" if usable else "no_results"

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


def should_clarify(state: AssistantState) -> Literal["clarify", "continue"]:
    """P1-8: 判断查询是否存在歧义，需要用户澄清。

    仅当 ambiguity 非空时触发澄清；LLM 被指示仅在歧义显著时返回。
    """
    ambiguities = state.get(STATE_AMBIGUITY, [])
    if ambiguities and len(ambiguities) > 0:
        return "clarify"
    return "continue"


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


def build_reason_subgraph() -> CompiledStateGraph[AssistantState]:
    """Compile the checkpoint-aware ReAct loop once per outer graph build."""
    graph = StateGraph(AssistantState)
    graph.add_node("prepare_reason", prepare_reason)
    graph.add_node("call_reason_model", call_reason_model)
    graph.add_node(
        "execute_reason_tools",
        ToolNode(tools, wrap_tool_call=authorize_reason_tool_call),
    )
    graph.add_node("record_tool_results", record_tool_results)
    graph.add_node("finalize_reason", finalize_reason)
    graph.add_node("tool_limit_response", tool_limit_response)

    graph.add_edge(START, "prepare_reason")
    graph.add_edge("prepare_reason", "call_reason_model")
    graph.add_conditional_edges(
        "call_reason_model",
        route_reason_model,
        {
            "tools": "execute_reason_tools",
            "finalize": "finalize_reason",
            "limit": "tool_limit_response",
        },
    )
    graph.add_edge("execute_reason_tools", "record_tool_results")
    graph.add_edge("record_tool_results", "call_reason_model")
    graph.add_edge("finalize_reason", END)
    graph.add_edge("tool_limit_response", END)
    return graph.compile()


def build_agent_graph() -> StateGraph[AssistantState]:
    """构建 fail-closed Agent Graph。

    流程：START → query_understand → planner → retrieve → grade_and_filter
             → reason → verify → compliance_check → compose → audit_log → END
    条件路由：检索不足则重新检索，验证失败则重新推理，合规拦截仍走 compose。
    """
    graph = StateGraph(AssistantState)
    reason_subgraph = build_reason_subgraph()

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
    # P1-8: 歧义澄清节点
    graph.add_node("clarify", _traced_node("clarify", clarify))
    # 生成检索计划
    graph.add_node("planner", _traced_node("planner", planner))
    # 执行检索
    graph.add_node("retrieve", _traced_node("retrieve", retrieve))
    # 过滤结果
    graph.add_node("grade_and_filter", _traced_node("grade_and_filter", grade_and_filter))
    # 全部结果被拒时提前终止
    graph.add_node(
        "permission_denied_response",
        _traced_node("permission_denied_response", permission_denied_response),
    )
    # ISSUE-3: 检索耗尽仍 0 结果时提前终止，不进入无证据推理
    graph.add_node(
        "no_results_response",
        _traced_node("no_results_response", no_results_response),
    )
    # ReAct 推理与工具调用
    graph.add_node("reason", reason_subgraph)
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
    # P1-8: 歧义检测——有歧义则澄清，否则继续检索计划
    graph.add_conditional_edges(
        "query_understand",
        should_clarify,
        {
            "continue": "planner",
            "clarify": "clarify",
        },
    )
    graph.add_edge("planner", "retrieve")
    graph.add_edge("retrieve", "grade_and_filter")

    # 条件路由：检索不足则重新规划并补充检索（最多 DEFAULT_MAX_HOPS 次）；
    # 耗尽仍无可用结果则短路返回"未找到资料"（ISSUE-3）
    graph.add_conditional_edges(
        "grade_and_filter",
        should_retry_retrieval,
        {
            "continue": "reason",
            "retrieve": "planner",
            "denied": "permission_denied_response",
            "no_results": "no_results_response",
        },
    )

    graph.add_edge("permission_denied_response", "persist_conversation_turn")
    graph.add_edge("no_results_response", "persist_conversation_turn")
    # P1-8: 澄清节点直接进入会话保存（跳过检索/推理/验证）
    graph.add_edge("clarify", "persist_conversation_turn")
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
