"""answer_status 对外终态映射：把图执行终态收敛成固定的五态对外契约。

优先级固定为：needs_clarification → permission_denied → no_evidence →
blocked → answered。缓存命中只缓存已通过验证与合规的成功答案，因此命中
路径在构造点显式填充 answered，不经过本函数。
"""

from typing import Any, Mapping

from src.schemas.constants import (
    ANSWER_STATUS_ANSWERED,
    ANSWER_STATUS_BLOCKED,
    ANSWER_STATUS_NEEDS_CLARIFICATION,
    ANSWER_STATUS_NO_EVIDENCE,
    ANSWER_STATUS_PERMISSION_DENIED,
    AnswerStatus,
    STATE_CLARIFICATION_NEEDED,
    STATE_COMPLIANCE,
    STATE_VERIFICATION,
    VERIFICATION_ISSUE_NO_RETRIEVAL_RESULTS,
    VERIFICATION_ISSUE_PERMISSION_DENIED,
)

# verification issues 中表示"没有可用结果"的标记集合
_NO_EVIDENCE_ISSUES = frozenset({VERIFICATION_ISSUE_NO_RETRIEVAL_RESULTS})
# verification issues 中表示"权限不足"的标记集合
_PERMISSION_ISSUES = frozenset({VERIFICATION_ISSUE_PERMISSION_DENIED})


def answer_status_from_state(state: Mapping[str, Any]) -> AnswerStatus:
    """从图终态解析对外 answer_status，优先级固定、缺信息 fail closed 为 blocked。"""
    if state.get(STATE_CLARIFICATION_NEEDED):
        return ANSWER_STATUS_NEEDS_CLARIFICATION

    issues = state.get(STATE_VERIFICATION, {}).get("issues", [])
    if any(issue in _PERMISSION_ISSUES for issue in issues):
        return ANSWER_STATUS_PERMISSION_DENIED
    if any(issue in _NO_EVIDENCE_ISSUES for issue in issues):
        return ANSWER_STATUS_NO_EVIDENCE

    verification = state.get(STATE_VERIFICATION, {})
    compliance = state.get(STATE_COMPLIANCE, {})
    if not verification.get("passed", False) or not compliance.get("passed", False):
        return ANSWER_STATUS_BLOCKED
    return ANSWER_STATUS_ANSWERED
