"""answer_status 对外终态映射：把图执行终态收敛成固定的五态对外契约。

优先级固定为：needs_clarification → permission_denied → no_evidence（无检索
结果）→ blocked（合规拦截）→ no_evidence（验证未通过 fail-closed）→ answered。
ISSUE-30：验证未通过说明没有可安全返回的依据，语义上是"无可靠证据"而非
"违规被拦"——此前该终态被归入 blocked，审计里 compliance_passed=1、flags
为空，对外却显示"已拦截"，误导用户以为违规被合规规则拦截。blocked 只保留
给合规层真正拦截的终态。缓存命中只缓存已通过验证与合规的成功答案，因此
命中路径在构造点显式填充 answered，不经过本函数。
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
    if not compliance.get("passed", False):
        return ANSWER_STATUS_BLOCKED
    if not verification.get("passed", False):
        # ISSUE-30：验证 fail-closed 的兜底终态（未通过来源/数字验证、工具上限）
        # 不携带"违规被拦"语义，归入 no_evidence；合规层真实拦截才映射 blocked
        return ANSWER_STATUS_NO_EVIDENCE
    return ANSWER_STATUS_ANSWERED
