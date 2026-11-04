"""answer_status 对外终态映射：优先级固定的五态收敛。"""

from src.schemas.constants import (
    ANSWER_STATUS_ANSWERED,
    ANSWER_STATUS_BLOCKED,
    ANSWER_STATUS_NEEDS_CLARIFICATION,
    ANSWER_STATUS_NO_EVIDENCE,
    ANSWER_STATUS_PERMISSION_DENIED,
    STATE_CLARIFICATION_NEEDED,
    STATE_TERMINAL,
    STATE_VERIFICATION,
    VERIFICATION_ISSUE_NO_RETRIEVAL_RESULTS,
    VERIFICATION_ISSUE_PERMISSION_DENIED,
    VERIFICATION_ISSUE_TOOL_LIMIT_EXCEEDED,
)
from src.utils.answer_status import answer_status_from_state


def test_no_results_maps_to_no_evidence():
    assert answer_status_from_state({
        STATE_TERMINAL: True,
        STATE_VERIFICATION: {"issues": [VERIFICATION_ISSUE_NO_RETRIEVAL_RESULTS]},
    }) == ANSWER_STATUS_NO_EVIDENCE


def test_permission_denied_maps_to_permission_denied():
    assert answer_status_from_state({
        STATE_TERMINAL: True,
        STATE_VERIFICATION: {"issues": [VERIFICATION_ISSUE_PERMISSION_DENIED]},
    }) == ANSWER_STATUS_PERMISSION_DENIED


def test_clarification_maps_to_needs_clarification():
    assert answer_status_from_state({
        STATE_TERMINAL: True,
        STATE_CLARIFICATION_NEEDED: True,
    }) == ANSWER_STATUS_NEEDS_CLARIFICATION


def test_compliance_block_maps_to_blocked():
    assert answer_status_from_state({
        STATE_TERMINAL: True,
        STATE_VERIFICATION: {"passed": True, "issues": []},
        "compliance": {"passed": False},
    }) == ANSWER_STATUS_BLOCKED


def test_verification_failure_maps_to_no_evidence():
    """ISSUE-30：验证 fail-closed（未通过来源/数字验证）≠ 合规拦截，归入 no_evidence。

    审计里 compliance_passed=1、flags 为空时，对外显示"已拦截"会误导用户
    以为违规被合规规则拦截；blocked 只保留给合规层真实拦截的终态。
    """
    assert answer_status_from_state({
        STATE_TERMINAL: True,
        STATE_VERIFICATION: {"passed": False, "issues": ["missing_citation"]},
        "compliance": {"passed": True},
    }) == ANSWER_STATUS_NO_EVIDENCE


def test_tool_limit_exceeded_maps_to_no_evidence():
    """ISSUE-30：工具上限兜底文案同属验证 fail-closed，不冒充合规拦截。"""
    assert answer_status_from_state({
        STATE_TERMINAL: True,
        STATE_VERIFICATION: {
            "passed": False,
            "issues": [VERIFICATION_ISSUE_TOOL_LIMIT_EXCEEDED],
        },
        "compliance": {"passed": True},
    }) == ANSWER_STATUS_NO_EVIDENCE


def test_verification_failure_with_compliance_block_stays_blocked():
    """验证与合规同时失败时 blocked 优先：合规拦截是最严重、最具体的终态。"""
    assert answer_status_from_state({
        STATE_TERMINAL: True,
        STATE_VERIFICATION: {"passed": False, "issues": ["missing_citation"]},
        "compliance": {"passed": False},
    }) == ANSWER_STATUS_BLOCKED


def test_passed_verification_and_compliance_map_to_answered():
    assert answer_status_from_state({
        STATE_TERMINAL: True,
        STATE_VERIFICATION: {"passed": True, "issues": []},
        "compliance": {"passed": True},
    }) == ANSWER_STATUS_ANSWERED


def test_clarification_takes_priority_over_permission_denied():
    """优先级固定：needs_clarification → permission_denied → no_evidence → blocked（合规）→ no_evidence（验证未通过）→ answered。"""
    assert answer_status_from_state({
        STATE_CLARIFICATION_NEEDED: True,
        STATE_VERIFICATION: {"issues": [VERIFICATION_ISSUE_PERMISSION_DENIED]},
    }) == ANSWER_STATUS_NEEDS_CLARIFICATION


def test_permission_denied_takes_priority_over_no_evidence():
    assert answer_status_from_state({
        STATE_VERIFICATION: {
            "issues": [
                VERIFICATION_ISSUE_NO_RETRIEVAL_RESULTS,
                VERIFICATION_ISSUE_PERMISSION_DENIED,
            ],
        },
    }) == ANSWER_STATUS_PERMISSION_DENIED


def test_unverifiable_state_fails_closed_to_blocked():
    """缺验证/合规信息的状态不得映射成 answered（fail closed）。"""
    assert answer_status_from_state({
        STATE_TERMINAL: True,
    }) == ANSWER_STATUS_BLOCKED
