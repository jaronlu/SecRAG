"""SSE 事件契约：前端收敛后的消费面（plan.md Task 3/7）。

产品收敛后问答页只走流式主流程：消费 answer_delta/answer/error/done，
不渲染内部节点进度（progress 事件后端可继续发送，前端必须忽略）。
历史 ISSUE-5 的 STREAM_NODES 进度条已随 StreamingProgress 组件删除，
前端不再跟踪后端节点名——这里守护的是"不再跟踪"这一新契约本身。
"""

from pathlib import Path

from src.agents.graph import CLIENT_PROGRESS_NODES

FRONTEND_TYPES_PATH = Path(__file__).resolve().parents[1] / "frontend" / "src" / "types.ts"
FRONTEND_CHAT_PAGE_PATH = (
    Path(__file__).resolve().parents[1] / "frontend" / "src" / "pages" / "ChatPage.tsx"
)


def _frontend_types_text() -> str:
    return FRONTEND_TYPES_PATH.read_text(encoding="utf-8")


def test_frontend_no_longer_tracks_internal_progress_nodes():
    """前端不得再定义/消费内部节点名（STREAM_NODES 已删除）。"""
    text = _frontend_types_text()
    assert "STREAM_NODES" not in text
    assert "currentNode" not in text


def test_frontend_progress_nodes_are_server_side_only():
    """后端仍声明 CLIENT_PROGRESS_NODES 供审计/观测使用，但前端不引用。"""
    assert CLIENT_PROGRESS_NODES  # 服务端进度集合仍存在（审计/调试入口）
    assert "StreamingProgress" not in _frontend_types_text()


def test_frontend_stream_event_types_declare_answer_delta():
    """ISSUE-9：SSE 新增 answer_delta 事件，前端类型必须先于消费端声明。"""
    text = _frontend_types_text()
    assert "'answer_delta'" in text


def test_frontend_stream_event_types_declare_answer_status():
    """answer_status 随 SSE answer 终态事件到达，前端类型必须声明五个取值。"""
    text = _frontend_types_text()
    for status in ("answered", "no_evidence", "permission_denied", "needs_clarification", "blocked"):
        assert f"'{status}'" in text
    assert "answer_status" in text


def test_frontend_chat_page_consumes_answer_delta_without_fake_typewriter():
    """ISSUE-9：ChatPage 消费真流增量；拿到全文再假打字机的渲染必须移除。"""
    text = FRONTEND_CHAT_PAGE_PATH.read_text(encoding="utf-8")
    assert "answer_delta" in text
    assert "setInterval" not in text
