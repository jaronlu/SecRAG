"""工具超时与请求截止时间测试（issues.md 一.8）。"""

import time
from typing import cast

from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest

import src.agents.nodes as nodes
from src.agents.nodes import authorize_reason_tool_call
from src.schemas.constants import (
    STATE_REQUEST_DEADLINE,
    TOOL_TIMEOUT_SECONDS,
)


class _FakeTool:
    name = "fake_tool"


class _FakeRequest:
    def __init__(self, state=None):
        self.state = state or {}
        self.tool_call = {"name": "fake_tool", "id": "call_1"}


def _allow_fake_tool(monkeypatch):
    monkeypatch.setattr(nodes, "_reason_tools", lambda state: [_FakeTool()])


def test_tool_timeout_returns_promptly(monkeypatch):
    """超时上限应约束包装函数整体返回时间，而不是任务实际耗时。

    旧实现在 with ThreadPoolExecutor 上下文内等待 future.result()，
    上下文退出 join 残留线程，把等待拉长为任务耗时。
    """
    _allow_fake_tool(monkeypatch)
    started = time.monotonic()

    def slow_execute(request):
        time.sleep(TOOL_TIMEOUT_SECONDS + 0.5)
        return ToolMessage(content="done", name="fake_tool", tool_call_id="call_1")

    result = authorize_reason_tool_call(cast(ToolCallRequest, _FakeRequest()), slow_execute)
    elapsed = time.monotonic() - started

    # 返回的是错误 ToolMessage，且等待时间接近超时上限而非任务耗时
    assert isinstance(result, ToolMessage)
    assert "超时" in result.content
    assert elapsed < TOOL_TIMEOUT_SECONDS + 0.5


def test_request_deadline_blocks_tool_execution(monkeypatch):
    _allow_fake_tool(monkeypatch)

    def execute(request):  # pragma: no cover - 不应被调用
        raise AssertionError("deadline exceeded 后不应执行工具")

    state = {STATE_REQUEST_DEADLINE: time.monotonic() - 1}
    result = authorize_reason_tool_call(cast(ToolCallRequest, _FakeRequest(state)), execute)
    assert isinstance(result, ToolMessage)
    assert "超时" in result.content
    assert result.status == "error"
