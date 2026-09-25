"""全链路测试案例：环节 D QA 问答含 SSE（TC-016~TC-023）。

TC-016/018 为 Graph 级 E2E：真实 Agent Graph + 真实验证器/合规器 +
隔离存储，仅 LLM 与向量检索为测试替身。
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from src.api.auth import AuthenticatedUser, authenticate_user
from src.api.main import app
from src.schemas.constants import (
    API_ROUTE_ASSISTANT_QA,
    API_ROUTE_ASSISTANT_QA_STREAM,
    AUDIT_REQUEST_ID,
    CONFIDENCE_LOW,
    CONFIDENCE_MEDIUM,
    ROLE_ADVISOR,
    STATE_CITATIONS,
    STATE_COMPLIANCE,
    STATE_CONFIDENCE,
    STATE_FINAL_ANSWER,
    STATE_REASON_ATTEMPTS,
    STATE_RETRIEVAL_RESULTS,
    STATE_THREAD_ID,
    STATE_VERIFICATION,
    STATE_AUDIT_TRAIL,
    STATE_USER_ID,
)


# ══════════════════════════════════════════════════════════════════════
# TC-016 QA 正常全链路（Graph 级 E2E）
# ══════════════════════════════════════════════════════════════════════


def test_tc016_qa_full_chain_happy_path(run_agent_graph, isolated_stores, fake_llm):
    """TC-016：查询 → 检索 → 推理 → 验证 → 合规 → 会话/审计落库 全链路成功。

    真实链路中 API 层先 ensure 会话再调图；这里同样先建会话再执行。
    """
    from tests.e2e.conftest import build_state

    thread = isolated_stores.conversation.create_thread(
        user_id="user_advisor", user_role=ROLE_ADVISOR, client_id=None, title="TC-016"
    )
    state = run_agent_graph(build_state(thread_id=thread["thread_id"]))

    # 终态答案：结构化 Markdown + [来源1] 引用
    assert state[STATE_FINAL_ANSWER].startswith("## 结论")
    assert "[来源1]" in state[STATE_FINAL_ANSWER]
    assert state[STATE_CITATIONS], "成功链路必须产出引用"
    assert state[STATE_CITATIONS][0]["source"].endswith("xx_money_fund_2024.html")
    assert state[STATE_CONFIDENCE] in (CONFIDENCE_MEDIUM, "high")

    # 验证与合规均通过
    assert state[STATE_VERIFICATION]["passed"] is True
    assert state[STATE_COMPLIANCE]["passed"] is True

    # 会话落库：同一用户可见本轮问答
    messages = isolated_stores.conversation.list_messages(
        thread_id=state[STATE_THREAD_ID], user_id=state[STATE_USER_ID]
    )
    roles = [m["role"] for m in messages]
    assert "assistant" in roles
    assistant_content = " ".join(
        m["content"] for m in messages if m["role"] == "assistant"
    )
    assert "风险等级" in assistant_content

    # 审计落库：按 request_id 可取完整审计条目
    request_id = state[STATE_AUDIT_TRAIL][AUDIT_REQUEST_ID]
    trail = isolated_stores.audit.get_by_request_id(request_id)
    assert trail is not None
    assert trail["query"]["original"] == "XX货币市场基金的风险等级是什么？"
    assert trail["compliance"]["passed"] is True
    assert trail["reasoning"]["execution_path"][-1] == "audit_log"


# ══════════════════════════════════════════════════════════════════════
# TC-018 验证失败重试与安全兜底（占位，后续提交）
# ══════════════════════════════════════════════════════════════════════
