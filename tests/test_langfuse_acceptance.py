"""Langfuse 验收测试——API 入口层。

逐条覆盖验收点（与实现自带单测 tests/test_langfuse_adapter.py、
tests/test_langfuse_wiring.py 互补，断言口径为验收语义）：
- Langfuse 关闭时问答返回正确业务结果、RunnableConfig 无 callbacks、
  零 Langfuse 调用与零计数（no-op 断言）；
- callbacks 从 API 请求注入 RunnableConfig 并绑定根 trace；
- payload 卫生：金丝雀标记（原始问题/完整回答/chunk 文本/SQL/客户 ID/持仓）
  无法经 metadata 白名单或内容属性进入上送 payload；
- 模拟网络超时/鉴权失败/写入异常：业务结果不变、失败计数按 reason 增加、
  本地日志可见；
- trace request_id 与 SQLite 审计共用，payload 不暴露用户身份；
- Langfuse 不可用时失败计数与本地告警生效。

替身模式与 tests/test_langfuse_wiring.py 一致：不连真实服务，client 依赖
注入 fake；每次用例独立 MetricsRegistry，避免全局计数污染。
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

import pytest
from fastapi import HTTPException
from pydantic import SecretStr

from src.api.auth import AuthenticatedUser
from src.api.main import assistant_qa
from src.config import LangfuseConfig
from src.schemas.constants import (
    AUDIT_REQUEST_ID,
    ROLE_TECHNICAL,
    STATE_AUDIT_TRAIL,
    STATE_COMPLIANCE,
    STATE_CONFIDENCE,
    STATE_CITATIONS,
    STATE_FINAL_ANSWER,
    STATE_VERIFICATION,
)
from src.schemas.request_response import AssistantQARequest
from src.utils.langfuse_adapter import (
    ROOT_TRACE_NAME,
    LangfuseAdapter,
    LangfuseTraceMetadata,
)
from src.utils.metrics import MetricsRegistry

# 金丝雀标记：出现在任何上送 payload 中即为隐私违规。
# 数字串仅用于 adapter 级测试（不进图流程，避免触发业务 PII 拦截）。
CANARY_QUESTION = "CANARY-QUESTION-MARKER"
CANARY_ANSWER = "CANARY-ANSWER-MARKER"
CANARY_CHUNK = "CANARY-CHUNK-MARKER"
CANARY_SQL = "SELECT * FROM canary_holdings WHERE client_id='CANARY-CUST-001'"
CANARY_CUSTOMER_ID = "CANARY-CUST-001"
CANARY_POSITIONS = '[{"security": "CANARY-SEC-001", "quantity": 8888}]'
CANARY_PHONE = "13800001111"
CANARY_EMAIL = "canary-owner@example.com"

EXPECTED_ANSWER = "货币基金风险等级为R1（低风险），流动性好。"
EXPECTED_CITATIONS: list[dict[str, Any]] = [{"source": "report"}]
EXPECTED_CONFIDENCE = "high"
EXPECTED_COMPLIANCE = {"passed": True, "risk_disclosure": ""}

USER_IDENTITY_MARKERS = ("user_accept", "wealth", ROLE_TECHNICAL)


def _enabled_cfg(**overrides: Any) -> LangfuseConfig:
    defaults: dict[str, Any] = {
        "enabled": True,
        "host": "https://cloud.langfuse.com",
        "public_key": f"pk-lf-{uuid.uuid4().hex}",
        "secret_key": f"sk-lf-{uuid.uuid4().hex}",
        "sample_rate": 1.0,
        "capture_content": False,
    }
    defaults.update(overrides)
    return LangfuseConfig(**defaults)


def _disabled_cfg() -> LangfuseConfig:
    return LangfuseConfig(
        enabled=False, host="", public_key="", secret_key=SecretStr("")
    )


class FakeLangfuseSpan:
    def __init__(self) -> None:
        self.trace_id = uuid.uuid4().hex
        self.id = uuid.uuid4().hex[:16]
        self.updates: list[dict[str, Any]] = []
        self.ended = False
        self.fail_update = False

    def update(self, **kwargs: Any) -> None:
        if self.fail_update:
            raise TimeoutError("langfuse span update timed out")
        self.updates.append(kwargs)

    def end(self, **kwargs: Any) -> None:
        self.ended = True


class FakeLangfuseClient:
    """假 client：记录 start_observation；fail_message 非空时模拟不可用。"""

    def __init__(self, fail_message: str | None = None) -> None:
        self.started: list[tuple[FakeLangfuseSpan, dict[str, Any]]] = []
        self.fail_message = fail_message
        self.flush_called = False

    def start_observation(self, **kwargs: Any) -> FakeLangfuseSpan:
        if self.fail_message is not None:
            raise RuntimeError(self.fail_message)
        span = FakeLangfuseSpan()
        self.started.append((span, kwargs))
        return span

    def flush(self) -> None:
        self.flush_called = True


class FakeCallbackHandler:
    """替身：记录绑定到根 trace 的 trace_context。"""

    def __init__(self, *, trace_context: dict[str, str] | None = None) -> None:
        self.trace_context = trace_context


class _StubAgentApp:
    """返回固定业务结果的 agent 替身；记录收到的 state 与 RunnableConfig。"""

    def __init__(self) -> None:
        self.config: dict[str, Any] | None = None
        self.state: dict[str, Any] | None = None

    def invoke(self, state: dict[str, Any], config: Any = None) -> dict[str, Any]:
        self.config = config
        self.state = state
        return {
            **state,
            STATE_FINAL_ANSWER: EXPECTED_ANSWER,
            STATE_CITATIONS: EXPECTED_CITATIONS,
            STATE_CONFIDENCE: EXPECTED_CONFIDENCE,
            STATE_COMPLIANCE: EXPECTED_COMPLIANCE,
            STATE_VERIFICATION: {"passed": True, "confidence": "high"},
        }


class _RaisingAgentApp:
    def invoke(self, state: dict[str, Any], config: Any = None) -> dict[str, Any]:
        raise ValueError("bad state")


def _adapter(
    cfg: LangfuseConfig, client: FakeLangfuseClient | None = None
) -> LangfuseAdapter:
    return LangfuseAdapter(
        cfg=cfg,
        app_env="development",
        client=client if client is not None else FakeLangfuseClient(),
        metrics=MetricsRegistry(),
    )


def _qa_request() -> AssistantQARequest:
    return AssistantQARequest(query=EXPECTED_ANSWER)


def _wire_api(
    monkeypatch: pytest.MonkeyPatch,
    adapter: LangfuseAdapter,
    agent: Any,
    tmp_path: Any,
) -> None:
    from src.utils.conversation import SQLiteConversationStore

    store = SQLiteConversationStore(tmp_path / "conversations.db")
    monkeypatch.setattr("src.api.main._get_conversation_store", lambda: store)
    monkeypatch.setattr("src.api.main.get_langfuse", lambda: adapter)
    monkeypatch.setattr("src.api.main._get_agent_app", lambda: agent)
    monkeypatch.setattr(
        "langfuse.langchain.CallbackHandler", FakeCallbackHandler, raising=True
    )


def _serialized_adapter_payloads(adapter: LangfuseAdapter, client: FakeLangfuseClient) -> str:
    """序列化 fake client 捕获到的全部上送内容（span 参数 + 后续 update）。"""
    chunks = [json.dumps(kwargs, ensure_ascii=False, default=str) for _, kwargs in client.started]
    chunks.extend(
        json.dumps(update, ensure_ascii=False, default=str)
        for span, _ in client.started
        for update in span.updates
    )
    return "\n".join(chunks)


# ─────────────── 验收点 1：Langfuse 关闭时问答行为不变 ───────────────


@pytest.mark.asyncio
async def test_disabled_langfuse_qa_returns_correct_business_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
):
    client = FakeLangfuseClient()
    adapter = _adapter(_disabled_cfg(), client)
    agent = _StubAgentApp()
    _wire_api(monkeypatch, adapter, agent, tmp_path)

    response = await assistant_qa(
        _qa_request(),
        AuthenticatedUser("user_accept_disabled", ROLE_TECHNICAL, "tech"),
    )

    # 业务结果与替身产出逐字段一致：问答行为不变
    assert response.answer == EXPECTED_ANSWER
    assert response.citations == EXPECTED_CITATIONS
    assert response.confidence == EXPECTED_CONFIDENCE
    assert response.compliance == EXPECTED_COMPLIANCE
    assert response.thread_id and response.turn_id

    # no-op 断言：RunnableConfig 无 callbacks 键、零 Langfuse 调用、零计数
    assert "callbacks" not in (agent.config or {})
    assert agent.config is not None and agent.config["recursion_limit"] > 0
    assert client.started == []
    assert client.flush_called is False
    assert adapter._metrics.langfuse_export_errors_total.collect() == []
    assert adapter._metrics.langfuse_dropped_total.collect() == []


# ─────────────── 验收点 2：callback 从 API 请求注入 ───────────────


@pytest.mark.asyncio
async def test_qa_injects_callback_bound_to_root_trace_into_runnable_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
):
    client = FakeLangfuseClient()
    adapter = _adapter(_enabled_cfg(), client)
    agent = _StubAgentApp()
    _wire_api(monkeypatch, adapter, agent, tmp_path)

    response = await assistant_qa(
        _qa_request(),
        AuthenticatedUser("user_accept_cb", ROLE_TECHNICAL, "tech"),
    )

    assert response.answer == EXPECTED_ANSWER
    callbacks = (agent.config or {}).get("callbacks")
    assert callbacks and len(callbacks) == 1
    assert isinstance(callbacks[0], FakeCallbackHandler)
    root_span, root_kwargs = client.started[0]
    assert root_kwargs["name"] == ROOT_TRACE_NAME
    # handler 绑定根 trace：trace_id/parent_span_id 与根 span 一致
    assert callbacks[0].trace_context == {
        "trace_id": root_span.trace_id,
        "parent_span_id": root_span.id,
    }
    assert root_span.ended is True


# ─────────────── 验收点 3：payload 卫生（金丝雀） ───────────────


def test_metadata_whitelist_blocks_all_canary_payloads():
    adapter = _adapter(_enabled_cfg())

    filtered = adapter.filter_metadata({
        "query": CANARY_QUESTION,
        "question": CANARY_QUESTION,
        "answer": CANARY_ANSWER,
        "final_answer": CANARY_ANSWER,
        "chunks": [CANARY_CHUNK],
        "chunk_text": CANARY_CHUNK,
        "sql": CANARY_SQL,
        "customer_id": CANARY_CUSTOMER_ID,
        "positions": CANARY_POSITIONS,
        "user_id": CANARY_CUSTOMER_ID,
        "contact": f"{CANARY_PHONE} {CANARY_EMAIL}",
        "model_name": "glm-4",
    })

    # 白名单内字段保留，其余（含全部金丝雀）一律不出现
    assert filtered == {"model_name": "glm-4"}
    serialized = json.dumps(filtered, ensure_ascii=False)
    for canary in (
        CANARY_QUESTION,
        CANARY_ANSWER,
        CANARY_CHUNK,
        CANARY_SQL,
        CANARY_CUSTOMER_ID,
        CANARY_POSITIONS,
        CANARY_PHONE,
        CANARY_EMAIL,
    ):
        assert canary not in serialized


def test_mask_deletes_canary_content_attributes_before_upload():
    """langchain handler 会把 input/output 原文挂到内容属性上；
    默认配置下 mask 必须整体删除，任何金丝雀都不得上送。"""
    adapter = _adapter(_enabled_cfg())
    span = _FakeOtelSpan({
        "langfuse.trace.input": CANARY_QUESTION,
        "langfuse.trace.output": CANARY_ANSWER,
        "langfuse.observation.input": CANARY_CHUNK,
        "langfuse.observation.output": f"{CANARY_SQL} {CANARY_POSITIONS}",
        "gen_ai.prompt.0.content": f"{CANARY_CUSTOMER_ID} {CANARY_PHONE}",
        "gen_ai.completion.0.content": CANARY_EMAIL,
        "langfuse.observation.model.name": "glm-4",
    })

    result = adapter._mask_otel_spans(params=_FakeOtelParams([span]))

    patch = result.span_patches[span.identifier]
    assert set(patch.delete_attributes) == {
        "langfuse.trace.input",
        "langfuse.trace.output",
        "langfuse.observation.input",
        "langfuse.observation.output",
        "gen_ai.prompt.0.content",
        "gen_ai.completion.0.content",
    }
    # 删除即不上送：patch.set_attributes 为空，没有任何金丝雀幸存
    assert patch.set_attributes == {}


def test_capture_content_dev_mode_redacts_pii_canary():
    """capture_content 仅开发环境生效：内容属性经 redact_pii 脱敏后保留，
    PII 金丝雀必须被替换，不得原文上送。"""
    adapter = _adapter(_enabled_cfg(capture_content=True))
    assert adapter.capture_content is True
    span = _FakeOtelSpan({
        "langfuse.observation.input": f"客户 {CANARY_PHONE} {CANARY_EMAIL} 的持仓 {CANARY_POSITIONS}",
    })

    result = adapter._mask_otel_spans(params=_FakeOtelParams([span]))

    kept = result.span_patches[span.identifier].set_attributes["langfuse.observation.input"]
    assert CANARY_PHONE not in kept
    assert CANARY_EMAIL not in kept


class _FakeOtelSpan:
    def __init__(self, attributes: dict[str, Any]) -> None:
        self.identifier = (uuid.uuid4().hex, uuid.uuid4().hex[:16])
        self.attributes = attributes


class _FakeOtelParams:
    def __init__(self, spans: list[_FakeOtelSpan]) -> None:
        self.spans = {span.identifier: span for span in spans}


# ─────────────── 验收点 4/6：超时/鉴权失败/写入异常降级 ───────────────


@pytest.mark.parametrize(
    ("fail_message", "reason", "exc_name"),
    [
        # _classify_failure 按异常名+消息的子串尽力分类：消息需含 "timeout" 字样
        ("langfuse request timeout exceeded", "timeout", "RuntimeError"),
        ("HTTP 401 Unauthorized from langfuse", "auth", "RuntimeError"),
        ("langfuse span write failed", "exception", "RuntimeError"),
    ],
)
@pytest.mark.asyncio
async def test_langfuse_failures_keep_business_result_and_count_locally(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    caplog: pytest.LogCaptureFixture,
    fail_message: str,
    reason: str,
    exc_name: str,
):
    client = FakeLangfuseClient(fail_message=fail_message)
    adapter = _adapter(_enabled_cfg(), client)
    agent = _StubAgentApp()
    _wire_api(monkeypatch, adapter, agent, tmp_path)

    with caplog.at_level(logging.WARNING, logger="secrag.langfuse"):
        response = await assistant_qa(
            _qa_request(),
            AuthenticatedUser(f"user_accept_{reason}", ROLE_TECHNICAL, "tech"),
        )

    # 业务结果仍正确：Langfuse 失败绝不抛进业务路径
    assert response.answer == EXPECTED_ANSWER
    assert response.citations == EXPECTED_CITATIONS
    assert response.compliance == EXPECTED_COMPLIANCE

    # 失败计数按 reason 增加
    assert adapter._metrics.langfuse_export_errors_total.get(labels={"reason": reason}) == 1.0

    # 本地日志可见：warning 带异常类型与原因
    assert "Langfuse 调用失败（业务不受影响）" in caplog.text
    assert exc_name in caplog.text
    assert fail_message in caplog.text


class _UpdateFailingClient(FakeLangfuseClient):
    """根 span 创建成功但后续 update 一律超时：请求收尾 trace.finish 触发。"""

    def start_observation(self, **kwargs: Any) -> FakeLangfuseSpan:
        span = super().start_observation(**kwargs)
        span.fail_update = True
        return span


@pytest.mark.asyncio
async def test_span_update_failure_mid_request_is_swallowed_and_logged(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any, caplog: pytest.LogCaptureFixture
):
    """请求收尾（finally 中 trace.finish）失败：响应先正常构建，收尾失败
    只进失败计数与本地日志，绝不上抛。"""
    client = _UpdateFailingClient()
    adapter = _adapter(_enabled_cfg(), client)
    agent = _StubAgentApp()
    _wire_api(monkeypatch, adapter, agent, tmp_path)

    with caplog.at_level(logging.WARNING, logger="secrag.langfuse"):
        response = await assistant_qa(
            _qa_request(),
            AuthenticatedUser("user_accept_midfail", ROLE_TECHNICAL, "tech"),
        )

    assert response.answer == EXPECTED_ANSWER
    root_span, _ = client.started[0]
    assert root_span.ended is True  # update 失败仍尽力 end，不阻塞
    assert adapter._metrics.langfuse_export_errors_total.get(
        labels={"reason": "timeout"}
    ) == 1.0
    assert "Langfuse 调用失败（业务不受影响）" in caplog.text
    assert "langfuse span update timed out" in caplog.text


@pytest.mark.asyncio
async def test_agent_failure_returns_http_error_not_langfuse_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
):
    """业务异常与 Langfuse 失败互不串扰：业务 500 保持 500，
    根 trace 收尾为 error（错误请求的 trace 必须保留）。"""
    client = FakeLangfuseClient()
    adapter = _adapter(_enabled_cfg(), client)
    _wire_api(monkeypatch, adapter, _RaisingAgentApp(), tmp_path)

    with pytest.raises(HTTPException) as exc:
        await assistant_qa(
            _qa_request(),
            AuthenticatedUser("user_accept_bizfail", ROLE_TECHNICAL, "tech"),
        )

    assert exc.value.status_code == 500
    root_span, _ = client.started[0]
    final_metadata = root_span.updates[-1]["metadata"]
    assert final_metadata["status"] == "error"
    assert final_metadata["error_type"] == "ValueError"
    assert adapter._metrics.langfuse_export_errors_total.get(
        labels={"reason": "exception"}
    ) == 0.0


# ─────────────── 验收点 5：request_id 关联且不暴露身份 ───────────────


@pytest.mark.asyncio
async def test_trace_request_id_correlates_with_audit_and_hides_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
):
    user_id = "user_accept_correlation"
    department = "tech"
    client = FakeLangfuseClient()
    adapter = _adapter(_enabled_cfg(), client)
    agent = _StubAgentApp()
    _wire_api(monkeypatch, adapter, agent, tmp_path)

    response = await assistant_qa(
        _qa_request(),
        AuthenticatedUser(user_id, ROLE_TECHNICAL, department),
    )

    assert agent.state is not None
    root_metadata = client.started[0][1]["metadata"]
    # Langfuse trace 与 SQLite 审计共用同一 request_id
    audit_request_id = agent.state[STATE_AUDIT_TRAIL][AUDIT_REQUEST_ID]
    assert root_metadata["request_id"] == audit_request_id
    assert root_metadata["thread_id"] == response.thread_id

    # 全部 payload（含后续 update）均不出现用户身份与问题原文
    payloads = _serialized_adapter_payloads(adapter, client)
    for marker in (user_id, department, ROLE_TECHNICAL, CANARY_QUESTION):
        assert marker not in payloads
    for key in ("user_id", "department", "role", "user_role"):
        for _, kwargs in client.started:
            assert key not in (kwargs.get("metadata") or {})
    for span, _ in client.started:
        for update in span.updates:
            metadata = update.get("metadata") or {}
            assert set(metadata) <= set(LangfuseTraceMetadata.__annotations__)


# ─────────────── 验收点 6：不可用时计数与本地告警 ───────────────


class _UnavailableClient(FakeLangfuseClient):
    """模拟 Langfuse 服务完全不可用：start_observation 与 flush 均失败。"""

    def flush(self) -> None:
        raise RuntimeError(self.fail_message or "connection refused")


def test_unavailable_langfuse_counts_all_failures_and_warns_locally(
    caplog: pytest.LogCaptureFixture,
):
    """client 全程不可用：建 trace、补偿错误 trace、flush 各路径全部
    fail-open，每次失败都计数并写本地 warning，节点 span 自动 no-op。"""
    adapter = _adapter(_enabled_cfg(), _UnavailableClient(fail_message="connection refused"))

    with caplog.at_level(logging.WARNING, logger="secrag.langfuse"):
        trace = adapter.start_request_trace("r", "t")
        assert trace.is_sampled is False
        # 建档失败后 trace 无根 span：节点 span 与收尾静默 no-op
        span = trace.start_span("retrieve", metadata={"node_name": "retrieve"})
        assert span.span_id is None
        span.finish(status="ok")
        # 失败请求收尾走补偿错误 trace 路径——同样 fail-open
        trace.finish(status="error", error_type="TimeoutError")
        adapter.flush()

    metrics = adapter._metrics
    # start_request_trace + 补偿错误 trace + flush，各计 1 次
    assert metrics.langfuse_export_errors_total.get(labels={"reason": "exception"}) == 3.0
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 3
    assert all("Langfuse 调用失败（业务不受影响）" in r.getMessage() for r in warnings)


def test_sampled_out_request_counts_drop_without_local_error(caplog: pytest.LogCaptureFixture):
    """采样丢弃是被设计的事件：计 dropped_total，不算失败、不告警。"""
    adapter = _adapter(_enabled_cfg(sample_rate=0.0))

    with caplog.at_level(logging.WARNING, logger="secrag.langfuse"):
        trace = adapter.start_request_trace("r", "t")
        trace.finish(status="ok")

    assert trace.is_sampled is False
    assert adapter._metrics.langfuse_dropped_total.get(labels={"reason": "sampled_out"}) == 1.0
    assert adapter._metrics.langfuse_export_errors_total.collect() == []
    assert caplog.text == ""
