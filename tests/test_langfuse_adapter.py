"""Langfuse adapter 单元测试：no-op、metadata 白名单、脱敏、fail-open、采样、
host 校验与 callback 绑定契约。

替身遵循 tests/ 现有模式：不连真实 Langfuse 服务，client 以依赖注入的
fake 代替；锁定版 langfuse.types 的 patch 类型用真实定义以校验语义。
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace
from typing import Any

import pytest

from src.config import LangfuseConfig
from src.utils.langfuse_adapter import (
    ERROR_TRACE_NAME,
    ROOT_TRACE_NAME,
    LangfuseAdapter,
    _is_content_attribute,
    get_langfuse,
    is_valid_langfuse_host,
    reset_langfuse_singleton,
)
from src.utils.metrics import MetricsRegistry
from src.utils.pii import redact_pii


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
    return LangfuseConfig(enabled=False, host="", public_key="", secret_key="")


class FakeLangfuseSpan:
    def __init__(self) -> None:
        self.trace_id = uuid.uuid4().hex
        self.id = uuid.uuid4().hex[:16]
        self.updates: list[dict[str, Any]] = []
        self.ended = False
        self.fail_update = False

    def update(self, **kwargs: Any) -> None:
        if self.fail_update:
            raise RuntimeError("update failed")
        self.updates.append(kwargs)

    def end(self, **kwargs: Any) -> None:
        self.ended = True


class FakeLangfuseClient:
    """记录 start_observation 调用的假 client；可注入失败。"""

    def __init__(self) -> None:
        self.started: list[tuple[FakeLangfuseSpan, dict[str, Any]]] = []
        self.fail_start = False
        self.flush_called = False

    def start_observation(self, **kwargs: Any) -> FakeLangfuseSpan:
        if self.fail_start:
            raise RuntimeError("langfuse unavailable")
        span = FakeLangfuseSpan()
        self.started.append((span, kwargs))
        return span

    def flush(self) -> None:
        self.flush_called = True


def _adapter(
    cfg: LangfuseConfig,
    client: FakeLangfuseClient | None = None,
    sampler: Any = None,
) -> tuple[LangfuseAdapter, FakeLangfuseClient]:
    fake = client if client is not None else FakeLangfuseClient()
    metrics = MetricsRegistry()
    adapter = LangfuseAdapter(
        cfg=cfg, app_env="development", client=fake, sampler=sampler, metrics=metrics
    )
    return adapter, fake


# ─────────────────────────────── no-op ───────────────────────────────


def test_disabled_adapter_is_fully_inert():
    adapter, fake = _adapter(_disabled_cfg())

    assert adapter.enabled is False
    trace = adapter.start_request_trace("req-1", "thread-1")
    assert trace.is_sampled is False
    assert trace.trace_id is None

    # 所有方法安全可调，不触碰任何 client
    trace.update({"model_name": "gpt-x"})
    span = trace.start_span("retrieve", metadata={"node_name": "retrieve"})
    span.finish(status="ok")
    trace.finish(status="ok")
    handler = adapter.get_callback_handler(trace)
    adapter.flush()

    assert handler is None
    assert fake.started == []
    assert fake.flush_called is False


def test_enabled_but_missing_keys_is_noop():
    cfg = LangfuseConfig(enabled=True, host="https://x", public_key="", secret_key="")
    adapter, fake = _adapter(cfg)

    assert adapter.enabled is False
    assert adapter.start_request_trace("r", "t").is_sampled is False
    assert fake.started == []


def test_invalid_host_disables_adapter_and_never_builds_client():
    # client 由注入提供也会因 host 非法被禁用：host 校验在构造期生效
    adapter, fake = _adapter(_enabled_cfg(host="cloud.langfuse.com"))

    assert adapter.enabled is False
    assert adapter.start_request_trace("r", "t").is_sampled is False
    assert fake.started == []


def test_noop_trace_error_finish_does_not_create_compensating_trace():
    # 整体禁用时业务零感知：finish(error) 也不产生任何 SDK 调用
    adapter, fake = _adapter(_disabled_cfg())

    adapter.start_request_trace("r", "t").finish(
        status="error", error_type="TimeoutError"
    )

    assert fake.started == []


# ─────────────────────────────── 白名单 ───────────────────────────────


def test_filter_metadata_keeps_only_whitelisted_scalars():
    adapter, _ = _adapter(_enabled_cfg())
    raw = {
        "request_id": "req-1",
        "thread_id": "thread-1",
        "node_name": "verify",
        "model_name": "glm-4",
        "duration_ms": 12.5,
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "total_tokens": 30,
        "retry_count": 1,
        "retrieval_count": 5,
        "verification_status": "passed",
        "compliance_status": "allowed",
        "status": "ok",
        "error_type": "TimeoutError",
    }

    filtered = adapter.filter_metadata(raw)

    assert filtered == raw


def test_filter_metadata_drops_unknown_and_unsafe_values():
    adapter, _ = _adapter(_enabled_cfg())
    raw: dict[str, Any] = {
        "question": "原始用户问题",
        "answer": "模型完整回答",
        "chunk_text": "文档原文",
        "sql": "SELECT * FROM holdings",
        "customer_id": "C001",
        "positions": [{"sec": "600000"}],
        "user_id": "u-1",
        "nested": {"a": 1},
        "tags": ["x"],
        "none_val": None,
        "model_name": "glm-4",
    }

    filtered = adapter.filter_metadata(raw)

    assert filtered == {"model_name": "glm-4"}


def test_filter_metadata_rejects_non_scalar_whitelisted_key():
    # 白名单键携带非标量值同样丢弃——禁止通过改值类型绕过脱敏
    adapter, _ = _adapter(_enabled_cfg())

    assert adapter.filter_metadata({"model_name": {"raw": "原文"}}) == {}
    assert adapter.filter_metadata({"request_id": ["req-1"]}) == {}
    assert adapter.filter_metadata(None) == {}


def test_root_trace_metadata_contains_request_and_thread_id():
    adapter, fake = _adapter(_enabled_cfg())
    trace = adapter.start_request_trace(
        "req-7", "thread-7", metadata={"question": "原文", "retrieval_count": 3}
    )
    trace.finish(status="ok")

    span, kwargs = fake.started[0]
    metadata = kwargs["metadata"]
    assert metadata["request_id"] == "req-7"
    assert metadata["thread_id"] == "thread-7"
    assert metadata["retrieval_count"] == 3
    assert "question" not in metadata


# ─────────────────────────────── 脱敏 ───────────────────────────────


def test_sanitize_text_redacts_pii():
    sanitized = LangfuseAdapter._sanitize_text(
        "客户 13812345678 持有 600000，邮箱 a.b@example.com"
    )

    assert "13812345678" not in sanitized
    assert "a.b@example.com" not in sanitized
    assert sanitized == redact_pii("客户 13812345678 持有 600000，邮箱 a.b@example.com")[0]


def test_is_content_attribute_matches_langfuse_and_genai_keys():
    assert _is_content_attribute("langfuse.observation.input")
    assert _is_content_attribute("langfuse.trace.output")
    assert not _is_content_attribute("langfuse.observation.model.name")
    assert not _is_content_attribute("langfuse.observation.metadata.request_id")


def test_mask_deletes_content_attributes_by_default():
    adapter, _ = _adapter(_enabled_cfg())
    span = SimpleNamespace(
        attributes={
            "langfuse.observation.input": '{"question":"原文"}',
            "langfuse.observation.output": "完整回答",
            "langfuse.observation.model.name": "glm-4",
            "gen_ai.prompt.0.content": "chunk 原文",
        }
    )
    params = SimpleNamespace(spans={("t", "s"): span})

    result = adapter._mask_otel_spans(params=params)

    patch = result.span_patches[("t", "s")]
    assert set(patch.delete_attributes) == {
        "langfuse.observation.input",
        "langfuse.observation.output",
        "gen_ai.prompt.0.content",
    }
    assert patch.set_attributes == {}


def test_mask_sanitizes_content_when_capture_content_enabled():
    adapter, _ = _adapter(_enabled_cfg(capture_content=True))
    assert adapter.capture_content is True
    span = SimpleNamespace(
        attributes={
            "langfuse.observation.input": "客户 13812345678 的持仓",
            "langfuse.observation.output": 42,
            "langfuse.observation.model.name": "glm-4",
        }
    )
    params = SimpleNamespace(spans={("t", "s"): span})

    result = adapter._mask_otel_spans(params=params)

    set_attrs = result.span_patches[("t", "s")].set_attributes
    assert "13812345678" not in set_attrs["langfuse.observation.input"]
    assert set_attrs["langfuse.observation.output"] == 42
    assert "langfuse.observation.model.name" not in set_attrs


def test_mask_fails_closed_when_attributes_unreadable():
    # 属性不可读时必须抛出，由 SDK 丢弃整个导出批次——绝不放行未脱敏原文
    class ExplodingSpan:
        @property
        def attributes(self) -> Any:
            raise RuntimeError("unreadable")

    adapter, _ = _adapter(_enabled_cfg())
    params = SimpleNamespace(spans={("t", "s"): ExplodingSpan()})

    with pytest.raises(RuntimeError):
        adapter._mask_otel_spans(params=params)


def test_capture_content_forced_off_outside_development():
    adapter = LangfuseAdapter(
        cfg=_enabled_cfg(capture_content=True),
        app_env="production",
        client=FakeLangfuseClient(),
        metrics=MetricsRegistry(),
    )

    assert adapter.capture_content is False
    assert adapter.enabled is True


# ─────────────────────────────── fail-open ───────────────────────────────


def test_client_start_failure_returns_noop_trace_and_counts():
    client = FakeLangfuseClient()
    client.fail_start = True
    adapter, fake = _adapter(_enabled_cfg(), client=client)

    trace = adapter.start_request_trace("r", "t")

    assert trace.is_sampled is False
    assert trace.trace_id is None
    assert adapter._metrics.langfuse_export_errors_total.get(
        labels={"reason": "exception"}
    ) == 1.0
    assert fake.started == []


def test_span_update_failure_is_swallowed_and_counted():
    adapter, fake = _adapter(_enabled_cfg())
    trace = adapter.start_request_trace("r", "t")
    span, _ = fake.started[0]
    span.fail_update = True

    trace.finish(status="error", error_type="TimeoutError")

    assert span.ended is True  # update 失败仍尽力收尾，不抛异常
    assert (
        adapter._metrics.langfuse_export_errors_total.get(
            labels={"reason": "exception"}
        )
        == 1.0
    )


def test_flush_failure_is_swallowed():
    class FailingFlushClient(FakeLangfuseClient):
        def flush(self) -> None:
            raise RuntimeError("flush failed")

    adapter, _ = _adapter(_enabled_cfg(), client=FailingFlushClient())

    adapter.flush()  # 不抛异常
    assert (
        adapter._metrics.langfuse_export_errors_total.get(
            labels={"reason": "exception"}
        )
        == 1.0
    )


def test_failure_classification_labels():
    assert LangfuseAdapter._classify_failure(TimeoutError("x")) == "timeout"
    assert LangfuseAdapter._classify_failure(RuntimeError("HTTP 401 unauthorized")) == "auth"
    assert LangfuseAdapter._classify_failure(RuntimeError("boom")) == "exception"


# ─────────────────────────────── 采样 ───────────────────────────────


def test_sampling_drops_trace_and_keeps_error_via_compensation():
    adapter, fake = _adapter(_enabled_cfg(sample_rate=0.5), sampler=lambda: 0.9)

    trace = adapter.start_request_trace("req-e", "thread-e")
    assert trace.is_sampled is False
    assert fake.started == []  # 未采样：不创建任何 span
    assert adapter._metrics.langfuse_dropped_total.get(
        labels={"reason": "sampled_out"}
    ) == 1.0

    trace.finish(status="error", error_type="TimeoutError")

    # 错误请求的 trace 必须保留：补偿一条仅含白名单 metadata 的错误 trace
    assert len(fake.started) == 1
    span, kwargs = fake.started[0]
    assert kwargs["name"] == ERROR_TRACE_NAME
    assert kwargs["level"] == "ERROR"
    assert kwargs["metadata"]["status"] == "error"
    assert kwargs["metadata"]["request_id"] == "req-e"
    assert kwargs["metadata"]["thread_id"] == "thread-e"
    assert kwargs["metadata"]["error_type"] == "TimeoutError"
    assert span.ended is True


def test_sampling_keeps_trace_within_rate():
    adapter, fake = _adapter(_enabled_cfg(sample_rate=0.5), sampler=lambda: 0.1)

    trace = adapter.start_request_trace("r", "t")

    assert trace.is_sampled is True
    assert len(fake.started) == 1
    assert fake.started[0][1]["name"] == ROOT_TRACE_NAME
    assert trace.trace_id == fake.started[0][0].trace_id


def test_sample_rate_zero_never_samples():
    adapter, fake = _adapter(_enabled_cfg(sample_rate=0.0), sampler=lambda: 0.0)

    assert adapter.start_request_trace("r", "t").is_sampled is False
    assert fake.started == []


def test_error_trace_bypasses_sampling():
    adapter, fake = _adapter(_enabled_cfg(sample_rate=0.0))

    trace = adapter.start_request_trace("r", "t", is_error=True)

    assert trace.is_sampled is True
    assert len(fake.started) == 1
    assert fake.started[0][1]["level"] == "ERROR"
    assert fake.started[0][1]["metadata"]["status"] == "error"


def test_finish_is_idempotent():
    adapter, fake = _adapter(_enabled_cfg())

    trace = adapter.start_request_trace("r", "t")
    trace.finish(status="ok")
    trace.finish(status="ok")

    span, _ = fake.started[0]
    assert span.ended is True
    assert len(fake.started) == 1


# ─────────────────────────────── host 校验 ───────────────────────────────


def test_host_validation_accepts_http_https_including_private():
    # 自托管（localhost/内网）是 TODO 明确选项：HOST 是运维配置而非用户输入
    assert is_valid_langfuse_host("https://cloud.langfuse.com")
    assert is_valid_langfuse_host("http://localhost:3000")
    assert is_valid_langfuse_host("http://10.0.0.8:3000")
    assert is_valid_langfuse_host("https://langfuse.internal.example")


def test_host_validation_rejects_non_http_schemes():
    assert not is_valid_langfuse_host("ftp://langfuse.example")
    assert not is_valid_langfuse_host("file:///etc/passwd")
    assert not is_valid_langfuse_host("cloud.langfuse.com")  # 缺 scheme
    assert not is_valid_langfuse_host("")
    assert not is_valid_langfuse_host("https://")  # 缺 host


# ─────────────────────── 导出过滤与 callback 绑定 ───────────────────────


def test_only_registered_traces_export():
    adapter, fake = _adapter(_enabled_cfg())
    adapter.start_request_trace("r", "t")
    trace_id = fake.started[0][0].trace_id

    assert adapter._should_export_span(
        SimpleNamespace(context=SimpleNamespace(trace_id=int(trace_id, 16)))
    )
    assert not adapter._should_export_span(
        SimpleNamespace(context=SimpleNamespace(trace_id=int("a" * 32, 16)))
    )


def test_get_callback_handler_binds_root_trace_context(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, Any] = {}

    class FakeHandler:
        def __init__(self, *, trace_context: dict[str, str] | None = None) -> None:
            captured["trace_context"] = trace_context

    monkeypatch.setattr(
        "langfuse.langchain.CallbackHandler", FakeHandler, raising=True
    )
    adapter, fake = _adapter(_enabled_cfg())
    trace = adapter.start_request_trace("r", "t")
    root_span = fake.started[0][0]

    handler = adapter.get_callback_handler(trace)

    assert isinstance(handler, FakeHandler)
    assert captured["trace_context"] == {
        "trace_id": root_span.trace_id,
        "parent_span_id": root_span.id,
    }


def test_get_callback_handler_returns_none_for_unsampled_trace(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        "langfuse.langchain.CallbackHandler",
        lambda **_: pytest.fail("handler must not be built for unsampled trace"),
        raising=True,
    )
    adapter, _ = _adapter(_enabled_cfg(sample_rate=0.0))

    assert adapter.get_callback_handler(adapter.start_request_trace("r", "t")) is None


def test_get_callback_handler_failure_is_fail_open(monkeypatch: pytest.MonkeyPatch):
    def _boom(**_: Any) -> None:
        raise RuntimeError("handler init failed")

    monkeypatch.setattr("langfuse.langchain.CallbackHandler", _boom, raising=True)
    adapter, fake = _adapter(_enabled_cfg())

    assert adapter.get_callback_handler(adapter.start_request_trace("r", "t")) is None
    assert (
        adapter._metrics.langfuse_export_errors_total.get(
            labels={"reason": "exception"}
        )
        == 1.0
    )


def test_node_span_attaches_to_root_trace_and_carries_whitelisted_metadata():
    adapter, fake = _adapter(_enabled_cfg())
    trace = adapter.start_request_trace("r", "t")
    root_span, _ = fake.started[0]

    span = trace.start_span(
        "verify_answer", metadata={"node_name": "verify", "answer": "原文"}
    )
    span.finish(status="ok", metadata={"verification_status": "passed"})

    assert len(fake.started) == 2
    _, kwargs = fake.started[1]
    assert kwargs["name"] == "verify_answer"
    assert kwargs["trace_context"] == {
        "trace_id": root_span.trace_id,
        "parent_span_id": root_span.id,
    }
    assert kwargs["metadata"] == {"node_name": "verify"}


# ─────────────────────────────── 单例 ───────────────────────────────


def test_get_langfuse_singleton_is_cached(monkeypatch: pytest.MonkeyPatch):
    reset_langfuse_singleton()

    class _StubSettings:
        app_env = "development"
        langfuse = _disabled_cfg()

    monkeypatch.setattr("src.config.config", _StubSettings(), raising=True)
    try:
        first = get_langfuse()
        second = get_langfuse()
        assert first is second
        assert first.enabled is False
    finally:
        reset_langfuse_singleton()
