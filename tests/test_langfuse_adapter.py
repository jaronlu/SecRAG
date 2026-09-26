"""Langfuse adapter 单元测试：no-op、metadata 白名单、脱敏、fail-open、采样、
host 校验与 callback 绑定契约。

替身遵循 tests/ 现有模式：不连真实 Langfuse 服务，client 以依赖注入的
fake 代替；锁定版 langfuse.types 的 patch 类型用真实定义以校验语义。
"""

from __future__ import annotations

import logging
import os
import uuid
from types import SimpleNamespace
from typing import Any

import pytest

from src.config import LangfuseConfig
from src.utils.langfuse_adapter import (
    ERROR_TRACE_NAME,
    ROOT_TRACE_NAME,
    LangfuseAdapter,
    _CONTENT_ATTRIBUTE_KEYS,
    _CONTENT_ATTRIBUTE_PREFIXES,
    _OTEL_EXPORTER_LOGGER_NAME,
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
    assert _is_content_attribute("langfuse.observation.status_message")
    assert _is_content_attribute("gen_ai.input.messages")
    assert not _is_content_attribute("langfuse.observation.model.name")
    assert not _is_content_attribute("langfuse.observation.metadata.request_id")


def test_mask_deletes_error_status_message_by_default():
    """langchain handler 的错误回调把异常原文整段写入 status_message 属性
    （_get_error_level_and_status_message 返回 str(error)），provider 4xx 响应体
    可能回显请求内容片段——默认必须整段删除；capture_content 下经脱敏保留。"""
    exception_message = (
        "Error code: 400 - {'error': {'message': '问题原文 客户 13812345678'}}"
    )
    span = SimpleNamespace(
        attributes={"langfuse.observation.status_message": exception_message}
    )
    params = SimpleNamespace(spans={("t", "s"): span})

    adapter, _ = _adapter(_enabled_cfg())
    result = adapter._mask_otel_spans(params=params)

    patch = result.span_patches[("t", "s")]
    assert patch.delete_attributes == ("langfuse.observation.status_message",)
    assert patch.set_attributes == {}

    capture_adapter, _ = _adapter(_enabled_cfg(capture_content=True))
    capture_result = capture_adapter._mask_otel_spans(params=params)
    kept = capture_result.span_patches[("t", "s")].set_attributes[
        "langfuse.observation.status_message"
    ]
    assert "13812345678" not in kept  # PII 必须被 redact_pii 替换


def test_mask_set_covers_all_sdk_registered_free_text_keys():
    """mask 删除集合完备性回归网：对照锁定版本 SDK 的真实属性注册表。

    1. SDK 注册的 langfuse.* 自由文本键（trace/observation input/output、
       observation.status_message）必须全部被默认删除；
    2. SDK 媒体键集合中的 gen_ai.* 内容键必须被删除集合或前缀覆盖；
    3. 其余 SDK 注册键必须显式登记在 reviewed-keep 清单——SDK 升级新增键会让
       本测试失败，强制先审阅再放行，杜绝"handler 写入键集合之外的属性"类
       缺口（真实 CallbackHandler 只写 langfuse.* 键；第三方 gen_ai
       instrumentation 本进程未启用，此处按 SDK 注册表兜底）。
    """
    from langfuse._client.attributes import LangfuseOtelSpanAttributes
    from langfuse._client.span_exporter import (
        _INPUT_MEDIA_ATTRIBUTE_KEYS,
        _OUTPUT_MEDIA_ATTRIBUTE_KEYS,
    )

    free_text_keys = {
        LangfuseOtelSpanAttributes.TRACE_INPUT,
        LangfuseOtelSpanAttributes.TRACE_OUTPUT,
        LangfuseOtelSpanAttributes.OBSERVATION_INPUT,
        LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT,
        LangfuseOtelSpanAttributes.OBSERVATION_STATUS_MESSAGE,
    }
    assert free_text_keys <= set(_CONTENT_ATTRIBUTE_KEYS)

    genai_content_keys = {
        key
        for key in _INPUT_MEDIA_ATTRIBUTE_KEYS | _OUTPUT_MEDIA_ATTRIBUTE_KEYS
        if key.startswith("gen_ai.")
    }
    assert genai_content_keys <= set(_CONTENT_ATTRIBUTE_KEYS) | set(
        _CONTENT_ATTRIBUTE_PREFIXES
    )

    # reviewed-keep 清单：SDK 注册表中允许通过 mask 的键（均不含内容原文）
    reviewed_keep = {
        # 标识与结构：名称/标签/等级/环境/内部标记
        LangfuseOtelSpanAttributes.TRACE_NAME,
        LangfuseOtelSpanAttributes.TRACE_USER_ID,
        LangfuseOtelSpanAttributes.TRACE_SESSION_ID,
        LangfuseOtelSpanAttributes.TRACE_TAGS,
        LangfuseOtelSpanAttributes.TRACE_PUBLIC,
        LangfuseOtelSpanAttributes.OBSERVATION_TYPE,
        LangfuseOtelSpanAttributes.OBSERVATION_LEVEL,
        LangfuseOtelSpanAttributes.ENVIRONMENT,
        LangfuseOtelSpanAttributes.RELEASE,
        LangfuseOtelSpanAttributes.VERSION,
        LangfuseOtelSpanAttributes.AS_ROOT,
        LangfuseOtelSpanAttributes.IS_APP_ROOT,
        # 观测数据：模型名与数值型用量/成本/时序
        LangfuseOtelSpanAttributes.OBSERVATION_MODEL,
        LangfuseOtelSpanAttributes.OBSERVATION_MODEL_PARAMETERS,
        LangfuseOtelSpanAttributes.OBSERVATION_USAGE_DETAILS,
        LangfuseOtelSpanAttributes.OBSERVATION_COST_DETAILS,
        LangfuseOtelSpanAttributes.OBSERVATION_COMPLETION_START_TIME,
        LangfuseOtelSpanAttributes.OBSERVATION_PROMPT_NAME,
        LangfuseOtelSpanAttributes.OBSERVATION_PROMPT_VERSION,
        # metadata 逐键展平（langfuse.*.metadata.<key>）：adapter 侧只放行
        # LangfuseTraceMetadata 白名单标量；langchain 侧为 ls_* 运行标识
        LangfuseOtelSpanAttributes.TRACE_METADATA,
        LangfuseOtelSpanAttributes.OBSERVATION_METADATA,
        # 实验键：本进程不使用 Langfuse 实验；若启用需先复核 expected_output
        # 等内容承载键
        LangfuseOtelSpanAttributes.EXPERIMENT_ID,
        LangfuseOtelSpanAttributes.EXPERIMENT_NAME,
        LangfuseOtelSpanAttributes.EXPERIMENT_DESCRIPTION,
        LangfuseOtelSpanAttributes.EXPERIMENT_METADATA,
        LangfuseOtelSpanAttributes.EXPERIMENT_DATASET_ID,
        LangfuseOtelSpanAttributes.EXPERIMENT_ITEM_ID,
        LangfuseOtelSpanAttributes.EXPERIMENT_ITEM_EXPECTED_OUTPUT,
        LangfuseOtelSpanAttributes.EXPERIMENT_ITEM_METADATA,
        LangfuseOtelSpanAttributes.EXPERIMENT_ITEM_ROOT_OBSERVATION_ID,
    }
    registered = {
        value
        for name, value in vars(LangfuseOtelSpanAttributes).items()
        if isinstance(value, str) and not name.startswith("__")
    }
    assert registered <= set(_CONTENT_ATTRIBUTE_KEYS) | reviewed_keep


def test_mask_deletes_every_sdk_free_text_key_end_to_end():
    """行为级：span 携带全部 SDK 注册自由文本键的 canary 时，导出补丁必须
    整段删除每一个键，不留任何 canary。"""
    from langfuse._client.attributes import LangfuseOtelSpanAttributes

    free_text_keys = (
        LangfuseOtelSpanAttributes.TRACE_INPUT,
        LangfuseOtelSpanAttributes.TRACE_OUTPUT,
        LangfuseOtelSpanAttributes.OBSERVATION_INPUT,
        LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT,
        LangfuseOtelSpanAttributes.OBSERVATION_STATUS_MESSAGE,
        "gen_ai.input.messages",
        "gen_ai.output.messages",
        "gen_ai.system_instructions",
        "gen_ai.tool.call.arguments",
        "gen_ai.tool.call.result",
    )
    span = SimpleNamespace(
        attributes={key: f"canary-{key}-原文" for key in free_text_keys}
    )
    adapter, _ = _adapter(_enabled_cfg())

    result = adapter._mask_otel_spans(
        params=SimpleNamespace(spans={("t", "s"): span})
    )

    patch = result.span_patches[("t", "s")]
    assert set(patch.delete_attributes) == set(free_text_keys)
    assert patch.set_attributes == {}


def test_init_client_disables_media_upload_before_client_construction(monkeypatch):
    """SDK 先做媒体预上传、后执行 mask：媒体形内容会在删除前离开进程，
    必须在构造 client（MediaManager 读取环境变量）之前整体关闭。"""
    captured: dict[str, str | None] = {}

    class EnvCapturingClient(FakeLangfuseClient):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__()
            captured["media_upload_enabled"] = os.environ.get(
                "LANGFUSE_MEDIA_UPLOAD_ENABLED"
            )

    monkeypatch.setattr("langfuse.Langfuse", EnvCapturingClient)
    monkeypatch.setenv("LANGFUSE_MEDIA_UPLOAD_ENABLED", "true")

    adapter = LangfuseAdapter(
        cfg=_enabled_cfg(), app_env="development", metrics=MetricsRegistry()
    )

    assert adapter.enabled is True
    assert captured["media_upload_enabled"] == "false"


def test_exporter_log_failures_count_into_prometheus_and_warn_locally(caplog):
    """export 阶段失败（网络超时/401/403/连接拒绝）由 OTLPSpanExporter 只写自身
    logger 并返回 FAILURE，不抛进 adapter——日志钩子必须把它们转入本地告警与
    secrag_langfuse_export_errors_total。"""
    adapter, _ = _adapter(_enabled_cfg())
    adapter._attach_export_failure_log_handler()
    exporter_logger = logging.getLogger(_OTEL_EXPORTER_LOGGER_NAME)
    try:
        with caplog.at_level(logging.WARNING, logger="secrag.langfuse"):
            exporter_logger.error(
                "Failed to export span batch code: 401, reason: Unauthorized"
            )
            exporter_logger.error(
                "Failed to export span batch due to timeout, max retries or shutdown."
            )
            exporter_logger.error(
                "Failed to export span batch code: None, reason: Connection refused"
            )
    finally:
        exporter_logger.removeHandler(adapter._export_log_handler)

    metrics = adapter._metrics
    assert metrics.langfuse_export_errors_total.get(labels={"reason": "auth"}) == 1.0
    assert metrics.langfuse_export_errors_total.get(labels={"reason": "timeout"}) == 1.0
    assert (
        metrics.langfuse_export_errors_total.get(labels={"reason": "exception"}) == 1.0
    )
    assert "Langfuse export 失败（业务不受影响）" in caplog.text


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
    # "timed out" 措辞（类型名不含 timeout，如 RuntimeError 包装）也命中 timeout
    assert LangfuseAdapter._classify_failure(RuntimeError("request timed out")) == "timeout"


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
