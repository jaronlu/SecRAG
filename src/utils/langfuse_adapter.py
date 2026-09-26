"""统一 Langfuse adapter——Agent/LLM 链路观测的唯一入口。

职责边界（todo/TODO.md）：
- Langfuse：Agent/LLM 运行链路、耗时、token、成本观察。
- SQLite（audit.py）：权限、引用、合规和业务审计，不经过本模块。
- Prometheus（metrics.py）：QPS/延迟/错误率/缓存指标，不重复建设。

隐私红线：用户原始问题、模型完整回答、文档/chunk 原文、工具原始参数、SQL、
客户 ID、持仓明细、PII 一律不进 payload。防线有两层：
1. metadata 白名单：adapter 对外只接受 ``LangfuseTraceMetadata`` 中列出的
   标量字段，其余键一律丢弃——新增字段必须改本文件的白名单，无法绕过。
2. export 阶段统一脱敏（``mask_otel_spans``）：锁定版本 langfuse 4.15.6 的
   langchain ``CallbackHandler`` 会把每次 chain/LLM/tool run 的 input/output
   原文挂到 span 属性上（CallbackHandler.py:462-468、618、629、995、1048），
   且该 handler 不提供 mask/hide-input 选项（全文件无 mask 入口）；SDK 客户端
   的 ``mask`` 只作用于 SDK API 写入的数据（client.py:220）。因此统一在
   ``mask_otel_spans`` 导出层兜底——它对该 client 导出的每个 span 生效
   （span_exporter.py:120-121），默认删除内容属性；仅当开发环境显式开启
   ``LANGFUSE_CAPTURE_CONTENT`` 时改为经 ``redact_pii`` 脱敏后保留。

可靠性红线（fail-open）：Langfuse 未配置、初始化失败、超时、鉴权失败、
写入异常时，业务链路照常完成；错误只写本地日志和
``secrag_langfuse_export_errors_total`` / ``secrag_langfuse_dropped_total``
计数，绝不抛进业务路径。

采样：``LANGFUSE_SAMPLE_RATE`` 在 adapter 请求边界做头部采样；被采样掉的
请求不创建任何 span。错误请求的 trace 必须保留：
- 建档时已知的错误（``is_error=True``）不参与采样，一律保留；
- 采样掉的请求若在执行中失败，``finish(status="error")`` 会在请求结束时
  补建一条仅含白名单 metadata 的错误 trace，保证每个错误请求都有 trace。
  无法做"整链尾部采样"的原因：SDK 的 ``should_export_span`` 在每个 span
  end 时逐个求值（span_processor.py on_end），请求结束时无法追溯保留已
  结束的子 span；SDK 自带 ``sample_rate`` 是 OTel ``TraceIdRatioBased``
  头部采样（resource_manager.py:672），同样无法保留错误。因此 SDK 采样
  固定为 1.0，采样决策集中在 adapter。

线程安全：请求经 asyncio.to_thread / 工具线程池并发执行，registry 与
client 调用均按并发场景设计；``mask_otel_spans`` / ``should_export_span``
在 SDK 批处理线程上运行，保持确定性、无请求局部依赖（types.py 对
``MaskOtelSpansFunction`` 的约束）。
"""

from __future__ import annotations

import logging
import random
import threading
from collections import OrderedDict
from typing import Any, Callable, Mapping, TypedDict
from urllib.parse import urlparse

from src.config import LangfuseConfig
from src.utils.metrics import MetricsRegistry, get_metrics
from src.utils.pii import redact_pii

logger = logging.getLogger("secrag.langfuse")

# 根 trace / 补偿错误 trace 的固定命名，便于在 Langfuse UI 过滤
ROOT_TRACE_NAME = "agent.request"
ERROR_TRACE_NAME = "agent.request.error"

# 允许写入 Langfuse 的 metadata 白名单。新增字段必须在此显式登记；
# 未列出的键（含嵌套 dict/list、对象、None）在 filter_metadata 处一律丢弃。
class LangfuseTraceMetadata(TypedDict, total=False):
    """Langfuse trace/span metadata 白名单（固定结构，标量值）。"""

    request_id: str
    thread_id: str
    node_name: str
    model_name: str
    duration_ms: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    retry_count: int
    retrieval_count: int
    verification_status: str
    compliance_status: str
    status: str
    error_type: str


_METADATA_WHITELIST = frozenset(LangfuseTraceMetadata.__annotations__)

# 内容属性：langchain handler 与 SDK API 写入的原文都落在这些键上
# （attributes.py:35-44）；gen_ai.* 是 OTel 生成式 AI 语义约定的兜底前缀。
_CONTENT_ATTRIBUTE_KEYS = frozenset(
    {
        "langfuse.trace.input",
        "langfuse.trace.output",
        "langfuse.observation.input",
        "langfuse.observation.output",
    }
)
_CONTENT_ATTRIBUTE_PREFIXES = ("gen_ai.prompt.", "gen_ai.completion.", "gen_ai.request.")

# trace 注册表容量上限：晚于登记条目被淘汰的迟到 span 会被包含过滤器丢弃
# （fail-closed），此窗口远大于单请求 span 数，正常链路不受影响。
_TRACE_REGISTRY_MAX = 4096


def _is_content_attribute(key: str) -> bool:
    return key in _CONTENT_ATTRIBUTE_KEYS or key.startswith(_CONTENT_ATTRIBUTE_PREFIXES)


def is_valid_langfuse_host(host: str) -> bool:
    """校验 LANGFUSE_HOST：仅接受 http/https 绝对地址。

    刻意**不**拒绝 localhost / 内网地址：自托管 Langfuse 是 todo/TODO.md 的
    明确选项，实例地址是运维配置而非用户输入——本 adapter 只从 Settings
    （环境变量/.env）读取 host，绝不接受来自用户输入或请求数据的 URL。
    """
    if not host:
        return False
    parsed = urlparse(host)
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


class _LangfuseSpanHandle:
    """节点级 span 句柄。所有方法 fail-open，绝不上抛异常。"""

    __slots__ = ("_adapter", "_otel_span_id", "_span")

    def __init__(self, adapter: LangfuseAdapter, span: Any, span_id: str | None):
        self._adapter = adapter
        self._span = span
        self._otel_span_id = span_id

    @property
    def span_id(self) -> str | None:
        return self._otel_span_id

    def finish(
        self,
        *,
        status: str = "ok",
        error_type: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if self._span is None:
            return
        try:
            filtered = self._adapter.filter_metadata(metadata)
            filtered["status"] = status
            if error_type:
                filtered["error_type"] = error_type
            level = "ERROR" if status != "ok" else "DEFAULT"
            self._span.update(
                metadata=filtered or None,
                level=level,
                status_message=error_type,
            )
            self._span.end()
        except Exception as exc:
            self._adapter._record_export_failure(exc)
            self._safe_end()

    def _safe_end(self) -> None:
        try:
            self._span.end()
        except Exception:
            logger.debug("langfuse span end failed after error", exc_info=True)


class _RequestTrace:
    """一次请求的根 trace 句柄；未采样/未启用时 span 为 None（业务零感知）。

    晚到事件容忍：``finish`` 只结束根 span，不等待子 span（非流式 504 路径
    中 to_thread 的图线程不可取消、可能继续产生事件），也不阻塞调用方。
    """

    __slots__ = (
        "_adapter",
        "_finished",
        "_request_id",
        "_span",
        "_span_id",
        "_sampled",
        "_thread_id",
        "trace_id",
    )

    def __init__(
        self,
        adapter: LangfuseAdapter,
        span: Any,
        trace_id: str | None,
        span_id: str | None,
        sampled: bool,
        request_id: str = "",
        thread_id: str = "",
    ):
        self._adapter = adapter
        self._span = span
        self.trace_id = trace_id
        self._span_id = span_id
        self._sampled = sampled
        self._request_id = request_id
        self._thread_id = thread_id
        self._finished = False

    @property
    def span_id(self) -> str | None:
        return self._span_id

    @property
    def is_sampled(self) -> bool:
        """是否被采样保留（False 表示本请求不产生任何 Langfuse 数据）。"""
        return self._sampled

    def update(self, metadata: Mapping[str, Any] | None = None) -> None:
        if self._span is None or self._finished:
            return
        try:
            filtered = self._adapter.filter_metadata(metadata)
            if filtered:
                self._span.update(metadata=filtered)
        except Exception as exc:
            self._adapter._record_export_failure(exc)

    def start_span(
        self,
        name: str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> _LangfuseSpanHandle:
        """在根 trace 下创建节点/工具 span；未采样或失败时返回 no-op 句柄。"""
        if self._span is None or self._adapter._client is None:
            return _LangfuseSpanHandle(self._adapter, None, None)
        try:
            filtered = self._adapter.filter_metadata(metadata)
            span = self._adapter._client.start_observation(
                name=name,
                as_type="span",
                metadata=filtered or None,
                # 挂到本请求根 trace 之下，与 langchain handler 同一 trace_id
                trace_context={
                    "trace_id": self.trace_id,
                    "parent_span_id": self._span_id,
                },
            )
            return _LangfuseSpanHandle(self._adapter, span, span.id)
        except Exception as exc:
            self._adapter._record_export_failure(exc)
            return _LangfuseSpanHandle(self._adapter, None, None)

    def finish(
        self,
        *,
        status: str = "ok",
        error_type: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if self._finished:
            return
        self._finished = True
        if self._span is not None:
            try:
                filtered = self._adapter.filter_metadata(metadata)
                filtered["status"] = status
                if error_type:
                    filtered["error_type"] = error_type
                level = "ERROR" if status != "ok" else "DEFAULT"
                self._span.update(
                    metadata=filtered or None,
                    level=level,
                    status_message=error_type,
                )
                self._span.end()
            except Exception as exc:
                self._adapter._record_export_failure(exc)
                try:
                    self._span.end()
                except Exception:
                    logger.debug("langfuse root span end failed", exc_info=True)
            return
        if status != "error":
            return
        # 未采样但请求失败：补偿一条仅含白名单 metadata 的错误 trace，
        # 保证错误请求不被采样丢弃（见模块 docstring 的采样说明）。
        # 仅在 adapter 启用且 client 可用时补偿；整体禁用时业务零感知。
        self._adapter._create_error_only_trace(
            error_type=error_type,
            metadata=metadata,
            request_id=self._request_id,
            thread_id=self._thread_id,
        )


class LangfuseAdapter:
    """Langfuse client、根 trace、节点 span 与 LangChain callback 的统一封装。

    通过 :func:`get_langfuse` 获取单例；不要在业务代码里直接构造
    （测试除外）。启用条件：``enabled=True`` 且 public/secret key 齐全且
    host 校验通过；否则整体 no-op——所有方法安全可调。
    """

    def __init__(
        self,
        *,
        cfg: LangfuseConfig,
        app_env: str,
        client: Any | None = None,
        sampler: Callable[[], float] | None = None,
        metrics: MetricsRegistry | None = None,
    ):
        self._cfg = cfg
        self._app_env = app_env
        self._sampler = sampler or random.random
        self._metrics = metrics or get_metrics()
        self._registry_lock = threading.Lock()
        # trace_id(hex) -> 是否导出。启用即登记；未登记的 span 一律不导出。
        self._trace_registry: OrderedDict[str, bool] = OrderedDict()

        keys_present = bool(cfg.public_key) and bool(
            cfg.secret_key.get_secret_value()
        )
        host_ok = is_valid_langfuse_host(cfg.host)
        if cfg.enabled and not host_ok:
            logger.error(
                "LANGFUSE_HOST 非法（仅接受 http/https 绝对地址），Langfuse 已禁用：%r",
                cfg.host,
            )
        self._enabled = cfg.enabled and keys_present and host_ok

        # capture_content 仅开发环境生效；其他环境强制关闭并告警。
        self._capture_content = cfg.capture_content and app_env == "development"
        if cfg.capture_content and not self._capture_content:
            logger.warning(
                "LANGFUSE_CAPTURE_CONTENT=true 仅在 APP_ENV=development 生效，"
                "当前环境 %s 已强制关闭内容捕获", app_env,
            )
        if self._capture_content:
            logger.warning(
                "Langfuse 内容捕获已开启（仅限开发环境）：内容属性经 redact_pii "
                "统一脱敏后上送，禁止用于生产"
            )

        if not self._enabled:
            # 整体禁用（含缺 key / host 非法）：不持有任何 client，业务零感知
            self._client = None
        elif client is not None:
            # 依赖注入（测试/特殊部署）：调用方保证 client 与配置一致。
            self._client = client
        else:
            self._client = self._init_client()

    # ── 对外只读状态 ────────────────────────────────────────────────────

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def capture_content(self) -> bool:
        return self._capture_content

    # ── 初始化与 fail-open ──────────────────────────────────────────────

    def _init_client(self) -> Any | None:
        """构造 Langfuse client。SDK 采样固定 1.0，采样决策集中在 adapter。"""
        try:
            from langfuse import Langfuse as LangfuseClient

            return LangfuseClient(
                public_key=self._cfg.public_key,
                secret_key=self._cfg.secret_key.get_secret_value(),
                host=self._cfg.host,
                tracing_enabled=True,
                sample_rate=1.0,
                environment=self._app_env,
                mask_otel_spans=self._mask_otel_spans,
                should_export_span=self._should_export_span,
            )
        except Exception as exc:
            self._record_export_failure(exc)
            logger.error("Langfuse client 初始化失败，Langfuse 已禁用：%s", exc)
            return None

    @staticmethod
    def _classify_failure(exc: BaseException) -> str:
        """将异常归类为 timeout/auth/exception 计数标签（尽力分类）。"""
        names = (type(exc).__name__ + " " + str(exc)).lower()
        if "timeout" in names:
            return "timeout"
        if "auth" in names or "unauthorized" in names or "401" in names or "403" in names:
            return "auth"
        return "exception"

    def _record_export_failure(self, exc: BaseException) -> None:
        """记录 adapter 可见的失败；export 阶段的网络错误由 SDK 批处理线程
        内部消化（OTLPSpanExporter 返回 FAILURE 而非抛出），不会到达这里。"""
        try:
            self._metrics.langfuse_export_errors_total.inc(
                labels={"reason": self._classify_failure(exc)}
            )
        except Exception:  # pragma: no cover - 指标自身绝不影响业务
            logger.debug("langfuse failure metric inc failed", exc_info=True)
        logger.warning("Langfuse 调用失败（业务不受影响）：%s: %s",
                       type(exc).__name__, exc)

    def _record_dropped(self, reason: str) -> None:
        try:
            self._metrics.langfuse_dropped_total.inc(labels={"reason": reason})
        except Exception:  # pragma: no cover
            logger.debug("langfuse drop metric inc failed", exc_info=True)

    # ── metadata 白名单 ─────────────────────────────────────────────────

    def filter_metadata(
        self, raw: Mapping[str, Any] | None
    ) -> LangfuseTraceMetadata:
        """按白名单过滤 metadata：未登记的键一律丢弃，标量类型校验。

        这是 Langfuse metadata 的唯一入口——新增字段必须改
        ``LangfuseTraceMetadata``，嵌套结构与其他类型不允许通过。
        """
        filtered: LangfuseTraceMetadata = {}
        if not raw:
            return filtered
        for key, value in raw.items():
            if key not in _METADATA_WHITELIST:
                logger.debug("langfuse metadata 非白名单键已丢弃：%s", key)
                continue
            # bool 需在 int 之前判断（bool 是 int 子类）
            if isinstance(value, (bool, str, int, float)):
                filtered[key] = value  # type: ignore[typeddict-item]
            else:
                logger.debug("langfuse metadata 非标量值已丢弃：%s", key)
        return filtered

    # ── 采样 ────────────────────────────────────────────────────────────

    def _should_sample(self, is_error: bool) -> bool:
        if is_error:
            return True  # 错误请求不参与采样丢弃
        rate = self._cfg.sample_rate
        if rate >= 1.0:
            return True
        if rate <= 0.0:
            return False
        return self._sampler() < rate

    # ── trace 注册与导出过滤 ────────────────────────────────────────────

    def _register_trace(self, trace_id: str | None) -> None:
        if not trace_id:
            return
        with self._registry_lock:
            self._trace_registry[trace_id] = True
            self._trace_registry.move_to_end(trace_id)
            while len(self._trace_registry) > _TRACE_REGISTRY_MAX:
                self._trace_registry.popitem(last=False)

    def _should_export_span(self, span: Any) -> bool:
        """只导出 adapter 登记过的 trace 的 span——越界/杂散 span 不出进程。"""
        try:
            trace_id = f"{span.context.trace_id:032x}"
        except Exception:
            return False
        with self._registry_lock:
            return trace_id in self._trace_registry

    # ── export 阶段统一脱敏 ─────────────────────────────────────────────

    @staticmethod
    def _sanitize_text(text: str) -> str:
        """统一脱敏器：所有进入 Langfuse 的内容文本必须经过这里。"""
        try:
            return redact_pii(text)[0]
        except Exception:
            # 脱敏器自身失败时整段替换，绝不放行原文（fail-closed）
            return "[REDACTED]"

    def _mask_otel_spans(self, *, params: Any) -> Any:
        """导出前兜底：默认删除内容属性；capture_content 时脱敏后保留。

        覆盖 langchain CallbackHandler 自动捕获的原文（该 handler 自身无
        mask/hide-input 机制）。函数在 SDK 批处理线程上运行，必须确定性且
        快速。fail-closed 语义：属性不可读时抛出异常，由 SDK 丢弃整个导出
        批次（langfuse.types.MaskOtelSpansResult 的批次丢弃契约）——宁可
        丢数据，不放行未脱敏原文；业务路径不受影响（export 为异步批处理）。
        """
        from langfuse.types import MaskOtelSpansResult, OtelSpanPatch

        patches: dict[Any, OtelSpanPatch] = {}
        for identifier, span in params.spans.items():
            try:
                attrs = dict(span.attributes or {})
            except Exception:
                # 属性不可读时无法保证脱敏：放弃整个导出批次
                raise
            content_keys = [k for k in attrs if _is_content_attribute(k)]
            if not content_keys:
                continue
            if self._capture_content:
                set_attributes: dict[str, Any] = {}
                for key in content_keys:
                    value = attrs[key]
                    if isinstance(value, str):
                        set_attributes[key] = self._sanitize_text(value)
                    elif isinstance(value, (bool, int, float)):
                        set_attributes[key] = value
                    # 其余类型无法安全脱敏，不写入 set 即删除
                patches[identifier] = OtelSpanPatch(set_attributes=set_attributes)
            else:
                patches[identifier] = OtelSpanPatch(
                    delete_attributes=tuple(content_keys)
                )
        return MaskOtelSpansResult(span_patches=patches)

    # ── 业务入口 ────────────────────────────────────────────────────────

    def start_request_trace(
        self,
        request_id: str,
        thread_id: str,
        *,
        metadata: Mapping[str, Any] | None = None,
        is_error: bool = False,
    ) -> _RequestTrace:
        """创建请求根 trace。未启用/未采样/失败时返回 no-op 句柄。"""
        if not self._enabled or self._client is None:
            return _RequestTrace(self, None, None, None, sampled=False)
        try:
            if not self._should_sample(is_error):
                self._record_dropped("sampled_out")
                return _RequestTrace(
                    self,
                    None,
                    None,
                    None,
                    sampled=False,
                    request_id=request_id,
                    thread_id=thread_id,
                )
            filtered = self.filter_metadata(metadata)
            filtered["request_id"] = request_id
            filtered["thread_id"] = thread_id
            filtered["status"] = "error" if is_error else "ok"
            span = self._client.start_observation(
                name=ERROR_TRACE_NAME if is_error else ROOT_TRACE_NAME,
                as_type="span",
                metadata=filtered,
                level="ERROR" if is_error else "DEFAULT",
            )
            self._register_trace(span.trace_id)
            return _RequestTrace(
                self,
                span,
                span.trace_id,
                span.id,
                sampled=True,
                request_id=request_id,
                thread_id=thread_id,
            )
        except Exception as exc:
            self._record_export_failure(exc)
            return _RequestTrace(self, None, None, None, sampled=False)

    def get_callback_handler(self, trace: _RequestTrace) -> Any | None:
        """返回绑定到根 trace 的 LangChain CallbackHandler（未采样时 None）。

        handler 内部经 ``get_client()`` 复用本 adapter 初始化的 client
        （``Langfuse()`` 构造时登记到 ``LangfuseResourceManager._instances``，
        mask/should_export_span 随实例保留），因此其自动捕获的原文也会被
        export 层统一脱敏。调用方将返回值放入 ``RunnableConfig.callbacks``。
        """
        if not self._enabled or trace.span_id is None:
            return None
        try:
            from langfuse.langchain import CallbackHandler

            return CallbackHandler(
                trace_context={
                    "trace_id": trace.trace_id,
                    "parent_span_id": trace.span_id,
                }
            )
        except Exception as exc:
            self._record_export_failure(exc)
            return None

    def _create_error_only_trace(
        self,
        *,
        error_type: str | None,
        metadata: Mapping[str, Any] | None,
        request_id: str = "",
        thread_id: str = "",
    ) -> None:
        """为被采样掉但失败的请求补建错误 trace（仅白名单 metadata）。"""
        if not self._enabled or self._client is None:
            return
        try:
            filtered = self.filter_metadata(metadata)
            if request_id:
                filtered["request_id"] = request_id
            if thread_id:
                filtered["thread_id"] = thread_id
            if error_type:
                filtered["error_type"] = error_type
            filtered["status"] = "error"
            span = self._client.start_observation(
                name=ERROR_TRACE_NAME,
                as_type="span",
                metadata=filtered,
                level="ERROR",
                status_message=error_type,
            )
            self._register_trace(span.trace_id)
            span.end()
        except Exception as exc:
            self._record_export_failure(exc)

    def flush(self) -> None:
        """测试/优雅退出用的手动 flush；请求热路径不得调用（会阻塞）。"""
        if self._client is None:
            return
        try:
            self._client.flush()
        except Exception as exc:
            self._record_export_failure(exc)


# 全局单例——与 metrics.get_metrics 相同的双重检查锁模式
_langfuse_adapter: LangfuseAdapter | None = None
_langfuse_adapter_lock = threading.Lock()


def get_langfuse() -> LangfuseAdapter:
    """获取全局 Langfuse adapter 单例（未启用时为 no-op 实现）。"""
    global _langfuse_adapter
    if _langfuse_adapter is None:
        with _langfuse_adapter_lock:
            if _langfuse_adapter is None:
                from src.config import config

                _langfuse_adapter = LangfuseAdapter(
                    cfg=config.langfuse, app_env=config.app_env
                )
    return _langfuse_adapter


def reset_langfuse_singleton() -> None:
    """重置单例，仅供测试使用。"""
    global _langfuse_adapter
    with _langfuse_adapter_lock:
        _langfuse_adapter = None
