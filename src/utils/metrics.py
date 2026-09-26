"""P1-5: 可观测性——轻量级指标采集与 Prometheus 导出。

收集 SecRAG 运行时的关键指标，支持 Prometheus 格式导出，
便于对接 Grafana 仪表盘监控。

设计要点：
- 线程安全：所有指标操作加锁
- 内存存储：进程内聚合，无需外部依赖
- Prometheus 兼容：/metrics 端点输出标准 Prometheus 文本格式
- 可扩展：预留 OpenTelemetry 接入点

指标清单：
- secrag_queries_total: 查询总数（按 role/status 标签）
- secrag_query_duration_seconds: 查询延迟直方图（P50/P95/P99）
- secrag_cache_hits_total: 缓存命中数
- secrag_cache_misses_total: 缓存未命中数
- secrag_retrieval_chunks_total: 检索返回 chunk 总数
- secrag_verification_passed_total: 验证通过数
- secrag_verification_failed_total: 验证失败数
- secrag_compliance_blocked_total: 合规拦截数
- secrag_active_requests: 当前活跃请求数（Gauge）
- secrag_langfuse_dropped_total: Langfuse adapter 丢弃的 trace/span（按 reason）
- secrag_langfuse_export_errors_total: Langfuse 客户端操作失败数（按 reason）
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from typing import Any


# ══════════════════════════════════════════════════════════════════════
# 指标基类
# ══════════════════════════════════════════════════════════════════════


class Metric:
    """指标基类。"""

    def __init__(self, name: str, help_text: str, labels: list[str] | None = None):
        self.name = name
        self.help_text = help_text
        self.labels = labels or []
        self._lock = threading.Lock()

    def _label_key(self, label_values: dict[str, str] | None) -> tuple:
        if not label_values:
            return ()
        return tuple(label_values.get(label, "") for label in self.labels)


class Counter(Metric):
    """计数器——只增不减。"""

    def __init__(self, name: str, help_text: str, labels: list[str] | None = None):
        super().__init__(name, help_text, labels)
        self._values: dict[tuple, float] = defaultdict(float)

    def inc(self, amount: float = 1.0, labels: dict[str, str] | None = None):
        """增加计数。"""
        key = self._label_key(labels)
        with self._lock:
            self._values[key] += amount

    def get(self, labels: dict[str, str] | None = None) -> float:
        """获取当前值。"""
        key = self._label_key(labels)
        with self._lock:
            return self._values.get(key, 0.0)

    def collect(self) -> list[tuple[dict[str, str], float]]:
        """收集所有标签组合的值。"""
        with self._lock:
            result = []
            for key, value in self._values.items():
                label_dict = dict(zip(self.labels, key)) if self.labels else {}
                result.append((label_dict, value))
            return result


class Gauge(Metric):
    """仪表盘——可增可减。"""

    def __init__(self, name: str, help_text: str, labels: list[str] | None = None):
        super().__init__(name, help_text, labels)
        self._values: dict[tuple, float] = defaultdict(float)

    def set(self, value: float, labels: dict[str, str] | None = None):
        """设置值。"""
        key = self._label_key(labels)
        with self._lock:
            self._values[key] = value

    def inc(self, amount: float = 1.0, labels: dict[str, str] | None = None):
        """增加。"""
        key = self._label_key(labels)
        with self._lock:
            self._values[key] += amount

    def dec(self, amount: float = 1.0, labels: dict[str, str] | None = None):
        """减少。"""
        key = self._label_key(labels)
        with self._lock:
            self._values[key] -= amount

    def get(self, labels: dict[str, str] | None = None) -> float:
        key = self._label_key(labels)
        with self._lock:
            return self._values.get(key, 0.0)

    def collect(self) -> list[tuple[dict[str, str], float]]:
        with self._lock:
            result = []
            for key, value in self._values.items():
                label_dict = dict(zip(self.labels, key)) if self.labels else {}
                result.append((label_dict, value))
            return result


class Histogram(Metric):
    """直方图——记录值分布，支持百分位计算。"""

    DEFAULT_BUCKETS = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]

    def __init__(
        self,
        name: str,
        help_text: str,
        labels: list[str] | None = None,
        buckets: list[float] | None = None,
    ):
        super().__init__(name, help_text, labels)
        self.buckets = sorted(buckets or self.DEFAULT_BUCKETS)
        self._counts: dict[tuple, int] = defaultdict(int)
        self._sums: dict[tuple, float] = defaultdict(float)
        self._bucket_counts: dict[tuple, list[int]] = defaultdict(
            lambda: [0] * len(self.buckets)
        )
        self._values: dict[tuple, list[float]] = defaultdict(list)

    def observe(self, value: float, labels: dict[str, str] | None = None):
        """记录一个观测值。"""
        key = self._label_key(labels)
        with self._lock:
            self._counts[key] += 1
            self._sums[key] += value
            # 找到第一个 >= value 的 bucket
            for i, bound in enumerate(self.buckets):
                if value <= bound:
                    self._bucket_counts[key][i] += 1
                    break
            else:
                # 超过最大 bucket，+Inf
                pass
            # 保留最近 1000 个值用于百分位计算
            vals = self._values[key]
            vals.append(value)
            if len(vals) > 1000:
                self._values[key] = vals[-1000:]

    def percentile(self, p: float, labels: dict[str, str] | None = None) -> float:
        """计算百分位（p: 0-100）。"""
        key = self._label_key(labels)
        with self._lock:
            vals = sorted(self._values.get(key, []))
            if not vals:
                return 0.0
            idx = int(len(vals) * p / 100)
            idx = min(idx, len(vals) - 1)
            return vals[idx]

    def get_count(self, labels: dict[str, str] | None = None) -> int:
        key = self._label_key(labels)
        with self._lock:
            return self._counts.get(key, 0)

    def get_sum(self, labels: dict[str, str] | None = None) -> float:
        key = self._label_key(labels)
        with self._lock:
            return self._sums.get(key, 0.0)

    def collect(self) -> list[tuple[dict[str, str], dict[str, Any]]]:
        with self._lock:
            result = []
            for key in self._counts:
                label_dict = dict(zip(self.labels, key)) if self.labels else {}
                cumulative = 0
                bucket_data = []
                for i, bound in enumerate(self.buckets):
                    cumulative += self._bucket_counts[key][i]
                    bucket_data.append((bound, cumulative))
                # +Inf bucket
                bucket_data.append((float("inf"), self._counts[key]))
                result.append(
                    (
                        label_dict,
                        {
                            "count": self._counts[key],
                            "sum": self._sums[key],
                            "buckets": bucket_data,
                        },
                    )
                )
            return result


# ══════════════════════════════════════════════════════════════════════
# SecRAG 指标注册表
# ══════════════════════════════════════════════════════════════════════


class MetricsRegistry:
    """SecRAG 指标注册表——集中管理所有指标。"""

    def __init__(self):
        self._metrics: dict[str, Metric] = {}
        self._lock = threading.Lock()
        self._start_time = time.time()
        self._init_metrics()

    def _init_metrics(self):
        """初始化所有指标。"""
        # 查询指标
        self.queries_total = Counter(
            "secrag_queries_total",
            "Total number of queries processed",
            labels=["role", "status"],
        )
        self.query_duration = Histogram(
            "secrag_query_duration_seconds",
            "Query processing duration in seconds",
            labels=["role"],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
        )
        self.active_requests = Gauge(
            "secrag_active_requests",
            "Number of active requests being processed",
        )

        # 缓存指标
        self.cache_hits_total = Counter(
            "secrag_cache_hits_total",
            "Total number of semantic cache hits",
        )
        self.cache_misses_total = Counter(
            "secrag_cache_misses_total",
            "Total number of semantic cache misses",
        )

        # 检索指标
        self.retrieval_chunks_total = Counter(
            "secrag_retrieval_chunks_total",
            "Total number of chunks retrieved",
            labels=["source"],
        )

        # 验证指标
        self.verification_passed_total = Counter(
            "secrag_verification_passed_total",
            "Total number of answers that passed verification",
        )
        self.verification_failed_total = Counter(
            "secrag_verification_failed_total",
            "Total number of answers that failed verification",
        )

        # 合规指标
        self.compliance_blocked_total = Counter(
            "secrag_compliance_blocked_total",
            "Total number of queries blocked by compliance check",
        )

        # Langfuse 观测链路指标——只计数，不参与业务判定（fail-open）
        self.langfuse_dropped_total = Counter(
            "secrag_langfuse_dropped_total",
            "Traces/spans dropped by the Langfuse adapter",
            labels=["reason"],
        )
        self.langfuse_export_errors_total = Counter(
            "secrag_langfuse_export_errors_total",
            "Langfuse failures seen by the adapter or the OTLP export pipeline",
            labels=["reason"],
        )

        # 注册所有指标
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, Metric):
                self._metrics[attr.name] = attr

    def record_query(
        self,
        role: str,
        status: str,
        duration: float,
        cached: bool = False,
    ):
        """记录一次查询的完整指标。

        Args:
            role: 用户角色
            status: 查询状态（success/error/timeout/blocked）
            duration: 处理耗时（秒）
            cached: 是否缓存命中
        """
        self.queries_total.inc(labels={"role": role, "status": status})
        self.query_duration.observe(duration, labels={"role": role})
        if cached:
            self.cache_hits_total.inc()
        else:
            self.cache_misses_total.inc()

    def get_summary(self) -> dict[str, Any]:
        """获取指标摘要（用于健康检查和快速查看）。"""
        total_queries = sum(v for _, v in self.queries_total.collect())
        success_queries = sum(
            v for labels, v in self.queries_total.collect()
            if labels.get("status") == "success"
        )
        cache_hits = self.cache_hits_total.get()
        cache_misses = self.cache_misses_total.get()
        total_cache = cache_hits + cache_misses
        cache_hit_rate = round(cache_hits / total_cache, 4) if total_cache > 0 else 0.0

        # 计算全局延迟百分位
        p50 = self.query_duration.percentile(50)
        p95 = self.query_duration.percentile(95)
        p99 = self.query_duration.percentile(99)

        uptime = round(time.time() - self._start_time, 1)

        return {
            "uptime_seconds": uptime,
            "total_queries": total_queries,
            "success_queries": success_queries,
            "success_rate": round(success_queries / total_queries, 4) if total_queries > 0 else 0.0,
            "active_requests": int(self.active_requests.get()),
            "cache_hits": int(cache_hits),
            "cache_misses": int(cache_misses),
            "cache_hit_rate": cache_hit_rate,
            "latency_p50_seconds": round(p50, 3),
            "latency_p95_seconds": round(p95, 3),
            "latency_p99_seconds": round(p99, 3),
            "verification_passed": int(self.verification_passed_total.get()),
            "verification_failed": int(self.verification_failed_total.get()),
            "compliance_blocked": int(self.compliance_blocked_total.get()),
        }

    def export_prometheus(self) -> str:
        """导出 Prometheus 文本格式。"""
        lines = []
        for name, metric in sorted(self._metrics.items()):
            lines.append(f"# HELP {name} {metric.help_text}")
            if isinstance(metric, Counter):
                lines.append(f"# TYPE {name} counter")
                for labels, value in metric.collect():
                    label_str = self._format_labels(labels)
                    lines.append(f"{name}{label_str} {value}")
            elif isinstance(metric, Gauge):
                lines.append(f"# TYPE {name} gauge")
                for labels, value in metric.collect():
                    label_str = self._format_labels(labels)
                    lines.append(f"{name}{label_str} {value}")
            elif isinstance(metric, Histogram):
                lines.append(f"# TYPE {name} histogram")
                for labels, data in metric.collect():
                    for bound, count in data["buckets"]:
                        bucket_labels = dict(labels)
                        bucket_labels["le"] = str(bound) if bound != float("inf") else "+Inf"
                        label_str = self._format_labels(bucket_labels)
                        lines.append(f'{name}_bucket{label_str} {count}')
                    label_str = self._format_labels(labels)
                    lines.append(f'{name}_sum{label_str} {data["sum"]}')
                    lines.append(f'{name}_count{label_str} {data["count"]}')
        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _format_labels(labels: dict[str, str]) -> str:
        """格式化 Prometheus 标签。"""
        if not labels:
            return ""
        parts = [f'{k}="{v}"' for k, v in sorted(labels.items())]
        return "{" + ",".join(parts) + "}"


# 全局单例
_metrics_registry: MetricsRegistry | None = None
_metrics_lock = threading.Lock()


def get_metrics() -> MetricsRegistry:
    """获取指标注册表单例。"""
    global _metrics_registry
    if _metrics_registry is None:
        with _metrics_lock:
            if _metrics_registry is None:
                _metrics_registry = MetricsRegistry()
    return _metrics_registry
