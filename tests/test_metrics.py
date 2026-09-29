"""可观测性指标注册表单元测试（ISSUE-23：TTFT 指标）。"""

from src.utils.metrics import MetricsRegistry


def test_metrics_registry_tracks_time_to_first_token():
    registry = MetricsRegistry()

    registry.record_ttft(role="advisor", seconds=1.5)
    registry.record_ttft(role="advisor", seconds=3.0)

    assert registry.time_to_first_token.get_count() == 2
    assert registry.get_summary()["ttft_p95_seconds"] == 3.0
    assert "secrag_time_to_first_token_seconds" in registry.export_prometheus()


def test_metrics_registry_ttft_starts_empty():
    registry = MetricsRegistry()

    assert registry.time_to_first_token.get_count() == 0
    assert registry.get_summary()["ttft_p95_seconds"] == 0.0


def test_labeled_histogram_aggregates_across_labels_for_summary():
    """摘要分位数不带标签取值，必须跨角色聚合，否则恒为 0。"""
    registry = MetricsRegistry()

    registry.record_query(role="advisor", status="success", duration=2.0)
    registry.record_query(role="compliance", status="success", duration=4.0)

    assert registry.query_duration.get_count() == 2
    assert registry.get_summary()["latency_p95_seconds"] == 4.0
