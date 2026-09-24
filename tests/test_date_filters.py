"""date_day 解析与时间过滤契约测试（issues.md 一.6）。"""

from src.agents.nodes import _time_range_to_filters
from src.retrieval.bm25_retriever import BM25Retriever
from src.utils.dates import parse_date_day


def test_parse_date_day_supported_formats():
    assert parse_date_day("2024-01-01") == 20240101
    assert parse_date_day("2024-12-31") == 20241231
    assert parse_date_day("2024/1/5") == 20240105
    assert parse_date_day("20240105") == 20240105
    assert parse_date_day("2024年1月5日") == 20240105
    assert parse_date_day("2024年") == 20240101
    assert parse_date_day("2024") == 20240101
    assert parse_date_day("2024-01-05T00:00:00Z") == 20240105
    assert parse_date_day("2024-01-05 00:00:00") == 20240105
    assert parse_date_day(20240105) == 20240105


def test_parse_date_day_invalid_returns_none():
    assert parse_date_day("") is None
    assert parse_date_day("最近三个月") is None
    assert parse_date_day(None) is None
    assert parse_date_day("2024-13-01") is None  # 非法月份


def test_time_range_filters_are_chroma_compatible():
    """上下界必须拆成两个数值条件并用 $and 组合，不能塞进同一字段表达式。"""
    filters = _time_range_to_filters({"start": "2024-01-01", "end": "2024-12-31"})
    assert filters == {
        "$and": [
            {"date_day": {"$gte": 20240101}},
            {"date_day": {"$lte": 20241231}},
        ]
    }


def test_time_range_filters_partial_bounds():
    only_start = _time_range_to_filters({"start": "2024-01-01", "end": ""})
    assert only_start == {"date_day": {"$gte": 20240101}}

    only_end = _time_range_to_filters({"start": "", "end": "2024年12月"})
    assert only_end == {"date_day": {"$lte": 20241201}}


def test_time_range_filters_unparseable_returns_none():
    assert _time_range_to_filters({"start": "去年", "end": "最近三个月"}) is None
    assert _time_range_to_filters(None) is None
    assert _time_range_to_filters({}) is None


def test_bm25_match_filters_handles_and_combination():
    metadata = {"date_day": 20240615, "product_type": "fund"}
    filters = {
        "$and": [
            {"date_day": {"$gte": 20240101}},
            {"date_day": {"$lte": 20241231}},
            {"product_type": "fund"},
        ]
    }
    assert BM25Retriever._match_filters(metadata, filters) is True
    assert BM25Retriever._match_filters({"date_day": 20230615}, filters) is False
