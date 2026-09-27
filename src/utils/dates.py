"""日期解析工具：把入库元数据与查询时间范围统一转为数值 date_day（yyyymmdd）。

issues.md 一.6：Chroma 1.5 的 where 查询要求 $gt/$gte/$lt/$lte 的操作数是数值，
且同一字段表达式只能有一个操作符。入库时写入数值 date_day 字段，
查询时拆成两个 $and 条件，双端都使用本模块解析。
"""

from __future__ import annotations

import re
from datetime import date, datetime

# 支持：ISO 日期/日期时间、2024/1/1、2024.1.1、20240101、2024年1月1日、2024年、2024
_DATE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^(\d{4})[-/.年](\d{1,2})[-/.月](\d{1,2})日?$"),
    re.compile(r"^(\d{4})(\d{2})(\d{2})$"),
    re.compile(r"^(\d{4})[-/.年](\d{1,2})月?$"),
    re.compile(r"^(\d{4})年?$"),
)


def parse_date_day(value: object) -> int | None:
    """把日期字符串/数值解析为 yyyymmdd 整数；无法解析返回 None。

    时间部分（T00:00:00、时区）会被忽略。
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.year * 10000 + value.month * 100 + value.day
    if isinstance(value, date):
        return value.year * 10000 + value.month * 100 + value.day
    if isinstance(value, int | float):
        text = str(int(value))
    elif isinstance(value, str):
        text = value.strip()
        # 去掉时间部分：2024-01-01T00:00:00 / 2024-01-01 00:00:00
        text = re.split(r"[T ]", text, maxsplit=1)[0]
    else:
        return None

    if not text:
        return None

    for pattern in _DATE_PATTERNS:
        match = pattern.match(text)
        if match is None:
            continue
        lastindex = match.lastindex or 0
        year = int(match.group(1))
        month = int(match.group(2)) if lastindex >= 2 else 1
        day = int(match.group(3)) if lastindex >= 3 else 1
        try:
            date(year, month, day)
        except ValueError:
            return None
        return year * 10000 + month * 100 + day
    return None
