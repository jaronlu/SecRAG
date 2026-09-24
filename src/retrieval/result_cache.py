"""检索结果 TTL 缓存（进程内）。

独立成模块（issues.md 二.1）：入库服务与 Agent 节点共享同一份缓存视图，
知识库发布新版本后调用 invalidate_retrieval_caches() 统一失效，
不让节点缓存继续返回旧结果。避免入库侧为清缓存而导入整个 Agent 模块。
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING

from src.schemas.constants import (
    PLAN_FILTERS,
    PLAN_QUERY,
    PLAN_SOURCE,
    PLAN_TOP_K,
    RETRIEVAL_CACHE_TTL_SECONDS,
)

if TYPE_CHECKING:
    from src.schemas.typed_dicts import RetrievalPlanStep, RetrievalResult

# key=(role, plan_fingerprint), value=(timestamp, results)
_retrieval_cache: dict[str, tuple[float, list[RetrievalResult]]] = {}


def plan_fingerprint(plan: list[RetrievalPlanStep]) -> str:
    """生成检索计划的稳定指纹，用于缓存 key。"""
    parts = []
    for step in plan:
        filters = step.get(PLAN_FILTERS)
        filter_str = json.dumps(filters, sort_keys=True, ensure_ascii=False) if filters else ""
        parts.append(f"{step.get(PLAN_SOURCE)}:{step.get(PLAN_QUERY)}:{step.get(PLAN_TOP_K)}:{filter_str}")
    return "|".join(parts)


def get_cached_plan(user_role: str, plan: list[RetrievalPlanStep]) -> list[RetrievalResult] | None:
    """命中未过期缓存时返回结果副本，未命中或已过期返回 None。"""
    cache_key = f"{user_role}:{plan_fingerprint(plan)}"
    cached = _retrieval_cache.get(cache_key)
    if cached is None:
        return None
    timestamp, results = cached
    if time.time() - timestamp < RETRIEVAL_CACHE_TTL_SECONDS:
        return list(results)
    _retrieval_cache.pop(cache_key, None)
    return None


def store_cached_plan(
    user_role: str,
    plan: list[RetrievalPlanStep],
    results: list[RetrievalResult],
) -> None:
    _retrieval_cache[f"{user_role}:{plan_fingerprint(plan)}"] = (time.time(), list(results))


def invalidate_retrieval_caches() -> None:
    """入库/知识库变更后统一失效进程内检索缓存（issues.md 二.1）。

    同时清理 BM25 索引缓存与检索计划 TTL 缓存，
    避免文档更新、删除或授权变化后继续返回旧结果。
    """
    from src.retrieval.bm25_retriever import invalidate_bm25_cache

    invalidate_bm25_cache()
    _retrieval_cache.clear()
