"""全链路测试案例：环节 C 检索（TC-011~TC-015）。

HybridRetriever 的角色/权限/容错逻辑保持真实，底层向量检索与 BM25
替换为 stub，不触碰生产 ChromaDB。
"""

from __future__ import annotations

import sys
from typing import Any

import pytest

from src.retrieval import hybrid_retriever as hr_module
from src.retrieval.hybrid_retriever import HybridRetriever
from src.schemas.constants import (
    GRADE_TOP_K,
    META_CHUNK_ID,
    META_ERROR,
    META_PERMISSION_LEVEL,
    META_RRF_SCORE,
    META_SOURCE,
    META_TITLE,
    PERMISSION_CONFIDENTIAL,
    PERMISSION_INTERNAL,
    PERMISSION_PUBLIC,
    PLAN_QUERY,
    PLAN_SOURCE,
    PLAN_TOP_K,
    RETRIEVAL_MIN_SCORE,
    RR_CONTENT,
    RR_DENIED,
    RR_METADATA,
    RR_SCORE,
    ROLE_ADVISOR,
    ROLE_COMPLIANCE,
    SOURCE_FAQ,
    SOURCE_PRODUCT,
)
from src.schemas.typed_dicts import RetrievalPlanStep

PUBLIC_CHUNK = "本基金风险等级为R1（低风险），适合保守型投资者。"


def plan_step(source: str, query: str = "货币基金风险等级", top_k: int = 3) -> RetrievalPlanStep:
    return RetrievalPlanStep(source=source, query=query, top_k=top_k)


class StubSourceRetriever:
    """固定结果的源检索器替身。"""

    def __init__(self, results: list[dict[str, Any]] | Exception):
        self._results = results

    def retrieve(self, query: str, top_k: int, filters: dict | None = None) -> list[dict[str, Any]]:
        if isinstance(self._results, Exception):
            raise self._results
        return list(self._results)[:top_k]


def make_result(
    content: str = PUBLIC_CHUNK,
    score: float = 0.9,
    permission_level: str = PERMISSION_PUBLIC,
    allowed_roles: list[str] | str | None = None,
    source: str = "data/raw/reports/fund.html",
    chunk_id: str = "c1",
    **extra_meta: Any,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        META_SOURCE: source,
        META_TITLE: "财报",
        META_CHUNK_ID: chunk_id,
        META_PERMISSION_LEVEL: permission_level,
    }
    if allowed_roles is not None:
        metadata["allowed_roles"] = allowed_roles
    metadata.update(extra_meta)
    return {RR_CONTENT: content, RR_METADATA: metadata, RR_SCORE: score}


@pytest.fixture()
def stub_factories(monkeypatch):
    """替换源检索器工厂 + BM25，杜绝真实向量库访问。"""
    factory_results: dict[str, list[dict[str, Any]] | Exception] = {}
    monkeypatch.setattr(
        hr_module,
        "_SOURCE_RETRIEVER_FACTORIES",
        {
            SOURCE_PRODUCT: lambda engine: StubSourceRetriever(factory_results.get(SOURCE_PRODUCT, [])),
            SOURCE_FAQ: lambda engine: StubSourceRetriever(factory_results.get(SOURCE_FAQ, [])),
        },
    )
    monkeypatch.setattr(hr_module, "ChromaVectorRetriever", lambda: object())
    monkeypatch.setattr(hr_module, "BM25Retriever", lambda engine: None)
    return factory_results


# ══════════════════════════════════════════════════════════════════════
# TC-011 计划级越权数据源拦截
# ══════════════════════════════════════════════════════════════════════


def test_tc011_plan_level_source_permission_filtering(stub_factories):
    """TC-011：advisor 计划含 faq_search → denied 结果；product_search 正常执行。"""
    stub_factories[SOURCE_PRODUCT] = [make_result(source="p1.html", chunk_id="p1")]
    retriever = HybridRetriever(user_role=ROLE_ADVISOR, data_permissions=[PERMISSION_PUBLIC])

    results = retriever.retrieve([plan_step(SOURCE_FAQ), plan_step(SOURCE_PRODUCT)])

    denied = [r for r in results if r.get(RR_DENIED)]
    usable = [r for r in results if not r.get(RR_DENIED)]
    assert len(denied) == 1
    assert denied[0][RR_METADATA][META_SOURCE] == SOURCE_FAQ
    assert "无权限" in denied[0]["reason"]
    assert len(usable) == 1
    assert usable[0][RR_METADATA][META_SOURCE] == "p1.html"


# ══════════════════════════════════════════════════════════════════════
# TC-012 结果级权限过滤
# ══════════════════════════════════════════════════════════════════════


def test_tc012_result_level_permission_filtering(stub_factories):
    """TC-012：permission_level/allowed_roles 组合的结果级过滤契约。

    1) confidential 对 advisor → denied
    2) internal 无 allowed_roles → 默认拒绝
    3) public 无 allowed_roles → 放行
    4) internal 且 allowed_roles 含 advisor → 放行
    5) allowed_roles 为逗号字符串且不含 advisor → denied
    """
    retriever = HybridRetriever(
        user_role=ROLE_ADVISOR,
        data_permissions=[PERMISSION_PUBLIC, PERMISSION_INTERNAL],
    )
    results = [
        make_result(source="conf.html", permission_level=PERMISSION_CONFIDENTIAL),
        make_result(source="no_roles.html", permission_level=PERMISSION_INTERNAL),
        make_result(source="pub.html", permission_level=PERMISSION_PUBLIC),
        make_result(
            source="ok.html",
            permission_level=PERMISSION_INTERNAL,
            allowed_roles=[ROLE_ADVISOR, ROLE_COMPLIANCE],
        ),
        make_result(source="str_roles.html", permission_level=PERMISSION_INTERNAL,
                    allowed_roles="compliance,technical"),
    ]

    filtered = retriever._filter_results_by_role(results)

    verdicts = {f[RR_METADATA][META_SOURCE]: not f.get(RR_DENIED) for f in filtered}
    assert verdicts == {
        "conf.html": False,
        "no_roles.html": False,
        "pub.html": True,
        "ok.html": True,
        "str_roles.html": False,
    }
    for f in filtered:
        if f.get(RR_DENIED):
            assert f[RR_CONTENT] == ""
            assert f[RR_SCORE] == 0.0


# ══════════════════════════════════════════════════════════════════════
# TC-013 BM25 失败静默降级
# ══════════════════════════════════════════════════════════════════════


def test_tc013_bm25_failure_degrades_to_vector_results(monkeypatch, stub_factories):
    """TC-013：BM25 检索抛异常 → 静默降级，向量结果原样返回且可用。"""
    stub_factories[SOURCE_PRODUCT] = [make_result(source="vec.html", chunk_id="v1")]

    class ExplodingBM25:
        def __init__(self, engine: Any) -> None:
            pass

        def retrieve(self, query: str, top_k: int, filters: dict | None = None) -> list:
            raise RuntimeError("BM25 index unavailable")

    monkeypatch.setattr(hr_module, "BM25Retriever", ExplodingBM25)

    retriever = HybridRetriever(
        user_role=ROLE_ADVISOR, data_permissions=[PERMISSION_PUBLIC, PERMISSION_INTERNAL]
    )
    results = retriever.retrieve([plan_step(SOURCE_PRODUCT)])

    usable = [r for r in results if not r.get(RR_DENIED)]
    assert len(usable) == 1
    assert usable[0][RR_CONTENT] == PUBLIC_CHUNK
    assert META_RRF_SCORE not in usable[0][RR_METADATA]
