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


# ══════════════════════════════════════════════════════════════════════
# TC-014 向量库不可用
# ══════════════════════════════════════════════════════════════════════


def test_tc014_vector_store_unavailable_yields_explicit_error_results(stub_factories):
    """TC-014：底层检索抛异常（ChromaDB 宕机）→ 显式错误结果，不崩溃、不返回空成功。"""
    stub_factories[SOURCE_PRODUCT] = RuntimeError("ChromaDB connection refused")

    retriever = HybridRetriever(
        user_role=ROLE_ADVISOR, data_permissions=[PERMISSION_PUBLIC, PERMISSION_INTERNAL]
    )
    results = retriever.retrieve([plan_step(SOURCE_PRODUCT)])

    assert len(results) == 1
    error_result = results[0]
    assert error_result.get(RR_DENIED) is not True, "错误结果不应伪装成权限拒绝"
    assert error_result[RR_METADATA].get(META_ERROR) is not None, "必须携带显式错误信息"
    assert "ChromaDB connection refused" in error_result[RR_METADATA][META_ERROR]
    assert error_result[RR_CONTENT] == "", "失败检索不得返回看似可用的内容"


# ══════════════════════════════════════════════════════════════════════
# TC-015 相关性过滤与重排降级
# ══════════════════════════════════════════════════════════════════════


def _grade_state(results: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "retrieval_results": results,
        "resolved_query": "货币基金风险等级",
        "rewritten_query": "货币基金 风险等级",
        "original_query": "货币基金风险等级",
        "user_role": ROLE_ADVISOR,
    }


def test_tc015_grade_filters_low_scores_dedupes_and_marks_reranker_unavailable(
    monkeypatch,
):
    """TC-015：低分过滤、同 source+chunk 去重、top-10 截断、reranker 不可用显式标记。"""
    from src.agents.nodes import grade_and_filter

    monkeypatch.setitem(sys.modules, "src.tools.rerank", None)  # ImportError → unavailable

    results = [
        make_result(source="a.html", chunk_id="dup", score=0.95),
        make_result(source="a.html", chunk_id="dup", score=0.93),  # 重复证据
        make_result(source="b.html", chunk_id="b1", score=RETRIEVAL_MIN_SCORE + 0.1),
        make_result(source="c.html", chunk_id="c1", score=0.2),  # 低于阈值
    ]
    results += [
        make_result(source=f"bulk{i}.html", chunk_id=f"bk{i}", score=0.9 - i * 0.01)
        for i in range(GRADE_TOP_K + 5)  # 撑爆候选池验证 top-k 截断
    ]

    update = grade_and_filter(_grade_state(results))

    filtered = [r for r in update["retrieval_results"] if not r.get(RR_DENIED)]
    assert all(float(r[RR_SCORE]) >= RETRIEVAL_MIN_SCORE for r in filtered)
    keys = [(r[RR_METADATA][META_SOURCE], r[RR_METADATA][META_CHUNK_ID]) for r in filtered]
    assert len(keys) == len(set(keys)), "重复证据必须去重"
    assert len(filtered) <= GRADE_TOP_K
    assert update["reranker_status"] == "unavailable"


def test_tc015_reranker_applied_keeps_semantic_order(monkeypatch):
    """TC-015：reranker 可用时 status=applied，结果按语义重排输出。"""
    from src.agents.nodes import grade_and_filter

    class FakeRerankService:
        def rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
            # 语义重排：反转顺序模拟与原始分不同的语义序
            return list(reversed(candidates))[:top_k]

    from src.tools import rerank as rerank_module

    monkeypatch.setattr(rerank_module, "RerankService", FakeRerankService)

    results = [
        make_result(source="a.html", chunk_id="a1", score=0.95),
        make_result(source="b.html", chunk_id="b1", score=0.90),
        make_result(source="c.html", chunk_id="c1", score=0.85),
    ]

    update = grade_and_filter(_grade_state(results))

    order = [r[RR_METADATA][META_CHUNK_ID] for r in update["retrieval_results"]]
    assert order == ["c1", "b1", "a1"]
    assert update["reranker_status"] == "applied"
