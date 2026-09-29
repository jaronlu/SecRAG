"""HybridRetriever 单元测试（mock 领域检索器，避免真实 ChromaDB）。"""

from __future__ import annotations

from typing import Any, Dict, Optional, cast

import pytest

from src.agents.state import AssistantState
from src.retrieval.base import BaseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.schemas.constants import (
    META_ALLOWED_ROLES,
    META_CHUNK_ID,
    META_ERROR,
    META_PERMISSION_LEVEL,
    META_SOURCE,
    PERMISSION_OVERFETCH_FACTOR,
    PLAN_FILTERS,
    PLAN_QUERY,
    PLAN_SOURCE,
    PLAN_TOP_K,
    ROLE_ADVISOR,
    ROLE_COMPLIANCE,
    ROLE_TECHNICAL,
    RR_CONTENT,
    RR_DENIED,
    RR_METADATA,
    RR_REASON,
    RR_SCORE,
    SOURCE_FAQ,
    SOURCE_PRODUCT,
    SOURCE_REGULATION,
    SOURCE_REPORT,
)
from src.schemas.typed_dicts import RetrievalPlanStep, RetrievalResult


class FakeRetriever(BaseRetriever):
    def __init__(self):
        self.calls: list[dict] = []

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict] = None,
    ) -> list[RetrievalResult]:
        self.calls.append({"query": query, "top_k": top_k, "filters": filters})
        return [
            {
                RR_CONTENT: f"{query}:{top_k}",
                RR_METADATA: {META_SOURCE: "fake"},
                RR_SCORE: 0.9,
            }
        ]


class FailingRetriever(BaseRetriever):
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict] = None,
    ) -> list[RetrievalResult]:
        raise RuntimeError("boom")


class RoleTaggedRetriever(BaseRetriever):
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict] = None,
    ) -> list[RetrievalResult]:
        return [
            {
                RR_CONTENT: "tech-only",
                RR_METADATA: {META_SOURCE: "fake", META_ALLOWED_ROLES: ROLE_TECHNICAL},
                RR_SCORE: 0.9,
            },
            {
                RR_CONTENT: "advisor-only",
                RR_METADATA: {META_SOURCE: "fake", META_ALLOWED_ROLES: ROLE_ADVISOR},
                RR_SCORE: 0.8,
            },
            {
                RR_CONTENT: "untagged",
                RR_METADATA: {META_SOURCE: "fake"},
                RR_SCORE: 0.7,
            },
        ]


class RestrictedRetriever(BaseRetriever):
    def retrieve(self, query: str, top_k: int = 5, filters: Optional[Dict] = None) -> list[RetrievalResult]:
        return [
            {
                RR_CONTENT: "confidential",
                RR_METADATA: {
                    META_SOURCE: "restricted",
                    META_PERMISSION_LEVEL: "confidential",
                    META_ALLOWED_ROLES: [ROLE_COMPLIANCE],
                },
                RR_SCORE: 0.9,
            }
        ]


class TestHybridRetriever:
    @pytest.fixture(autouse=True)
    def _disable_bm25(self, monkeypatch):
        """单元测试默认关闭 BM25，避免依赖本机 Chroma 数据；BM25 行为在
        test_bm25_results_kept_with_realistic_source_metadata 中单独验证。"""
        monkeypatch.setattr(HybridRetriever, "_get_bm25_retriever", lambda self: None)

    def test_executes_allowed_plan_and_passes_arguments(self):
        fake = FakeRetriever()
        retriever = HybridRetriever(user_role=ROLE_ADVISOR)
        retriever._retriever_cache[SOURCE_PRODUCT] = fake

        results = retriever.retrieve([
            {
                PLAN_SOURCE: SOURCE_PRODUCT,
                PLAN_QUERY: "风险等级",
                PLAN_TOP_K: 3,
                PLAN_FILTERS: {"product_type": "fund"},
            }
        ])

        fetch_k = 3 * PERMISSION_OVERFETCH_FACTOR
        assert results[0][RR_CONTENT] == f"风险等级:{fetch_k}"
        assert fake.calls == [
            {
                "query": "风险等级",
                "top_k": fetch_k,
                "filters": {"product_type": "fund"},
            }
        ]

    def test_denies_source_not_allowed_for_unmapped_role(self):
        retriever = HybridRetriever(user_role="unknown")

        results = retriever.retrieve([
            {
                PLAN_SOURCE: SOURCE_REPORT,
                PLAN_QUERY: "内部研报摘要",
                PLAN_TOP_K: 5,
            }
        ])

        assert len(results) == 1
        assert results[0].get(RR_DENIED) is True
        assert "无权限访问" in (results[0].get(RR_REASON) or "")
        assert results[0][RR_METADATA][META_SOURCE] == SOURCE_REPORT

    def test_unknown_role_fails_closed(self):
        retriever = HybridRetriever(user_role="unknown")

        results = retriever.retrieve([
            {
                PLAN_SOURCE: SOURCE_FAQ,
                PLAN_QUERY: "操作流程",
            }
        ])

        assert results[0].get(RR_DENIED) is True
        assert results[0][RR_CONTENT] == ""

    def test_unknown_source_returns_error_result(self):
        retriever = HybridRetriever(user_role=ROLE_TECHNICAL)

        results = retriever.retrieve([
            {
                PLAN_SOURCE: "unknown_search",
                PLAN_QUERY: "测试",
            }
        ])

        assert results[0][RR_SCORE] == 0.0
        assert results[0][RR_CONTENT] == ""
        assert results[0].get(RR_REASON) == "未知检索源"
        assert results[0][RR_METADATA][META_ERROR] == "未知检索源: unknown_search"

    def test_retriever_exception_returns_error_result(self):
        retriever = HybridRetriever(user_role=ROLE_ADVISOR)
        retriever._retriever_cache[SOURCE_PRODUCT] = FailingRetriever()

        results = retriever.retrieve([
            {
                PLAN_SOURCE: SOURCE_PRODUCT,
                PLAN_QUERY: "风险等级",
            }
        ])

        assert results[0][RR_SCORE] == 0.0
        assert results[0][RR_CONTENT] == ""
        assert results[0].get(RR_REASON) == "检索失败"
        assert results[0][RR_METADATA][META_ERROR] == "boom"

    def test_merges_multiple_allowed_sources(self):
        product = FakeRetriever()
        faq = FakeRetriever()
        retriever = HybridRetriever(user_role=ROLE_TECHNICAL)
        retriever._retriever_cache[SOURCE_PRODUCT] = product
        retriever._retriever_cache[SOURCE_FAQ] = faq

        results = retriever.retrieve([
            {PLAN_SOURCE: SOURCE_PRODUCT, PLAN_QUERY: "产品", PLAN_TOP_K: 2},
            {PLAN_SOURCE: SOURCE_FAQ, PLAN_QUERY: "FAQ", PLAN_TOP_K: 1},
        ])

        results = [
            r
            for r in results
            if not r.get(RR_DENIED)
        ]
        assert sorted(r[RR_CONTENT] for r in results) == [
            f"FAQ:{1 * PERMISSION_OVERFETCH_FACTOR}",
            f"产品:{2 * PERMISSION_OVERFETCH_FACTOR}",
        ]

    def test_chunk_permission_denial_preserves_safe_placeholder(self):
        fake = FakeRetriever()
        retriever = HybridRetriever(user_role=ROLE_TECHNICAL)
        retriever._retriever_cache[SOURCE_FAQ] = fake
        retriever._retriever_cache[SOURCE_REGULATION] = RestrictedRetriever()

        results = retriever.retrieve([
            {PLAN_SOURCE: SOURCE_FAQ, PLAN_QUERY: "FAQ"},
            {PLAN_SOURCE: SOURCE_REGULATION, PLAN_QUERY: "法规"},
        ])

        assert results[0][RR_CONTENT] == f"FAQ:{5 * PERMISSION_OVERFETCH_FACTOR}"
        assert results[1].get(RR_DENIED) is True
        assert "confidential" not in results[1][RR_CONTENT]

    def test_filters_allowed_roles_metadata_within_allowed_source(self):
        retriever = HybridRetriever(user_role=ROLE_TECHNICAL)
        retriever._retriever_cache[SOURCE_FAQ] = RoleTaggedRetriever()

        results = retriever.retrieve([{PLAN_SOURCE: SOURCE_FAQ, PLAN_QUERY: "LangGraph"}])

        usable = [result for result in results if not result.get(RR_DENIED)]
        denied = [result for result in results if result.get(RR_DENIED)]
        assert [r[RR_CONTENT] for r in usable] == ["tech-only", "untagged"]
        assert len(denied) == 1

    def test_domain_retrievers_share_one_vector_engine(self, monkeypatch):
        created_engines = []

        def fake_vector_engine():
            engine = FakeRetriever()
            created_engines.append(engine)
            return engine

        monkeypatch.setattr(
            "src.retrieval.hybrid_retriever.ChromaVectorRetriever",
            fake_vector_engine,
        )
        retriever = HybridRetriever(user_role=ROLE_ADVISOR)

        product = retriever._get_retriever(SOURCE_PRODUCT)
        faq = retriever._get_retriever(SOURCE_FAQ)

        assert len(created_engines) == 1
        assert product is not None and faq is not None
        # _engine 是领域检索器包装的私有属性，经 getattr 断言共享同一向量引擎
        assert getattr(product, "_engine") is created_engines[0]
        assert getattr(faq, "_engine") is created_engines[0]


def test_bm25_results_kept_with_realistic_source_metadata():
    """issues.md 一.4：BM25 来源过滤必须用 retrieval_source 精确匹配。

    metadata.source 是文件路径（如 data/raw/reports/example.pdf），
    旧实现检查路径是否包含 "report_search"，会误杀合法 BM25 命中。
    放在类外：TestHybridRetriever 的 autouse fixture 会禁用 BM25。
    """
    vector_result = {
        RR_CONTENT: "vector hit",
        RR_METADATA: {
            META_SOURCE: "data/raw/reports/example.pdf",
            "retrieval_source": "report_search",
        },
        RR_SCORE: 0.9,
    }
    bm25_result = {
        RR_CONTENT: "bm25 hit 股票代码 600519",
        RR_METADATA: {
            META_SOURCE: "data/raw/reports/example2.pdf",
            "retrieval_source": "report_search",
        },
        RR_SCORE: 7.3,
    }

    retriever = HybridRetriever(user_role=ROLE_ADVISOR)
    retriever._retriever_cache[SOURCE_REPORT] = _FakeVectorSource([vector_result])
    fake_bm25 = _FakeBM25([bm25_result])
    retriever._bm25_retriever = cast(Any, fake_bm25)

    plan: list[RetrievalPlanStep] = [
        {
            PLAN_SOURCE: SOURCE_REPORT,
            PLAN_QUERY: "贵州茅台 评级",
            PLAN_TOP_K: 5,
        }
    ]
    results = retriever.retrieve(plan)

    # BM25 收到精确的 retrieval_source 前置过滤
    bm25_filters = fake_bm25.calls[0]["filters"]
    assert bm25_filters.get("retrieval_source") == SOURCE_REPORT

    # BM25 独有结果没有因 metadata.source 是文件路径而被过滤掉
    contents = [r[RR_CONTENT] for r in results if not r.get(RR_DENIED)]
    assert "bm25 hit 股票代码 600519" in contents
    assert "vector hit" in contents


class _FakeVectorSource(BaseRetriever):
    def __init__(self, results):
        self._results = results

    def retrieve(self, query, top_k=5, filters=None):
        return list(self._results)


def test_role_filter_happens_before_truncation(monkeypatch):
    """issues.md 二.4：高分候选全部越权时，应继续取回可访问文档而不是误判全部越权。

    旧实现先按 top_k=2 截断（取回的全是 confidential），再按角色过滤，
    结果全是 denied 占位符。超量取回后低位的可访问文档得以保留。
    """
    monkeypatch.setattr(HybridRetriever, "_get_bm25_retriever", lambda self: None)
    from src.schemas.constants import PERMISSION_PUBLIC

    denied_pool: list[RetrievalResult] = [
        {
            RR_CONTENT: f"secret-{i}",
            RR_METADATA: {META_SOURCE: "s", META_PERMISSION_LEVEL: "confidential"},
            RR_SCORE: 0.95 - i * 0.01,
        }
        for i in range(3)
    ]
    public_pool: list[RetrievalResult] = [
        {
            RR_CONTENT: f"public-{i}",
            RR_METADATA: {META_SOURCE: "s", META_PERMISSION_LEVEL: PERMISSION_PUBLIC},
            RR_SCORE: 0.6 - i * 0.01,
        }
        for i in range(2)
    ]
    pool = denied_pool + public_pool

    class _PooledRetriever(BaseRetriever):
        def retrieve(
            self, query: str, top_k: int = 5, filters: Optional[Dict] = None
        ) -> list[RetrievalResult]:
            return list(pool[:top_k])

    retriever = HybridRetriever(user_role=ROLE_ADVISOR)
    retriever._retriever_cache[SOURCE_REPORT] = _PooledRetriever()

    results = retriever.retrieve([
        {PLAN_SOURCE: SOURCE_REPORT, PLAN_QUERY: "评级", PLAN_TOP_K: 2}
    ])

    usable = [r[RR_CONTENT] for r in results if not r.get(RR_DENIED)]
    denied = [r for r in results if r.get(RR_DENIED)]
    # 可访问文档没有被截断丢弃
    assert usable == ["public-0", "public-1"]
    # 越权结果仍保留占位符，供上层明确提示
    assert len(denied) == 3


class _FakeBM25:
    def __init__(self, results):
        self.calls: list[dict] = []
        self._results = results

    def retrieve(self, query, top_k=5, filters=None):
        self.calls.append({"query": query, "top_k": top_k, "filters": filters})
        return list(self._results)


def test_rrf_order_survives_grade_and_filter(monkeypatch):
    """issues.md 一.5：RRF 融合排序必须保留到 grade_and_filter。

    旧实现丢弃融合分数，下游按原始 score 重排，导致 BM25 原始分
    （量纲不同、数值更大）把同时命中的文档挤到后面。

    本用例验证的是重排之前的融合序，须把 BGE reranker 固定为不可用——
    重排可用时语义重排合法地改变最终顺序（见 rerank 相关用例）。
    """
    from src.agents import nodes
    from src.agents.nodes import grade_and_filter
    from src.retrieval.bm25_retriever import rrf_fuse
    from src.schemas.constants import (
        STATE_RETRIEVAL_RESULTS,
    )

    monkeypatch.setattr(
        nodes, "_try_rerank_candidates", lambda query, candidates: (candidates, "unavailable")
    )

    vec_a: RetrievalResult = {RR_CONTENT: "A", RR_METADATA: {META_SOURCE: "s", META_CHUNK_ID: "a"}, RR_SCORE: 0.9}
    vec_b: RetrievalResult = {RR_CONTENT: "B", RR_METADATA: {META_SOURCE: "s", META_CHUNK_ID: "b"}, RR_SCORE: 0.8}
    bm_b: RetrievalResult = {RR_CONTENT: "B", RR_METADATA: {META_SOURCE: "s", META_CHUNK_ID: "b"}, RR_SCORE: 12.0}
    bm_c: RetrievalResult = {RR_CONTENT: "C", RR_METADATA: {META_SOURCE: "s", META_CHUNK_ID: "c"}, RR_SCORE: 9.0}

    fused = rrf_fuse([vec_a, vec_b], [bm_b, bm_c], top_k=5)
    assert [r[RR_CONTENT] for r in fused] == ["B", "A", "C"]
    assert fused[0][RR_METADATA]["rrf_score"] > fused[1][RR_METADATA]["rrf_score"]

    updated = grade_and_filter(cast(AssistantState, {STATE_RETRIEVAL_RESULTS: list(fused)}))
    graded = [r[RR_CONTENT] for r in updated[STATE_RETRIEVAL_RESULTS]]
    # C 的 BM25 原始分 9.0 不应再把 B、A 挤到后面
    assert graded == ["B", "A", "C"]


def test_mixed_pool_does_not_demote_fused_results(monkeypatch):
    """issues.md 一.5 残留：混合池中 rrf_score 与原始 cosine 不得直接混排。

    部分来源有 BM25 命中（结果带 rrf_score，约 <=2/(k+1)）、部分没有
    （结果只剩 cosine，阈值 0.6 起）时，两种量纲进同一个排序键会让
    融合结果被系统性压底——双路命中的最优证据反而沉底。

    固定 reranker 不可用：本用例验证的是重排之前融合分与余弦分的同量纲排序。
    """
    from src.agents import nodes
    from src.agents.nodes import grade_and_filter
    from src.retrieval.bm25_retriever import rrf_fuse
    from src.schemas.constants import STATE_RETRIEVAL_RESULTS

    monkeypatch.setattr(
        nodes, "_try_rerank_candidates", lambda query, candidates: (candidates, "unavailable")
    )

    vec_a1: RetrievalResult = {RR_CONTENT: "A1", RR_METADATA: {META_SOURCE: "s-a", META_CHUNK_ID: "a1"}, RR_SCORE: 0.9}
    vec_a2: RetrievalResult = {RR_CONTENT: "A2", RR_METADATA: {META_SOURCE: "s-a", META_CHUNK_ID: "a2"}, RR_SCORE: 0.8}
    bm25_a1: RetrievalResult = {RR_CONTENT: "A1", RR_METADATA: {META_SOURCE: "s-a", META_CHUNK_ID: "a1"}, RR_SCORE: 12.0}
    # 来源 A 有 BM25 命中：结果带 rrf_score，A1 同时命中双路
    fused = rrf_fuse([vec_a1, vec_a2], [bm25_a1], top_k=5)

    # 来源 B 没有 BM25 命中：结果是纯向量结果，没有 rrf_score
    unfused: list[RetrievalResult] = [
        {RR_CONTENT: "B1", RR_METADATA: {META_SOURCE: "s-b", META_CHUNK_ID: "b1"}, RR_SCORE: 0.95},
        {RR_CONTENT: "B2", RR_METADATA: {META_SOURCE: "s-b", META_CHUNK_ID: "b2"}, RR_SCORE: 0.7},
    ]

    updated = grade_and_filter(cast(AssistantState, {STATE_RETRIEVAL_RESULTS: fused + unfused}))
    graded = [r[RR_CONTENT] for r in updated[STATE_RETRIEVAL_RESULTS]]

    # 双路命中的 A1 必须排在最前，而不是被高 cosine 的未融合结果压底
    assert graded[0] == "A1"
    # 未融合结果之间保持 cosine 序
    assert graded.index("B1") < graded.index("B2")
