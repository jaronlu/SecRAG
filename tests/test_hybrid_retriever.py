"""HybridRetriever 单元测试（mock 领域检索器，避免真实 ChromaDB）。"""

from __future__ import annotations

from typing import Dict, List, Optional

from src.retrieval.hybrid_retriever import HybridRetriever
from src.schemas.constants import (
    META_ERROR,
    META_SOURCE,
    PLAN_FILTERS,
    PLAN_QUERY,
    PLAN_SOURCE,
    PLAN_TOP_K,
    ROLE_ADVISOR,
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


class FakeRetriever:
    def __init__(self):
        self.calls: list[dict] = []

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict] = None,
    ) -> List[Dict]:
        self.calls.append({"query": query, "top_k": top_k, "filters": filters})
        return [{
            RR_CONTENT: f"{query}:{top_k}",
            RR_METADATA: {META_SOURCE: "fake"},
            RR_SCORE: 0.9,
        }]


class FailingRetriever:
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict] = None,
    ) -> List[Dict]:
        raise RuntimeError("boom")


class TestHybridRetriever:
    def test_executes_allowed_plan_and_passes_arguments(self):
        fake = FakeRetriever()
        retriever = HybridRetriever(user_role=ROLE_ADVISOR)
        retriever._retriever_cache[SOURCE_PRODUCT] = fake

        results = retriever.retrieve([{
            PLAN_SOURCE: SOURCE_PRODUCT,
            PLAN_QUERY: "风险等级",
            PLAN_TOP_K: 3,
            PLAN_FILTERS: {"product_type": "fund"},
        }])

        assert results[0][RR_CONTENT] == "风险等级:3"
        assert fake.calls == [{
            "query": "风险等级",
            "top_k": 3,
            "filters": {"product_type": "fund"},
        }]

    def test_denies_source_not_allowed_for_role(self):
        retriever = HybridRetriever(user_role=ROLE_TECHNICAL)

        results = retriever.retrieve([{
            PLAN_SOURCE: SOURCE_REPORT,
            PLAN_QUERY: "内部研报摘要",
            PLAN_TOP_K: 5,
        }])

        assert len(results) == 1
        assert results[0][RR_DENIED] is True
        assert "无权限访问" in results[0][RR_REASON]
        assert results[0][RR_METADATA][META_SOURCE] == SOURCE_REPORT

    def test_unknown_role_falls_back_to_faq(self):
        fake = FakeRetriever()
        retriever = HybridRetriever(user_role="unknown")
        retriever._retriever_cache[SOURCE_FAQ] = fake

        results = retriever.retrieve([{
            PLAN_SOURCE: SOURCE_FAQ,
            PLAN_QUERY: "操作流程",
        }])

        assert results[0][RR_CONTENT] == "操作流程:5"
        assert fake.calls[0]["top_k"] == 5

    def test_unknown_source_returns_error_result(self):
        retriever = HybridRetriever(user_role=ROLE_ADVISOR)

        results = retriever.retrieve([{
            PLAN_SOURCE: "unknown_search",
            PLAN_QUERY: "测试",
        }])

        assert results[0][RR_SCORE] == 0.0
        assert "未知检索源" in results[0][RR_CONTENT]
        assert results[0][RR_METADATA][META_ERROR]

    def test_retriever_exception_returns_error_result(self):
        retriever = HybridRetriever(user_role=ROLE_ADVISOR)
        retriever._retriever_cache[SOURCE_PRODUCT] = FailingRetriever()

        results = retriever.retrieve([{
            PLAN_SOURCE: SOURCE_PRODUCT,
            PLAN_QUERY: "风险等级",
        }])

        assert results[0][RR_SCORE] == 0.0
        assert "检索失败" in results[0][RR_CONTENT]
        assert results[0][RR_METADATA][META_ERROR] == "boom"

    def test_merges_multiple_allowed_sources(self):
        product = FakeRetriever()
        faq = FakeRetriever()
        retriever = HybridRetriever(user_role=ROLE_ADVISOR)
        retriever._retriever_cache[SOURCE_PRODUCT] = product
        retriever._retriever_cache[SOURCE_FAQ] = faq

        results = retriever.retrieve([
            {PLAN_SOURCE: SOURCE_PRODUCT, PLAN_QUERY: "产品", PLAN_TOP_K: 2},
            {PLAN_SOURCE: SOURCE_FAQ, PLAN_QUERY: "FAQ", PLAN_TOP_K: 1},
        ])

        assert [r[RR_CONTENT] for r in results] == ["产品:2", "FAQ:1"]

    def test_mixed_allowed_and_denied_sources_preserves_both_results(self):
        fake = FakeRetriever()
        retriever = HybridRetriever(user_role=ROLE_TECHNICAL)
        retriever._retriever_cache[SOURCE_FAQ] = fake

        results = retriever.retrieve([
            {PLAN_SOURCE: SOURCE_FAQ, PLAN_QUERY: "FAQ"},
            {PLAN_SOURCE: SOURCE_REGULATION, PLAN_QUERY: "法规"},
        ])

        assert results[0][RR_CONTENT] == "FAQ:5"
        assert results[1][RR_DENIED] is True
