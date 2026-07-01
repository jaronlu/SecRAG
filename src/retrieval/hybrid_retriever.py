"""角色感知混合检索器：按权限过滤并执行检索计划。"""

from typing import Dict, List, Optional

from src.retrieval.base import BaseRetriever
from src.retrieval.faq_retriever import FAQRetriever
from src.retrieval.product_retriever import ProductRetriever
from src.retrieval.regulation_retriever import RegulationRetriever
from src.retrieval.report_retriever import ReportRetriever
from src.schemas.constants import (
    DEFAULT_TOP_K,
    META_ERROR,
    META_SOURCE,
    PLAN_DENIED,
    PLAN_FILTERS,
    PLAN_QUERY,
    PLAN_REASON,
    PLAN_SOURCE,
    PLAN_TOP_K,
    ROLE_ALLOWED_SOURCES,
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

_SOURCE_RETRIEVER_CLASSES: dict[str, type[BaseRetriever]] = {
    SOURCE_PRODUCT: ProductRetriever,
    SOURCE_REGULATION: RegulationRetriever,
    SOURCE_REPORT: ReportRetriever,
    SOURCE_FAQ: FAQRetriever,
}


class HybridRetriever:
    """执行 Planner 生成的检索计划，并在执行层再次做角色权限过滤。"""

    def __init__(self, user_role: str):
        self.user_role = user_role
        self.allowed_sources = ROLE_ALLOWED_SOURCES.get(user_role, [SOURCE_FAQ])
        self._retriever_cache: dict[str, BaseRetriever] = {}

    def retrieve(self, plan: List[Dict]) -> List[Dict]:
        """按角色过滤并执行一轮检索计划。

        多跳次数由 Agent Graph 条件路由控制；本类不生成计划、不循环调用 Planner。
        """
        results = []

        for step in self._filter_plan_by_role(plan):
            if step.get(PLAN_DENIED):
                results.append(self._denied_result(step))
                continue

            source = step.get(PLAN_SOURCE)
            retriever = self._get_retriever(source)
            if retriever is None:
                results.append(self._error_result(source, f"未知检索源: {source}"))
                continue

            try:
                results.extend(retriever.retrieve(
                    query=step.get(PLAN_QUERY, ""),
                    top_k=step.get(PLAN_TOP_K, DEFAULT_TOP_K),
                    filters=step.get(PLAN_FILTERS),
                ))
            except Exception as exc:
                results.append(self._error_result(source, f"检索失败: {exc}", str(exc)))

        return results

    def _filter_plan_by_role(self, plan: List[Dict]) -> List[Dict]:
        """保留允许 source；越权 source 转为显式拒绝结果。"""
        filtered = []
        for step in plan:
            source = step.get(PLAN_SOURCE)
            if source not in _SOURCE_RETRIEVER_CLASSES:
                filtered.append(step)
            elif source in self.allowed_sources:
                filtered.append(step)
            else:
                filtered.append({
                    PLAN_SOURCE: source,
                    PLAN_QUERY: step.get(PLAN_QUERY, ""),
                    PLAN_TOP_K: step.get(PLAN_TOP_K, 0),
                    PLAN_DENIED: True,
                    PLAN_REASON: f"角色 {self.user_role} 无权限访问 {source}",
                })
        return filtered

    def _get_retriever(self, source: Optional[str]) -> Optional[BaseRetriever]:
        if source is None:
            return None
        if source not in self._retriever_cache:
            cls = _SOURCE_RETRIEVER_CLASSES.get(source)
            if cls is not None:
                self._retriever_cache[source] = cls()
        return self._retriever_cache.get(source)

    def _denied_result(self, step: Dict) -> Dict:
        source = step.get(PLAN_SOURCE)
        return {
            RR_CONTENT: "",
            RR_METADATA: {META_SOURCE: source, "permission_denied": True},
            RR_SCORE: 0.0,
            RR_DENIED: True,
            RR_REASON: step.get(PLAN_REASON, "权限不足"),
        }

    def _error_result(self, source: Optional[str], content: str, error: Optional[str] = None) -> Dict:
        return {
            RR_CONTENT: content,
            RR_METADATA: {META_SOURCE: source, META_ERROR: error or content},
            RR_SCORE: 0.0,
        }
