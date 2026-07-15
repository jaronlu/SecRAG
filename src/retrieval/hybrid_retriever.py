"""角色感知混合检索器：按权限过滤并执行检索计划。"""

from typing import Callable, Optional

from src.retrieval.base import BaseRetriever
from src.retrieval.faq_retriever import FAQRetriever
from src.retrieval.product_retriever import ProductRetriever
from src.retrieval.regulation_retriever import RegulationRetriever
from src.retrieval.report_retriever import ReportRetriever
from src.retrieval.vector_retriever import ChromaVectorRetriever
from src.schemas.constants import (
    DEFAULT_TOP_K,
    META_ALLOWED_ROLES,
    META_ERROR,
    META_PERMISSION_LEVEL,
    META_SOURCE,
    PERMISSION_PUBLIC,
    PLAN_DENIED,
    PLAN_FILTERS,
    PLAN_QUERY,
    PLAN_REASON,
    PLAN_SOURCE,
    PLAN_TOP_K,
    ROLE_ALLOWED_SOURCES,
    RR_METADATA,
    SOURCE_FAQ,
    SOURCE_PRODUCT,
    SOURCE_REGULATION,
    SOURCE_REPORT,
)
from src.schemas.typed_dicts import RetrievalPlanStep, RetrievalResult

# 工厂映射：把 source 名称映射到对应的 Retriever 类
# 当 Planner 说"去查 product"，这里就实例化 ProductRetriever
_SOURCE_RETRIEVER_FACTORIES: dict[str, Callable[[BaseRetriever], BaseRetriever]] = {
    SOURCE_PRODUCT: ProductRetriever,
    SOURCE_REGULATION: RegulationRetriever,
    SOURCE_REPORT: ReportRetriever,
    SOURCE_FAQ: FAQRetriever,
}


#  这是权限感知的混合检索器，接收 Planner 生成的检索计划，按用户角色过滤可访问的数据源，执行检索后再按元数据权限过滤结果，最终返回带权限控制的结果列表。
class HybridRetriever:
    """执行 Planner 生成的检索计划，并在执行层再次做角色权限过滤。"""

    def __init__(self, user_role: str, data_permissions: list[str] | None = None):
        self.user_role = user_role
        self.allowed_sources = ROLE_ALLOWED_SOURCES.get(user_role, [])
        self.data_permissions = set(data_permissions or [PERMISSION_PUBLIC])
        self._retriever_cache: dict[str, BaseRetriever] = {}
        self._vector_engine: Optional[ChromaVectorRetriever] = None

    def retrieve(self, plan: list[RetrievalPlanStep]) -> list[RetrievalResult]:
        """按角色过滤并执行一轮检索计划。

        多跳次数由 Agent Graph 条件路由控制；本类不生成计划、不循环调用 Planner。
        """
        results: list[RetrievalResult] = []

        for step in self._filter_plan_by_role(plan):
            if step.get(PLAN_DENIED):
                results.append(self._denied_result(step))
                continue

            source = step.get(PLAN_SOURCE)
            retriever = self._get_retriever(source)
            if retriever is None:
                results.append(self._error_result(source, "未知检索源", f"未知检索源: {source}"))
                continue

            try:
                retrieved = retriever.retrieve(
                    query=step.get(PLAN_QUERY, ""),
                    top_k=step.get(PLAN_TOP_K, DEFAULT_TOP_K),
                    filters=step.get(PLAN_FILTERS),
                )
                results.extend(self._filter_results_by_role(retrieved))
            except Exception as exc:
                results.append(self._error_result(source, "检索失败", str(exc)))

        return results

    def _filter_plan_by_role(self, plan: list[RetrievalPlanStep]) -> list[RetrievalPlanStep]:
        """保留允许 source；越权 source 转为显式拒绝结果。"""
        filtered: list[RetrievalPlanStep] = []
        for step in plan:
            source = step.get(PLAN_SOURCE)
            if source not in _SOURCE_RETRIEVER_FACTORIES:
                filtered.append(step)  # 未知 source，不过滤（可能是自定义 source）
            elif source in self.allowed_sources:
                filtered.append(step)  # 用户有权访问，正常执行
            else:
                filtered.append(
                    RetrievalPlanStep(
                        source=source or "",
                        query=step.get(PLAN_QUERY, ""),
                        top_k=step.get(PLAN_TOP_K, 0),
                        denied=True,  # 用户无权访问，转为显式拒绝，如果直接跳过（不 append），上层调用方就不知道哪些 source 被跳过了。
                        reason=f"角色 {self.user_role} 无权限访问 {source}",
                    )
                )
        return filtered

    # 获取检索器实例，带缓存
    def _get_retriever(self, source: Optional[str]) -> Optional[BaseRetriever]:
        if source is None:
            return None
        if source not in self._retriever_cache:
            # 从工厂取类，传入共享的向量引擎实例
            factory = _SOURCE_RETRIEVER_FACTORIES.get(source)
            if factory is not None:
                self._retriever_cache[source] = factory(self._get_vector_engine())
        return self._retriever_cache.get(source)

    # 懒加载共享向量引擎：所有 domain retriever 共用同一个 ChromaVectorRetriever
    def _get_vector_engine(self) -> ChromaVectorRetriever:
        if self._vector_engine is None:
            self._vector_engine = ChromaVectorRetriever()
        return self._vector_engine

    def _filter_results_by_role(self, results: list[RetrievalResult]) -> list[RetrievalResult]:
        """按结果级 metadata 过滤：先看 permission_level，再看 allowed_roles。"""
        filtered: list[RetrievalResult] = []
        for result in results:
            metadata = result.get(RR_METADATA, {})
            # 先看 permission_level：是否在用户允许的权限范围内
            permission_level = metadata.get(META_PERMISSION_LEVEL, PERMISSION_PUBLIC)
            if permission_level not in self.data_permissions:
                # 用户无权访问该权限级别的数据
                filtered.append(
                    self._permission_denied_result(
                        metadata.get(META_SOURCE),
                        f"角色 {self.user_role} 无权访问 {permission_level} 数据",
                    )
                )
                continue

            # 再看 allowed_roles：具体允许哪些角色访问
            allowed_roles = metadata.get(META_ALLOWED_ROLES)
            if not allowed_roles:
                # 没有 allowed_roles 字段
                if permission_level == PERMISSION_PUBLIC:
                    # 公开数据，直接放行
                    filtered.append(result)
                else:
                    # 非公开数据但没写 allowed_roles，默认拒绝（安全优先）
                    filtered.append(
                        self._permission_denied_result(
                            metadata.get(META_SOURCE),
                            "非公开数据缺少 allowed_roles，已默认拒绝",
                        )
                    )
                continue

            # 解析 allowed_roles：可能是字符串 "admin,user" 或列表 ["admin", "user"]
            if isinstance(allowed_roles, str):
                allowed = {role.strip() for role in allowed_roles.split(",") if role.strip()}
            else:
                allowed = set(allowed_roles)

            if self.user_role in allowed:
                filtered.append(result)
            else:
                filtered.append(
                    self._permission_denied_result(
                        metadata.get(META_SOURCE),
                        f"角色 {self.user_role} 不在 allowed_roles 中",
                    )
                )
        return filtered

    # 生成权限拒绝结果（结果级）
    def _permission_denied_result(self, source: object, reason: str) -> RetrievalResult:
        return RetrievalResult(
            content="",
            metadata={META_SOURCE: source, "permission_denied": True},
            score=0.0,
            denied=True,
            reason=reason,
        )

    # 生成权限拒绝结果（计划级)
    def _denied_result(self, step: RetrievalPlanStep) -> RetrievalResult:
        source = step.get(PLAN_SOURCE)
        return RetrievalResult(
            content="",
            metadata={META_SOURCE: source, "permission_denied": True},
            score=0.0,
            denied=True,
            reason=step.get(PLAN_REASON, "权限不足"),
        )

    # 生成错误结果（检索失败、未知 source 等）
    def _error_result(
        self, source: Optional[str], content: str, error: Optional[str] = None
    ) -> RetrievalResult:
        return RetrievalResult(
            content="",
            metadata={META_SOURCE: source, META_ERROR: error or content},
            reason=content,
            score=0.0,
        )
