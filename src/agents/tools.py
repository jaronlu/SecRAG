import json
from typing import Optional, cast

from langchain_core.tools import tool
from langchain.tools import ToolRuntime

from src.agents.state import AssistantState
from src.retrieval.hybrid_retriever import HybridRetriever
from src.schemas.constants import (
    DEFAULT_TOP_K,
    META_PRODUCT_TYPE,
    META_SOURCE,
    ROLE_ALLOWED_SOURCES,
    SOURCE_FAQ,
    SOURCE_PRODUCT,
    SOURCE_REGULATION,
    SOURCE_REPORT,
    SOURCE_SQL,
    STATE_DATA_PERMISSIONS,
    STATE_USER_ROLE,
)
from src.schemas.typed_dicts import RetrievalPlanStep
from src.tools import (
    calculator,
    financial_ratios_tool,
    market_data_tool,
    rerank_tool,
    sql_query_tool,
    suitability_check,
)

# ══════════════════════════════════════════════════════════════════════
# 4 个知识检索工具（统一经 HybridRetriever 执行 source + chunk 权限过滤）
# ══════════════════════════════════════════════════════════════════════


def _role_aware_search(
    *,
    query: str,
    top_k: int,
    source: str,
    filters: dict[str, str] | None,
    runtime: ToolRuntime,
) -> str:
    state = cast(AssistantState, runtime.state)
    retriever = HybridRetriever(
        user_role=state.get(STATE_USER_ROLE, ""),
        data_permissions=state.get(STATE_DATA_PERMISSIONS, []),
    )
    plan = RetrievalPlanStep(
        source=source,
        query=query,
        top_k=top_k,
        filters=filters,
    )
    results = retriever.retrieve([plan])
    return json.dumps(results, ensure_ascii=False)


@tool
def product_search(
    query: str,
    runtime: ToolRuntime,
    top_k: int = DEFAULT_TOP_K,
    product_type: Optional[str] = None,
) -> str:
    """产品知识库检索：搜索理财产品说明书、产品合同、风险揭示书。
    参数:
      query: 检索关键词
      top_k: 返回条数，默认 5
      product_type: 可选过滤，如 fund / insurance / structured_product
    返回: 检索结果列表
    """
    filters = {META_PRODUCT_TYPE: product_type} if product_type else None
    return _role_aware_search(
        query=query,
        top_k=top_k,
        source=SOURCE_PRODUCT,
        filters=filters,
        runtime=runtime,
    )


@tool
def regulation_search(
    query: str,
    runtime: ToolRuntime,
    top_k: int = DEFAULT_TOP_K,
    source: Optional[str] = None,
) -> str:
    """法规库检索：搜索规则法规、内部制度、处罚案例。
    参数:
      query: 检索关键词
      top_k: 返回条数，默认 5
      source: 可选过滤，如 csrc / exchange / internal
    返回: 检索结果列表
    """
    filters = {META_SOURCE: source} if source else None
    return _role_aware_search(
        query=query,
        top_k=top_k,
        source=SOURCE_REGULATION,
        filters=filters,
        runtime=runtime,
    )


@tool
def report_search(
    query: str,
    runtime: ToolRuntime,
    top_k: int = DEFAULT_TOP_K,
    report_type: Optional[str] = None,
) -> str:
    """研报摘要检索：搜索内部研报摘要、晨会纪要、策略周报。
    参数:
      query: 检索关键词
      top_k: 返回条数，默认 5
      report_type: 可选过滤，如 equity / fixed_income / macro
    返回: 检索结果列表
    """
    filters = {"report_type": report_type} if report_type else None
    return _role_aware_search(
        query=query,
        top_k=top_k,
        source=SOURCE_REPORT,
        filters=filters,
        runtime=runtime,
    )


@tool
def faq_search(query: str, runtime: ToolRuntime, top_k: int = DEFAULT_TOP_K) -> str:
    """FAQ 检索：搜索常见问题解答、操作流程。
    适用于运营/支持场景。
    """
    return _role_aware_search(
        query=query,
        top_k=top_k,
        source=SOURCE_FAQ,
        filters=None,
        runtime=runtime,
    )


# ══════════════════════════════════════════════════════════════════════
# 工具注册表（reason 节点内 ReAct Agent 使用）
# ══════════════════════════════════════════════════════════════════════

tools = [
    product_search,
    regulation_search,
    report_search,
    faq_search,
    calculator,
    suitability_check,
    market_data_tool,
    sql_query_tool,
    financial_ratios_tool,
    rerank_tool,
]

_RETRIEVAL_TOOL_SOURCES = {
    product_search.name: SOURCE_PRODUCT,
    regulation_search.name: SOURCE_REGULATION,
    report_search.name: SOURCE_REPORT,
    faq_search.name: SOURCE_FAQ,
    sql_query_tool.name: SOURCE_SQL,
}


def get_tools_for_role(
    user_role: str,
    excluded_retrieval_sources: set[str] | None = None,
):
    """Return tools visible to the ReAct agent for the given role."""
    allowed_sources = set(ROLE_ALLOWED_SOURCES.get(user_role, []))
    excluded_sources = excluded_retrieval_sources or set()
    return [
        tool_item
        for tool_item in tools
        if (
            _RETRIEVAL_TOOL_SOURCES.get(tool_item.name) in allowed_sources
            and _RETRIEVAL_TOOL_SOURCES.get(tool_item.name) not in excluded_sources
        )
        or tool_item.name not in _RETRIEVAL_TOOL_SOURCES
    ]
