import json
from typing import Optional

from langchain_core.tools import tool

from src.schemas.constants import (
    DEFAULT_TOP_K,
    META_PRODUCT_TYPE,
    META_SOURCE,
    ROLE_ALLOWED_SOURCES,
    SOURCE_FAQ,
    SOURCE_PRODUCT,
    SOURCE_REGULATION,
    SOURCE_REPORT,
)
from src.tools import (
    calculator,
    financial_ratios_tool,
    market_data_tool,
    rerank_tool,
    sql_query_tool,
    suitability_check,
)

# ══════════════════════════════════════════════════════════════════════
# 4 个知识检索工具（封装领域检索器，doc_type 过滤由检索器内部处理）
# ══════════════════════════════════════════════════════════════════════


@tool
def product_search(
    query: str, top_k: int = DEFAULT_TOP_K, product_type: Optional[str] = None
) -> str:
    """产品知识库检索：搜索理财产品说明书、产品合同、风险揭示书。
    参数:
      query: 检索关键词
      top_k: 返回条数，默认 5
      product_type: 可选过滤，如 fund / insurance / structured_product
    返回: 检索结果列表
    """
    from src.retrieval.product_retriever import ProductRetriever

    retriever = ProductRetriever()
    filters = {META_PRODUCT_TYPE: product_type} if product_type else None
    results = retriever.retrieve(query=query, top_k=top_k, filters=filters)
    return json.dumps(results, ensure_ascii=False)


@tool
def regulation_search(query: str, top_k: int = DEFAULT_TOP_K, source: Optional[str] = None) -> str:
    """法规库检索：搜索规则法规、内部制度、处罚案例。
    参数:
      query: 检索关键词
      top_k: 返回条数，默认 5
      source: 可选过滤，如 csrc / exchange / internal
    返回: 检索结果列表
    """
    from src.retrieval.regulation_retriever import RegulationRetriever

    retriever = RegulationRetriever()
    filters = {META_SOURCE: source} if source else None
    results = retriever.retrieve(query=query, top_k=top_k, filters=filters)
    return json.dumps(results, ensure_ascii=False)


@tool
def report_search(query: str, top_k: int = DEFAULT_TOP_K, report_type: Optional[str] = None) -> str:
    """研报摘要检索：搜索内部研报摘要、晨会纪要、策略周报。
    参数:
      query: 检索关键词
      top_k: 返回条数，默认 5
      report_type: 可选过滤，如 equity / fixed_income / macro
    返回: 检索结果列表
    """
    from src.retrieval.report_retriever import ReportRetriever

    retriever = ReportRetriever()
    filters = {"report_type": report_type} if report_type else None
    results = retriever.retrieve(query=query, top_k=top_k, filters=filters)
    return json.dumps(results, ensure_ascii=False)


@tool
def faq_search(query: str, top_k: int = DEFAULT_TOP_K) -> str:
    """FAQ 检索：搜索常见问题解答、操作流程。
    适用于运营/支持场景。
    """
    from src.retrieval.faq_retriever import FAQRetriever

    retriever = FAQRetriever()
    results = retriever.retrieve(query=query, top_k=top_k)
    return json.dumps(results, ensure_ascii=False)


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
}


def get_tools_for_role(user_role: str):
    """Return tools visible to the ReAct agent for the given role."""
    allowed_sources = set(ROLE_ALLOWED_SOURCES.get(user_role, [SOURCE_FAQ]))
    return [
        tool_item
        for tool_item in tools
        if _RETRIEVAL_TOOL_SOURCES.get(tool_item.name) in allowed_sources
        or tool_item.name not in _RETRIEVAL_TOOL_SOURCES
    ]
