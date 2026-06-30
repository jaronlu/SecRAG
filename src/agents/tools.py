import json
from typing import Optional

from langchain_core.tools import tool

from src.schemas.constants import DEFAULT_TOP_K, META_PRODUCT_TYPE

# ══════════════════════════════════════════════════════════════════════
# 4 个知识检索工具（封装领域检索器，doc_type 过滤由检索器内部处理）
# ══════════════════════════════════════════════════════════════════════


@tool
def product_search(
    query: str, top_k: int = DEFAULT_TOP_K, product_type: Optional[str] = None
) -> str:
    """产品知识库检索：搜索理财产品说明书、基金合同、风险揭示书。
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
    """法规库检索：搜索监管法规、内部制度、处罚案例。
    参数:
      query: 检索关键词
      top_k: 返回条数，默认 5
      source: 可选过滤，如 csrc / exchange / internal
    返回: 检索结果列表
    """
    from src.retrieval.regulation_retriever import RegulationRetriever

    retriever = RegulationRetriever()
    filters = {"source": source} if source else None
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
# 占位工具（Week 2 占位，Week 3 由 impl-05 升级版替换）
# ══════════════════════════════════════════════════════════════════════


@tool
def calculator(expression: str) -> str:
    """精确计算：用于费用计算、收益率计算等。
    Week 2 占位；Week 3 由 src/tools/calculator.py（Decimal + AST）替换。
    示例: "申购费 100万 * 1.5%"
    """
    # TODO: Phase 2 — 接入 Decimal + AST 安全计算器
    return f"calculator stub: {expression}"


@tool
def suitability_check(client_id: str, product_id: str) -> str:
    """适当性匹配检查：检查客户是否适合购买指定产品。
    仅合规/投顾场景可用。
    """
    # TODO: Phase 2 — 接入 src/utils/suitability.py
    return f"suitability_check stub: client={client_id}, product={product_id}"


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
]
