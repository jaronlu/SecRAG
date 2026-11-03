import json
from typing import Optional, cast

from langchain_core.tools import BaseTool, tool
from langchain.tools import ToolRuntime

from src.agents.state import AssistantState
from src.product_scope import (
    ProductProfile,
    allowed_sources_for_profile,
    allowed_tools_for_profile,
    get_product_profile,
)
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
from src.tools.market_data import market_data_available
from src.tools.rerank import reranker_available
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
    getattr(product_search, "name", "product_search"): SOURCE_PRODUCT,
    getattr(regulation_search, "name", "regulation_search"): SOURCE_REGULATION,
    getattr(report_search, "name", "report_search"): SOURCE_REPORT,
    getattr(faq_search, "name", "faq_search"): SOURCE_FAQ,
    getattr(sql_query_tool, "name", "sql_query_tool"): SOURCE_SQL,
}

# 非检索工具共享白名单（P0-3 反转默认放行）：必须显式列入才可见。
# 未在权限映射/白名单声明的新工具默认对所有角色不可见，避免漏配时
# 可见性层与 authorize_reason_tool_call 授权层一致放行。
_NON_RETRIEVAL_TOOL_WHITELIST: set[str] = {
    calculator.name,
    suitability_check.name,
    market_data_tool.name,
    financial_ratios_tool.name,
    rerank_tool.name,
}


def get_tools_for_role(
    user_role: str,
    excluded_retrieval_sources: set[str] | None = None,
    *,
    profile: ProductProfile | None = None,
) -> list[BaseTool]:
    """Return tools visible to the ReAct agent for the given role and profile.

    profile 未显式传入时按配置解析（默认 knowledge_qa）。先执行既有
    ROLE_ALLOWED_SOURCES 角色过滤，再与 profile 的源/工具白名单求交集，
    保证默认产品路径不把实验工具暴露给模型；授权层复用同一过滤结果。
    """
    effective_profile = get_product_profile(profile)
    allowed_sources = set(ROLE_ALLOWED_SOURCES.get(user_role, []))
    excluded_sources = excluded_retrieval_sources or set()
    # ISSUE-10/ISSUE-19：数据源未配置的工具摘除出白名单——恒失败工具反复
    # 进入工具清单只会浪费 LLM 轮次；reranker 生效（安装 FlagEmbedding +
    # 模型本地化）或行情数据就绪（本地表入库 / 安装 baostock）后自动恢复
    non_retrieval_whitelist: set[str] = set(_NON_RETRIEVAL_TOOL_WHITELIST)
    if not reranker_available():
        non_retrieval_whitelist.discard(rerank_tool.name)
    if not market_data_available():
        non_retrieval_whitelist.discard(market_data_tool.name)

    profile_sources = allowed_sources_for_profile(effective_profile)
    profile_tools = allowed_tools_for_profile(effective_profile)

    def _visible(tool_item: BaseTool) -> bool:
        name = tool_item.name
        source = _RETRIEVAL_TOOL_SOURCES.get(name)
        role_visible = (
            source in allowed_sources and source not in excluded_sources
        ) or name in non_retrieval_whitelist
        if not role_visible:
            return False
        # profile 白名单：knowledge_qa 只保留产品文档/FAQ 两个知识检索工具；
        # full profile 两项均为 None，完全沿用既有角色过滤结果
        if profile_sources is not None and source not in profile_sources:
            return False
        if profile_tools is not None and name not in profile_tools:
            return False
        return True

    return [tool_item for tool_item in tools if _visible(tool_item)]
