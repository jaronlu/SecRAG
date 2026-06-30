"""Agent Graph 节点实现

每个节点接收 AssistantState，返回部分更新的 AssistantState。
"""

import json

from langchain_core.messages import HumanMessage

from src.agents.state import AssistantState
from src.config import config
from src.schemas.constants import (
    DEFAULT_TOP_K,
    LLM_PROVIDER_OPENAI,
    META_ERROR,
    META_SOURCE,
    PLAN_FILTERS,
    PLAN_QUERY,
    PLAN_SOURCE,
    PLAN_TOP_K,
    ROLE_ALLOWED_SOURCES,
    RR_CONTENT,
    RR_METADATA,
    RR_SCORE,
    SOURCE_FAQ,
    SOURCE_PRODUCT,
    SOURCE_REGULATION,
    SOURCE_REPORT,
)


def _build_llm():
    """根据 config 选择 LLM 后端（复用 rag/chain.py 的同名模式）"""
    if config.llm.provider == LLM_PROVIDER_OPENAI:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            base_url=config.llm.base_url,
            model=config.llm.model,
            temperature=0,
            api_key=config.llm.api_key,
        )
    from langchain_ollama import ChatOllama

    return ChatOllama(
        base_url=config.llm.base_url,
        model=config.llm.model,
        temperature=0,
    )


llm = _build_llm()


# ══════════════════════════════════════════════════════════════════════
# 4.2 Query Understand — 意图分类、实体抽取、查询重写、歧义检测
# ══════════════════════════════════════════════════════════════════════


def query_understand(state: AssistantState) -> AssistantState:
    """查询理解：意图分类、实体抽取、查询重写、歧义检测"""
    prompt = f"""请分析以下证券业务查询：

【用户查询】{state["original_query"]}
【用户角色】{state["user_role"]}
【用户部门】{state["department"]}

请以 JSON 格式返回：
{{
  "intent": "产品咨询 | 交易规则 | 法规咨询 | 研报观点 | 合规审查 | FAQ | 技术支持",
  "query_type": "product_inquiry | rule_inquiry | regulation_inquiry | report_inquiry | faq_inquiry | technical_inquiry",
  "entities": {{"product_name": "", "product_type": "", "stock_code": "", "regulation_name": "", "client_segment": ""}},
  "rewritten_query": "优化后的结构化查询",
  "ambiguity": ["是指开放式基金还是封闭式基金？"]
}}

只返回 JSON，不要其他内容。"""

    response = llm.invoke([HumanMessage(content=prompt)])
    try:
        raw = response.content
        if not isinstance(raw, str):
            raw = str(raw)
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {
            "intent": "unknown",
            "query_type": "unknown",
            "entities": {},
            "rewritten_query": state["original_query"],
            "ambiguity": [],
        }

    return {
        **state,
        "intent": result.get("intent", "unknown"),
        "query_type": result.get("query_type", "unknown"),
        "entities": result.get("entities", {}),
        "rewritten_query": result.get("rewritten_query", state["original_query"]),
        "ambiguity": result.get("ambiguity", []),
    }


# ══════════════════════════════════════════════════════════════════════
# 4.3 Planner — 根据角色权限生成多源检索计划
# ══════════════════════════════════════════════════════════════════════


def planner(state: AssistantState) -> AssistantState:
    """检索计划生成：根据意图、角色、查询类型生成多步检索计划"""
    allowed_sources = ROLE_ALLOWED_SOURCES.get(state["user_role"], [SOURCE_FAQ])

    prompt = f"""根据以下查询理解结果，生成检索计划：

【原始查询】{state["original_query"]}
【重写查询】{state["rewritten_query"]}
【意图】{state["intent"]}
【查询类型】{state["query_type"]}
【实体】{json.dumps(state["entities"], ensure_ascii=False)}
【用户角色】{state["user_role"]}

可用数据源（基于角色权限）：
- product_search: 理财产品说明书、基金合同、风险揭示书
- regulation_search: 监管法规、内部制度、处罚案例
- report_search: 研报摘要、晨会纪要、策略周报
- faq_search: 常见问题解答、操作流程

请以 JSON 数组返回检索计划，只使用当前角色允许的数据源：
[
  {{"source": "product_search", "query": "...", "top_k": 5, "filters": {{"product_type": "fund"}}}},
  {{"source": "regulation_search", "query": "...", "top_k": 3, "filters": {{"source": "csrc"}}}},
  {{"source": "report_search", "query": "...", "top_k": 5}}
]

只返回 JSON 数组。"""

    response = llm.invoke([HumanMessage(content=prompt)])
    try:
        raw = response.content
        if not isinstance(raw, str):
            raw = str(raw)
        plan = json.loads(raw)
    except json.JSONDecodeError:
        plan = [{PLAN_SOURCE: SOURCE_FAQ, PLAN_QUERY: state["rewritten_query"], PLAN_TOP_K: 3}]

    # 按角色权限过滤：去掉 LLM 可能越权生成的 source
    filtered_plan = [step for step in plan if step.get(PLAN_SOURCE) in allowed_sources]

    return {
        **state,
        "retrieval_plan": filtered_plan,
    }


# ══════════════════════════════════════════════════════════════════════
# 4.4 Retrieve — 按计划并行执行多源检索
# ══════════════════════════════════════════════════════════════════════


def retrieve(state: AssistantState) -> AssistantState:
    """并行执行检索计划：按 source 分发到对应领域检索器"""
    results = []

    for step in state["retrieval_plan"]:
        source = step.get(PLAN_SOURCE)
        try:
            if source == SOURCE_PRODUCT:
                from src.retrieval.product_retriever import ProductRetriever

                retriever = ProductRetriever()
                res = retriever.retrieve(
                    query=step.get(PLAN_QUERY, state["rewritten_query"]),
                    top_k=step.get(PLAN_TOP_K, DEFAULT_TOP_K),
                    filters=step.get(PLAN_FILTERS),
                )
                results.extend(res)
            elif source == SOURCE_REGULATION:
                from src.retrieval.regulation_retriever import RegulationRetriever

                retriever = RegulationRetriever()
                res = retriever.retrieve(
                    query=step.get(PLAN_QUERY, state["rewritten_query"]),
                    top_k=step.get(PLAN_TOP_K, DEFAULT_TOP_K),
                    filters=step.get(PLAN_FILTERS),
                )
                results.extend(res)
            elif source == SOURCE_REPORT:
                from src.retrieval.report_retriever import ReportRetriever

                retriever = ReportRetriever()
                res = retriever.retrieve(
                    query=step.get(PLAN_QUERY, state["rewritten_query"]),
                    top_k=step.get(PLAN_TOP_K, DEFAULT_TOP_K),
                    filters=step.get(PLAN_FILTERS),
                )
                results.extend(res)
            elif source == SOURCE_FAQ:
                from src.retrieval.faq_retriever import FAQRetriever

                retriever = FAQRetriever()
                res = retriever.retrieve(
                    query=step.get(PLAN_QUERY, state["rewritten_query"]),
                    top_k=step.get(PLAN_TOP_K, DEFAULT_TOP_K),
                )
                results.extend(res)
        except Exception as e:
            results.append({
                RR_CONTENT: f"检索失败: {e}",
                RR_METADATA: {META_SOURCE: source, META_ERROR: str(e)},
                RR_SCORE: 0.0,
            })

    return {
        **state,
        "retrieval_results": state["retrieval_results"] + results,
    }
