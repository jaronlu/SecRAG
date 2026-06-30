"""Agent Graph 节点实现

每个节点接收 AssistantState，返回部分更新的 AssistantState。
"""

import json

from langchain_core.messages import HumanMessage

from src.agents.state import AssistantState
from src.config import config
from src.schemas.constants import (
    DEFAULT_TOP_K,
    GRADE_TOP_K,
    LLM_PROVIDER_OPENAI,
    META_ERROR,
    META_SOURCE,
    META_TITLE,
    PLAN_FILTERS,
    PLAN_QUERY,
    PLAN_SOURCE,
    PLAN_TOP_K,
    ROLE_ADVISOR,
    ROLE_ALLOWED_SOURCES,
    ROLE_COMPLIANCE,
    ROLE_INSTITUTIONAL_SALES,
    ROLE_OPERATIONS,
    ROLE_TECHNICAL,
    RR_CONTENT,
    RR_METADATA,
    RR_SCORE,
    SOURCE_FAQ,
    SOURCE_PRODUCT,
    SOURCE_REGULATION,
    SOURCE_REPORT,
    STATE_AMBIGUITY,
    STATE_DEPARTMENT,
    STATE_ENTITIES,
    STATE_FINAL_ANSWER,
    STATE_INTENT,
    STATE_MESSAGES,
    STATE_ORIGINAL_QUERY,
    STATE_QUERY_TYPE,
    STATE_RETRIEVAL_PLAN,
    STATE_RETRIEVAL_RESULTS,
    STATE_REWRITTEN_QUERY,
    STATE_USER_ROLE,
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

【用户查询】{state[STATE_ORIGINAL_QUERY]}
【用户角色】{state[STATE_USER_ROLE]}
【用户部门】{state[STATE_DEPARTMENT]}

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
            "rewritten_query": state[STATE_ORIGINAL_QUERY],
            "ambiguity": [],
        }

    return {
        **state,
        STATE_INTENT: result.get("intent", "unknown"),
        STATE_QUERY_TYPE: result.get("query_type", "unknown"),
        STATE_ENTITIES: result.get("entities", {}),
        STATE_REWRITTEN_QUERY: result.get("rewritten_query", state[STATE_ORIGINAL_QUERY]),
        STATE_AMBIGUITY: result.get("ambiguity", []),
    }


# ══════════════════════════════════════════════════════════════════════
# 4.3 Planner — 根据角色权限生成多源检索计划
# ══════════════════════════════════════════════════════════════════════


def planner(state: AssistantState) -> AssistantState:
    """检索计划生成：根据意图、角色、查询类型生成多步检索计划"""
    allowed_sources = ROLE_ALLOWED_SOURCES.get(state[STATE_USER_ROLE], [SOURCE_FAQ])

    prompt = f"""根据以下查询理解结果，生成检索计划：

【原始查询】{state[STATE_ORIGINAL_QUERY]}
【重写查询】{state[STATE_REWRITTEN_QUERY]}
【意图】{state[STATE_INTENT]}
【查询类型】{state[STATE_QUERY_TYPE]}
【实体】{json.dumps(state[STATE_ENTITIES], ensure_ascii=False)}
【用户角色】{state[STATE_USER_ROLE]}

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
        plan = [{PLAN_SOURCE: SOURCE_FAQ, PLAN_QUERY: state[STATE_REWRITTEN_QUERY], PLAN_TOP_K: 3}]

    # 按角色权限过滤：去掉 LLM 可能越权生成的 source
    filtered_plan = [step for step in plan if step.get(PLAN_SOURCE) in allowed_sources]

    return {
        **state,
        STATE_RETRIEVAL_PLAN: filtered_plan,
    }


# ══════════════════════════════════════════════════════════════════════
# 4.4 Retrieve — 按计划并行执行多源检索
# ══════════════════════════════════════════════════════════════════════


def retrieve(state: AssistantState) -> AssistantState:
    """并行执行检索计划：按 source 分发到对应领域检索器"""
    results = []

    for step in state[STATE_RETRIEVAL_PLAN]:
        source = step.get(PLAN_SOURCE)
        try:
            if source == SOURCE_PRODUCT:
                from src.retrieval.product_retriever import ProductRetriever

                retriever = ProductRetriever()
                res = retriever.retrieve(
                    query=step.get(PLAN_QUERY, state[STATE_REWRITTEN_QUERY]),
                    top_k=step.get(PLAN_TOP_K, DEFAULT_TOP_K),
                    filters=step.get(PLAN_FILTERS),
                )
                results.extend(res)
            elif source == SOURCE_REGULATION:
                from src.retrieval.regulation_retriever import RegulationRetriever

                retriever = RegulationRetriever()
                res = retriever.retrieve(
                    query=step.get(PLAN_QUERY, state[STATE_REWRITTEN_QUERY]),
                    top_k=step.get(PLAN_TOP_K, DEFAULT_TOP_K),
                    filters=step.get(PLAN_FILTERS),
                )
                results.extend(res)
            elif source == SOURCE_REPORT:
                from src.retrieval.report_retriever import ReportRetriever

                retriever = ReportRetriever()
                res = retriever.retrieve(
                    query=step.get(PLAN_QUERY, state[STATE_REWRITTEN_QUERY]),
                    top_k=step.get(PLAN_TOP_K, DEFAULT_TOP_K),
                    filters=step.get(PLAN_FILTERS),
                )
                results.extend(res)
            elif source == SOURCE_FAQ:
                from src.retrieval.faq_retriever import FAQRetriever

                retriever = FAQRetriever()
                res = retriever.retrieve(
                    query=step.get(PLAN_QUERY, state[STATE_REWRITTEN_QUERY]),
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
        STATE_RETRIEVAL_RESULTS: state[STATE_RETRIEVAL_RESULTS] + results,
    }


# ══════════════════════════════════════════════════════════════════════
# 4.5 Grade and Filter — 按相似度排序，保留 top-10
# ══════════════════════════════════════════════════════════════════════


def grade_and_filter(state: AssistantState) -> AssistantState:
    """相关性评分与过滤：按 score 排序，保留前 GRADE_TOP_K 条"""
    results = state.get(STATE_RETRIEVAL_RESULTS, [])
    if not results:
        return state

    filtered = sorted(results, key=lambda x: x.get(RR_SCORE, 0), reverse=True)[:GRADE_TOP_K]

    return {
        **state,
        STATE_RETRIEVAL_RESULTS: filtered,
    }


# ══════════════════════════════════════════════════════════════════════
# 4.6 Reason — LLM 推理 + 工具调用（ReAct Agent）
# ══════════════════════════════════════════════════════════════════════


def reason(state: AssistantState) -> AssistantState:
    """LLM 推理 + 工具调用

    技术债：当前使用 langchain.agents.create_agent 嵌套在 LangGraph 节点内。
    已知问题：
      - 每次调用重新编译 agent graph 和绑定工具（性能浪费）
      - 嵌套 Graph 的状态管理与外层 checkpoint 隔离
    Phase 2 重构：将 ReAct 循环建模为 Graph 子图
      reason → [tool_calls?] → tool_executor → reason
    全程在 StateGraph 范式内，状态与 checkpoint 完整保留。
    """
    from langchain.agents import create_agent
    from src.agents.tools import tools

    results = state.get(STATE_RETRIEVAL_RESULTS, [])

    # 构建 context
    context = "\n\n".join([
        f"[来源{i+1}] {r[RR_METADATA].get(META_TITLE, '未知')}\n{r[RR_CONTENT]}"
        for i, r in enumerate(results[:5])
    ])

    # 角色化 prompt
    role_instructions = {
        ROLE_ADVISOR: "你是券商投顾助手。基于内部知识库为客户提供准确的产品/规则/市场信息。",
        ROLE_INSTITUTIONAL_SALES: "你是机构销售助手。为机构客户提供研究支持和市场洞察。",
        ROLE_COMPLIANCE: "你是合规助手。基于法规和制度提供准确的合规判断和引用。",
        ROLE_OPERATIONS: "你是运营支持助手。回答客户常见问题，提供操作指引。",
        ROLE_TECHNICAL: "你是技术支持助手。回答系统/数据相关问题。",
    }

    role = state.get(STATE_USER_ROLE, ROLE_OPERATIONS)
    role_instruction = role_instructions.get(role, "你是券商内部知识助手。")

    system_prompt = f"""{role_instruction}

【检索结果】
{context}

【用户问题】{state[STATE_ORIGINAL_QUERY]}

【用户角色】{role}

如果需要计算或查询更多信息，可以使用工具。
回答必须附引用标注 [来源1][来源2]。
数字必须来自检索结果，禁止编造。
{"投顾/销售角色：不得输出买入/卖出/目标价等投资建议，仅提供事实信息。" if role in (ROLE_ADVISOR, ROLE_INSTITUTIONAL_SALES) else ""}
{"合规角色：引用必须精确到条款/条文号。" if role == ROLE_COMPLIANCE else ""}"""

    agent = create_agent(model=llm, tools=tools, system_prompt=system_prompt)
    response = agent.invoke({"messages": [HumanMessage(content=state[STATE_ORIGINAL_QUERY])]})

    final_msg = response[STATE_MESSAGES][-1]
    return {
        **state,
        STATE_MESSAGES: state.get(STATE_MESSAGES, []) + response[STATE_MESSAGES],
        STATE_FINAL_ANSWER: final_msg.content,
    }
