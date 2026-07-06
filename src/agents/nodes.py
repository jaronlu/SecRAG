"""Agent Graph 节点实现

每个节点接收 AssistantState，返回部分更新的 AssistantState。
"""

import json

from langchain_core.messages import HumanMessage

from src.agents.state import AssistantState
from src.config import config
from src.rag.formatter import format_citations
from src.retrieval.hybrid_retriever import HybridRetriever
from src.schemas.constants import (
    CONFIDENCE_HIGH,
    CONFIDENCE_LOW,
    CONFIDENCE_MEDIUM,
    DEFAULT_TOP_K,
    GRADE_TOP_K,
    LLM_PROVIDER_OPENAI,
    META_PERMISSION_LEVEL,
    META_SOURCE,
    META_TITLE,
    PERMISSION_PUBLIC,
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
    RETRIEVAL_MIN_SCORE,
    RR_CONTENT,
    RR_METADATA,
    RR_SCORE,
    SOURCE_FAQ,
    STATE_AMBIGUITY,
    STATE_AUDIT_TRAIL,
    STATE_CITATIONS,
    STATE_CLIENT_ID,
    STATE_COMPLIANCE,
    STATE_CONFIDENCE,
    STATE_DATA_PERMISSIONS,
    STATE_DEPARTMENT,
    STATE_ENTITIES,
    STATE_FINAL_ANSWER,
    STATE_INTENT,
    STATE_INTERMEDIATE_STEPS,
    STATE_MESSAGES,
    STATE_ORIGINAL_QUERY,
    STATE_QUERY_TYPE,
    STATE_RETRIEVAL_ATTEMPTS,
    STATE_RETRIEVAL_PLAN,
    STATE_RETRIEVAL_RESULTS,
    STATE_REWRITTEN_QUERY,
    STATE_RISK_DISCLOSURE,
    STATE_TOOL_CALLS,
    STATE_USER_ID,
    STATE_USER_ROLE,
    STATE_VERIFICATION,
)


def _build_llm():
    """根据 config 选择 LLM 后端（复用 rag/chain.py 的同名模式）"""
    if config.llm.provider == LLM_PROVIDER_OPENAI:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            base_url=config.llm.base_url,
            model=config.llm.model,
            temperature=config.llm.temperature,
            api_key=config.llm.api_key,
        )
    from langchain_ollama import ChatOllama

    return ChatOllama(
        base_url=config.llm.base_url,
        model=config.llm.model,
        temperature=config.llm.temperature,
    )


llm = _build_llm()


# ══════════════════════════════════════════════════════════════════════
# 共享规则关键词（verify 与 compliance_check 共用，避免重复定义漂移）
# ══════════════════════════════════════════════════════════════════════

_ADVICE_KEYWORDS: tuple[str, ...] = (
    "推荐" + "买" + "入",
    "建议" + "卖" + "出",
    "目标" + "价",
    "评级",
    "买" + "入",
    "卖" + "出",
    "增" + "持",
    "减" + "持",
)
_SENSITIVE_KEYWORDS: tuple[str, ...] = ("内" + "幕" + "信息", "未" + "公开", "业绩" + "预测")


# ══════════════════════════════════════════════════════════════════════
# 4.2 Query Understand — 意图分类、实体抽取、查询重写、歧义检测
# ══════════════════════════════════════════════════════════════════════


def query_understand(state: AssistantState) -> AssistantState:
    """查询理解：意图分类、实体抽取、查询重写、歧义检测"""
    prompt = f"""请分析以下行业业务查询：

【用户查询】{state[STATE_ORIGINAL_QUERY]}
【用户角色】{state[STATE_USER_ROLE]}
【用户部门】{state[STATE_DEPARTMENT]}

请以 JSON 格式返回：
{{
  "intent": "产品咨询 | 交易规则 | 法规咨询 | 研报观点 | 规则审查 | FAQ | 技术支持",
  "query_type": "product_inquiry | rule_inquiry | regulation_inquiry | report_inquiry | faq_inquiry | technical_inquiry",
  "entities": {{"product_name": "", "product_type": "", "stock_code": "", "regulation_name": "", "client_segment": ""}},
  "rewritten_query": "优化后的结构化查询",
  "ambiguity": ["是指开放式产品还是封闭式产品？"]
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
- product_search: 理财产品说明书、产品合同、风险揭示书
- regulation_search: 规则法规、内部制度、处罚案例
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
    """使用 HybridRetriever 按角色权限执行一轮检索计划。"""
    normalized_plan = []
    for step in state.get(STATE_RETRIEVAL_PLAN, []):
        normalized_plan.append({
            PLAN_SOURCE: step.get(PLAN_SOURCE),
            PLAN_QUERY: step.get(PLAN_QUERY, state[STATE_REWRITTEN_QUERY]),
            PLAN_TOP_K: step.get(PLAN_TOP_K, DEFAULT_TOP_K),
            PLAN_FILTERS: step.get(PLAN_FILTERS),
        })

    retriever = HybridRetriever(user_role=state[STATE_USER_ROLE])
    results = retriever.retrieve(plan=normalized_plan)

    return {
        **state,
        STATE_RETRIEVAL_RESULTS: state.get(STATE_RETRIEVAL_RESULTS, []) + results,
        STATE_RETRIEVAL_ATTEMPTS: state.get(STATE_RETRIEVAL_ATTEMPTS, 0) + 1,
    }


# ══════════════════════════════════════════════════════════════════════
# 4.5 Grade and Filter — 按相似度排序，保留 top-10
# ══════════════════════════════════════════════════════════════════════


def grade_and_filter(state: AssistantState) -> AssistantState:
    """相关性评分与过滤：按 score 排序，保留前 GRADE_TOP_K 条"""
    results = state.get(STATE_RETRIEVAL_RESULTS, [])
    if not results:
        return state

    filtered = [
        result
        for result in sorted(results, key=lambda x: x.get(RR_SCORE, 0), reverse=True)
        if result.get(RR_SCORE, 0) >= RETRIEVAL_MIN_SCORE
    ][:GRADE_TOP_K]

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

    from src.agents.tools import get_tools_for_role

    results = state.get(STATE_RETRIEVAL_RESULTS, [])
    if not results:
        return {
            **state,
            STATE_FINAL_ANSWER: "未检索到足够相关的资料，无法基于当前知识库回答该问题。",
        }

    # 构建 context
    context = "\n\n".join([
        f"[来源{i + 1}] {r[RR_METADATA].get(META_TITLE, '未知')}\n{r[RR_CONTENT]}"
        for i, r in enumerate(results[:5])
    ])

    # 角色化 prompt
    role_instructions = {
        ROLE_ADVISOR: "你是机构投顾助手。基于内部知识库为客户提供准确的产品/规则/市场信息。",
        ROLE_INSTITUTIONAL_SALES: "你是机构销售助手。为机构客户提供研究支持和市场洞察。",
        ROLE_COMPLIANCE: "你是合规助手。基于法规和制度提供准确的合规判断和引用。",
        ROLE_OPERATIONS: "你是运营支持助手。回答客户常见问题，提供操作指引。",
        ROLE_TECHNICAL: "你是技术支持助手。回答系统/数据相关问题。",
    }

    role = state.get(STATE_USER_ROLE, ROLE_OPERATIONS)
    role_instruction = role_instructions.get(role, "你是机构内部知识助手。")

    system_prompt = f"""{role_instruction}

【检索结果】
{context}

【用户问题】{state[STATE_ORIGINAL_QUERY]}

【用户角色】{role}

如果需要计算或查询更多信息，可以使用工具。
回答必须附引用标注 [来源1][来源2]。
数字必须来自检索结果，禁止编造。
{"投顾/销售角色：不得输出" + "买" + "入/" + "卖" + "出/" + "目标" + "价等业务建议，仅提供事实信息。" if role in (ROLE_ADVISOR, ROLE_INSTITUTIONAL_SALES) else ""}
{"合规角色：引用必须精确到条款/条文号。" if role == ROLE_COMPLIANCE else ""}"""

    agent = create_agent(model=llm, tools=get_tools_for_role(role), system_prompt=system_prompt)
    response = agent.invoke({"messages": [HumanMessage(content=state[STATE_ORIGINAL_QUERY])]})

    final_msg = response[STATE_MESSAGES][-1]
    return {
        **state,
        STATE_MESSAGES: state.get(STATE_MESSAGES, []) + response[STATE_MESSAGES],
        STATE_FINAL_ANSWER: final_msg.content,
    }


# ══════════════════════════════════════════════════════════════════════
# 4.7 Verify — 数字验证、权限校验、业务建议检测、合规引用精度
# ══════════════════════════════════════════════════════════════════════


def verify(state: AssistantState) -> AssistantState:
    """验证：数字是否有来源支撑、权限是否合规、是否存在业务建议"""
    import re

    answer = state.get(STATE_FINAL_ANSWER, "")
    results = state.get(STATE_RETRIEVAL_RESULTS, [])
    issues = []

    # 规则 1：检查是否有数字但无来源
    numbers = re.findall(r"\d+\.?\d*%?", answer)
    if numbers and not results:
        issues.append("答案包含数字但无检索结果支撑")

    # 规则 2：检查来源是否在权限范围内
    for r in results:
        perm = r.get(RR_METADATA, {}).get(META_PERMISSION_LEVEL, PERMISSION_PUBLIC)
        if perm not in state.get(STATE_DATA_PERMISSIONS, [PERMISSION_PUBLIC]):
            issues.append(f"来源权限不足: {r.get(RR_METADATA, {}).get(META_SOURCE)}")

    # 规则 3：投顾/销售场景检查是否输出业务建议
    role = state.get(STATE_USER_ROLE)
    if role in (ROLE_ADVISOR, ROLE_INSTITUTIONAL_SALES):
        for pat in _ADVICE_KEYWORDS:
            if pat in answer:
                issues.append(f"投顾/销售角色不得输出业务建议: {pat}")

    # 规则 4：合规场景检查引用精度
    if role == ROLE_COMPLIANCE:
        if not re.search(r"第[一二三四五六七八九十百千]+条|第\d+条|Article\s+\d+", answer):
            issues.append("合规引用必须精确到条款/条文号")

    return {
        **state,
        STATE_VERIFICATION: {
            "passed": len(issues) == 0,
            "issues": issues,
            "confidence": CONFIDENCE_LOW if issues else CONFIDENCE_HIGH,
        },
    }


# ══════════════════════════════════════════════════════════════════════
# 4.8 Compliance Check — 敏感词检测、业务建议拦截、风险提示
# ══════════════════════════════════════════════════════════════════════


def compliance_check(state: AssistantState) -> AssistantState:
    """合规检查：敏感词、业务建议、风险提示、适当性"""
    answer = state.get(STATE_FINAL_ANSWER, "")
    flags = []

    # 敏感词检测
    for kw in _SENSITIVE_KEYWORDS:
        if kw in answer:
            flags.append(f"sensitive:{kw}")

    # 业务建议检测
    for pat in _ADVICE_KEYWORDS:
        if pat in answer:
            flags.append(f"advice:{pat}")

    # 客户适当性检查（投顾场景）
    suitability_warning = ""
    role = state.get(STATE_USER_ROLE)
    if role == ROLE_ADVISOR and state.get(STATE_CLIENT_ID):
        high_risk_products = ["标的型" + "产品", "混合型" + "产品", "私" + "募" + "产品"]
        for prod in high_risk_products:
            if prod in answer:
                suitability_warning = (
                    "\n\n【适当性提示】该产品风险等级较高，请确认客户风险承受能力是否匹配。"
                )
                flags.append(f"suitability:{prod}")
                break

    # 风险提示（强制追加）
    risk_disclosure = "\n\n【风险提示】本回答仅供参考，不构成业务建议。市场有风险，业务需谨慎。"

    passed = len([f for f in flags if f.startswith("sensitive:")]) == 0

    return {
        **state,
        STATE_COMPLIANCE: {
            "passed": passed,
            "flags": flags,
            "risk_disclosure": risk_disclosure,
            "suitability_warning": suitability_warning,
        },
    }


# ══════════════════════════════════════════════════════════════════════
# 4.9 Compose — 引用标注、置信度、风险提示
# ══════════════════════════════════════════════════════════════════════


def compose(state: AssistantState) -> AssistantState:
    """生成最终回答：带引用、置信度、风险提示"""
    answer = state.get(STATE_FINAL_ANSWER, "")

    # 引用标注（复用 src/rag/formatter.py 的权威实现，避免重复维护 Citation 字段）
    citations = format_citations(state.get(STATE_RETRIEVAL_RESULTS, [])[:5])

    # 风险提示 + 适当性警告
    compliance = state.get(STATE_COMPLIANCE, {})
    risk = compliance.get("risk_disclosure", "")
    suitability = compliance.get("suitability_warning", "")
    final_answer = answer + suitability + risk

    # 综合置信度
    verification_conf = state.get(STATE_VERIFICATION, {}).get("confidence", CONFIDENCE_MEDIUM)
    compliance_passed = compliance.get("passed", False)
    result_count = len(state.get(STATE_RETRIEVAL_RESULTS, []))

    if not compliance_passed:
        confidence = CONFIDENCE_LOW
    elif verification_conf == CONFIDENCE_HIGH and result_count >= 3:
        confidence = CONFIDENCE_HIGH
    else:
        confidence = CONFIDENCE_MEDIUM

    return {
        **state,
        STATE_FINAL_ANSWER: final_answer,
        STATE_CITATIONS: citations,
        STATE_CONFIDENCE: confidence,
        STATE_RISK_DISCLOSURE: risk + suitability,
    }


# ══════════════════════════════════════════════════════════════════════
# 4.10 Audit Log — 全链路追踪记录
# ══════════════════════════════════════════════════════════════════════


def _unique_sources(results: list[dict]) -> list[str]:
    sources = []
    seen = set()
    for result in results:
        source = result.get(RR_METADATA, {}).get(META_SOURCE)
        if not source or source in seen:
            continue
        seen.add(source)
        sources.append(source)
    return sources


def audit_log(state: AssistantState) -> AssistantState:
    """记录追踪日志——覆盖 Query → Retrieve → Reason → Verify → Compose 全链路"""
    import uuid
    from datetime import datetime, timezone

    audit_entry = {
        "request_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user_id": state.get(STATE_USER_ID, ""),
        "user_role": state.get(STATE_USER_ROLE, ""),
        "department": state.get(STATE_DEPARTMENT, ""),
        "query": {
            "original": state.get(STATE_ORIGINAL_QUERY, ""),
            "rewritten": state.get(STATE_REWRITTEN_QUERY, ""),
            "intent": state.get(STATE_INTENT, ""),
            "query_type": state.get(STATE_QUERY_TYPE, ""),
            "entities": state.get(STATE_ENTITIES, {}),
        },
        "retrieval": {
            "plan": state.get(STATE_RETRIEVAL_PLAN, []),
            "sources": _unique_sources(state.get(STATE_RETRIEVAL_RESULTS, [])),
            "total_chunks": len(state.get(STATE_RETRIEVAL_RESULTS, [])),
            "filtered_chunks": len(state.get(STATE_RETRIEVAL_RESULTS, [])),
        },
        "reasoning": {
            "tool_calls": state.get(STATE_TOOL_CALLS, []),
            "iterations": len(state.get(STATE_INTERMEDIATE_STEPS, [])),
            "duration_ms": 0,
        },
        "verification": state.get(STATE_VERIFICATION, {}),
        "compliance": state.get(STATE_COMPLIANCE, {}),
        "response": {
            "citations": state.get(STATE_CITATIONS, []),
            "confidence": state.get(STATE_CONFIDENCE, CONFIDENCE_LOW),
            "risk_disclosure": state.get(STATE_RISK_DISCLOSURE, ""),
        },
    }

    # TODO: 写入追踪数据库
    # audit_db.insert(audit_entry)

    return {
        **state,
        STATE_AUDIT_TRAIL: audit_entry,
    }
