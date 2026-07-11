"""Agent Graph 节点实现

每个节点接收 AssistantState，返回部分更新的 AssistantState。
"""

import json
from typing import Any, cast

from langchain_core.messages import HumanMessage, ToolMessage

from src.agents.state import AssistantState
from src.config import config
from src.retrieval.hybrid_retriever import HybridRetriever
from src.schemas.constants import (
    CONFIDENCE_HIGH,
    CONFIDENCE_LOW,
    CONFIDENCE_MEDIUM,
    DEFAULT_TOP_K,
    GRADE_TOP_K,
    LLM_PROVIDER_OPENAI,
    META_CHUNK_ID,
    META_SOURCE,
    META_TITLE,
    PERMISSION_PUBLIC,
    PLAN_FILTERS,
    PLAN_QUERY,
    PLAN_SOURCE,
    PLAN_TOP_K,
    RETRIEVAL_MIN_SCORE,
    ROLE_ADVISOR,
    ROLE_ALLOWED_SOURCES,
    ROLE_COMPLIANCE,
    ROLE_INSTITUTIONAL_SALES,
    ROLE_OPERATIONS,
    ROLE_TECHNICAL,
    RR_CONTENT,
    RR_DENIED,
    RR_METADATA,
    RR_SCORE,
    STATE_AMBIGUITY,
    STATE_AUDIT_TRAIL,
    STATE_CITATIONS,
    STATE_CHAT_HISTORY,
    STATE_CLIENT_ID,
    STATE_COMPLIANCE,
    STATE_CONVERSATION_SUMMARY,
    STATE_CONFIDENCE,
    STATE_DATA_PERMISSIONS,
    STATE_DEPARTMENT,
    STATE_ENTITIES,
    STATE_FINAL_ANSWER,
    STATE_INTENT,
    STATE_MESSAGES,
    STATE_ORIGINAL_QUERY,
    STATE_QUERY_TYPE,
    STATE_RESOLVED_QUERY,
    STATE_REASON_ATTEMPTS,
    STATE_RETRIEVAL_ATTEMPTS,
    STATE_RETRIEVAL_PLAN,
    STATE_RETRIEVAL_RESULTS,
    STATE_REWRITTEN_QUERY,
    STATE_RISK_DISCLOSURE,
    STATE_THREAD_ID,
    STATE_TOOL_CALLS,
    STATE_USER_ID,
    STATE_USER_ROLE,
    STATE_VERIFICATION,
)
from src.schemas.typed_dicts import RetrievalPlanStep, ToolCallDict
from src.utils.compliance import ComplianceChecker, INVESTMENT_ADVICE_PATTERNS
from src.utils.verifier import CitationExtractor, ComprehensiveVerifier


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


def _get_audit_store():
    from src.utils.audit import SQLiteAuditStore

    return SQLiteAuditStore(config.audit_db_path)


def _get_conversation_store():
    from src.utils.conversation import SQLiteConversationStore

    return SQLiteConversationStore(config.conversation_db_path)


# ══════════════════════════════════════════════════════════════════════
# 共享规则关键词（verify 与 compliance_check 共用，避免重复定义漂移）
# ══════════════════════════════════════════════════════════════════════

_ADVICE_KEYWORDS = INVESTMENT_ADVICE_PATTERNS
_COMPLIANCE_CHECKER = ComplianceChecker()
_CITATION_EXTRACTOR = CitationExtractor()
_VERIFIER = ComprehensiveVerifier()


def _with_state_updates(state: AssistantState, updates: dict[str, Any]) -> AssistantState:
    return cast(AssistantState, {**state, **updates})


def _normalize_plan_step(step: object, default_query: str = "") -> RetrievalPlanStep | None:
    if not isinstance(step, dict):
        return None

    source = step.get(PLAN_SOURCE)
    if not isinstance(source, str):
        return None

    query = step.get(PLAN_QUERY, default_query)
    top_k = step.get(PLAN_TOP_K, DEFAULT_TOP_K)
    filters = step.get(PLAN_FILTERS)

    return RetrievalPlanStep(
        source=source,
        query=query if isinstance(query, str) else default_query,
        top_k=top_k if isinstance(top_k, int) else DEFAULT_TOP_K,
        filters=filters if isinstance(filters, dict) else None,
    )


# ══════════════════════════════════════════════════════════════════════
# 4.1 Conversation Context — 校验会话并加载用户可见历史
# ══════════════════════════════════════════════════════════════════════


def load_conversation_context(state: AssistantState) -> AssistantState:
    """Load recent visible messages and entity summary for the current owner."""
    history, summary = _get_conversation_store().load_context(
        thread_id=state[STATE_THREAD_ID],
        user_id=state[STATE_USER_ID],
    )
    return _with_state_updates(
        state,
        {
            STATE_CHAT_HISTORY: history,
            STATE_CONVERSATION_SUMMARY: summary,
        },
    )


def resolve_followup_query(state: AssistantState) -> AssistantState:
    """Resolve pronoun-style follow-ups using stored entity context only."""
    query = state[STATE_ORIGINAL_QUERY]
    summary = state.get(STATE_CONVERSATION_SUMMARY, "")
    followup_markers = ("它", "这个", "该产品", "该公司", "那", "上述", "前面")
    resolved = f"基于会话实体（{summary}），{query}" if summary and any(
        marker in query for marker in followup_markers
    ) else query
    return _with_state_updates(state, {STATE_RESOLVED_QUERY: resolved})


# ══════════════════════════════════════════════════════════════════════
# 4.2 Query Understand — 意图分类、实体抽取、查询重写、歧义检测
# ══════════════════════════════════════════════════════════════════════


def query_understand(state: AssistantState) -> AssistantState:
    """查询理解：意图分类、实体抽取、查询重写、歧义检测"""
    effective_query = state.get(STATE_RESOLVED_QUERY) or state[STATE_ORIGINAL_QUERY]
    prompt = f"""请分析以下行业业务查询：

【用户查询】{effective_query}
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
            "rewritten_query": effective_query,
            "ambiguity": [],
        }

    return _with_state_updates(state, {
        STATE_INTENT: result.get("intent", "unknown"),
        STATE_QUERY_TYPE: result.get("query_type", "unknown"),
        STATE_ENTITIES: result.get("entities", {}),
        STATE_REWRITTEN_QUERY: result.get("rewritten_query", effective_query),
        STATE_AMBIGUITY: result.get("ambiguity", []),
    })


# ══════════════════════════════════════════════════════════════════════
# 4.3 Planner — 根据角色权限生成多源检索计划
# ══════════════════════════════════════════════════════════════════════


def planner(state: AssistantState) -> AssistantState:
    """检索计划生成：根据意图、角色、查询类型生成多步检索计划"""
    allowed_sources = ROLE_ALLOWED_SOURCES.get(state[STATE_USER_ROLE], [])

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
        parsed_plan = json.loads(raw)
    except json.JSONDecodeError:
        parsed_plan = (
            [{
                PLAN_SOURCE: allowed_sources[0],
                PLAN_QUERY: state[STATE_REWRITTEN_QUERY],
                PLAN_TOP_K: 3,
            }]
            if allowed_sources
            else []
        )

    # 按角色权限过滤：去掉 LLM 可能越权生成的 source
    raw_steps = parsed_plan if isinstance(parsed_plan, list) else []
    filtered_plan: list[RetrievalPlanStep] = []
    for raw_step in raw_steps:
        step = _normalize_plan_step(raw_step, state[STATE_REWRITTEN_QUERY])
        if step is not None and step.get(PLAN_SOURCE) in allowed_sources:
            filtered_plan.append(step)

    return _with_state_updates(state, {
        STATE_RETRIEVAL_PLAN: filtered_plan,
    })


# ══════════════════════════════════════════════════════════════════════
# 4.4 Retrieve — 按计划并行执行多源检索
# ══════════════════════════════════════════════════════════════════════


def retrieve(state: AssistantState) -> AssistantState:
    """使用 HybridRetriever 按角色权限执行一轮检索计划。"""
    normalized_plan: list[RetrievalPlanStep] = []
    for step in state.get(STATE_RETRIEVAL_PLAN, []):
        normalized_plan.append(
            RetrievalPlanStep(
                source=step.get(PLAN_SOURCE, ""),
                query=step.get(PLAN_QUERY, state[STATE_REWRITTEN_QUERY]),
                top_k=step.get(PLAN_TOP_K, DEFAULT_TOP_K),
                filters=step.get(PLAN_FILTERS),
            )
        )

    retriever = HybridRetriever(
        user_role=state[STATE_USER_ROLE],
        data_permissions=state.get(STATE_DATA_PERMISSIONS, [PERMISSION_PUBLIC]),
    )
    results = retriever.retrieve(plan=normalized_plan)

    return _with_state_updates(state, {
        STATE_RETRIEVAL_RESULTS: state.get(STATE_RETRIEVAL_RESULTS, []) + results,
        STATE_RETRIEVAL_ATTEMPTS: state.get(STATE_RETRIEVAL_ATTEMPTS, 0) + 1,
    })


# ══════════════════════════════════════════════════════════════════════
# 4.5 Grade and Filter — 按相似度排序，保留 top-10
# ══════════════════════════════════════════════════════════════════════


def grade_and_filter(state: AssistantState) -> AssistantState:
    """相关性评分与过滤：按 score 排序，保留前 GRADE_TOP_K 条"""
    results = state.get(STATE_RETRIEVAL_RESULTS, [])
    if not results:
        return state

    denied = [result for result in results if result.get(RR_DENIED)]
    filtered = []
    seen_evidence = set()
    for result in sorted(results, key=lambda x: x.get(RR_SCORE, 0), reverse=True):
        if result.get(RR_DENIED) or result.get(RR_SCORE, 0) < RETRIEVAL_MIN_SCORE:
            continue
        metadata = result.get(RR_METADATA, {})
        key = (
            metadata.get(META_SOURCE),
            metadata.get(META_CHUNK_ID) or result.get(RR_CONTENT, ""),
        )
        if key in seen_evidence:
            continue
        seen_evidence.add(key)
        filtered.append(result)
        if len(filtered) >= GRADE_TOP_K:
            break

    return _with_state_updates(state, {
        STATE_RETRIEVAL_RESULTS: filtered + denied,
    })


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
    reason_attempts = state.get(STATE_REASON_ATTEMPTS, 0) + 1
    if not results:
        return _with_state_updates(state, {
            STATE_FINAL_ANSWER: "未检索到足够相关的资料，无法基于当前知识库回答该问题。",
            STATE_REASON_ATTEMPTS: reason_attempts,
        })

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

    【用户问题】{state.get(STATE_RESOLVED_QUERY) or state[STATE_ORIGINAL_QUERY]}

【用户角色】{role}

如果需要计算或查询更多信息，可以使用工具。
回答必须附引用标注 [来源1][来源2]。
数字必须来自检索结果或成功的工具输出，禁止编造。
{"投顾/销售角色：不得输出" + "买" + "入/" + "卖" + "出/" + "目标" + "价等业务建议，仅提供事实信息。" if role in (ROLE_ADVISOR, ROLE_INSTITUTIONAL_SALES) else ""}
{"合规角色：引用必须精确到条款/条文号。" if role == ROLE_COMPLIANCE else ""}"""

    agent = create_agent(model=llm, tools=get_tools_for_role(role), system_prompt=system_prompt)
    response = agent.invoke({
        "messages": [
            HumanMessage(content=state.get(STATE_RESOLVED_QUERY) or state[STATE_ORIGINAL_QUERY])
        ]
    })

    final_msg = response[STATE_MESSAGES][-1]
    final_content = final_msg.content if isinstance(final_msg.content, str) else str(final_msg.content)
    tool_calls: list[ToolCallDict] = list(state.get(STATE_TOOL_CALLS, []))
    for message in response[STATE_MESSAGES]:
        if isinstance(message, ToolMessage):
            output = message.content if isinstance(message.content, str) else str(message.content)
            tool_calls.append(
                ToolCallDict(
                    tool=message.name or "unknown",
                    output=output,
                    success=message.status != "error",
                )
            )
    return _with_state_updates(state, {
        STATE_MESSAGES: state.get(STATE_MESSAGES, []) + response[STATE_MESSAGES],
        STATE_TOOL_CALLS: tool_calls,
        STATE_FINAL_ANSWER: final_content,
        STATE_REASON_ATTEMPTS: reason_attempts,
    })


# ══════════════════════════════════════════════════════════════════════
# 4.7 Citation Extraction and Verification
# ══════════════════════════════════════════════════════════════════════


def extract_citations(state: AssistantState) -> AssistantState:
    """Extract citations only from this turn's usable retrieval results."""
    query = (
        state.get(STATE_RESOLVED_QUERY)
        or state.get(STATE_REWRITTEN_QUERY)
        or state.get(STATE_ORIGINAL_QUERY, "")
    )
    citations = _CITATION_EXTRACTOR.extract(
        state.get(STATE_RETRIEVAL_RESULTS, []),
        query=query,
    )
    return _with_state_updates(state, {STATE_CITATIONS: citations})


def verify(state: AssistantState) -> AssistantState:
    """Run source, number, consistency, and hallucination verification."""
    verification = _VERIFIER.verify(
        answer=state.get(STATE_FINAL_ANSWER, ""),
        citations=state.get(STATE_CITATIONS, []),
        retrieval_results=state.get(STATE_RETRIEVAL_RESULTS, []),
        tool_calls=state.get(STATE_TOOL_CALLS, []),
    )

    role = state.get(STATE_USER_ROLE)
    issues = list(verification.get("issues", []))
    if role in (ROLE_ADVISOR, ROLE_INSTITUTIONAL_SALES):
        for pattern in _ADVICE_KEYWORDS:
            if pattern in state.get(STATE_FINAL_ANSWER, ""):
                issues.append(f"投顾/销售角色不得输出业务建议: {pattern}")
    if issues:
        verification.update(passed=False, issues=issues, confidence=CONFIDENCE_LOW)
    return _with_state_updates(state, {STATE_VERIFICATION: verification})


# ══════════════════════════════════════════════════════════════════════
# 4.8 Compliance Check — 敏感词检测、业务建议拦截、风险提示
# ══════════════════════════════════════════════════════════════════════


def compliance_check(state: AssistantState) -> AssistantState:
    """合规检查：敏感词、业务建议、风险提示、适当性"""
    answer = state.get(STATE_FINAL_ANSWER, "")
    compliance = _COMPLIANCE_CHECKER.check(
        answer,
        user_role=state.get(STATE_USER_ROLE),
        client_id=state.get(STATE_CLIENT_ID),
    )

    return _with_state_updates(state, {
        STATE_COMPLIANCE: compliance,
    })


def permission_denied_response(state: AssistantState) -> AssistantState:
    """Terminate before LLM reasoning when every retrieval result is denied."""
    return _with_state_updates(
        state,
        {
            STATE_FINAL_ANSWER: "当前角色无权限访问完成该请求所需的数据源。",
            STATE_CITATIONS: [],
            STATE_CONFIDENCE: CONFIDENCE_LOW,
            STATE_RISK_DISCLOSURE: "",
            STATE_VERIFICATION: {
                "passed": False,
                "issues": ["permission_denied"],
                "confidence": CONFIDENCE_LOW,
            },
            STATE_COMPLIANCE: {
                "passed": False,
                "flags": ["permission_denied"],
                "risk_disclosure": "",
            },
        },
    )


# ══════════════════════════════════════════════════════════════════════
# 4.9 Compose — 引用标注、置信度、风险提示
# ══════════════════════════════════════════════════════════════════════


def compose(state: AssistantState) -> AssistantState:
    """生成最终回答：带引用、置信度、风险提示"""
    answer = state.get(STATE_FINAL_ANSWER, "")
    citations = state.get(STATE_CITATIONS, [])

    # 风险提示 + 适当性警告
    compliance = state.get(STATE_COMPLIANCE, {})
    risk = compliance.get("risk_disclosure", "")
    suitability = compliance.get("suitability_warning", "")
    verification_passed = state.get(STATE_VERIFICATION, {}).get("passed", False)
    compliance_passed = compliance.get("passed", False)
    if not verification_passed:
        answer = "当前答案未通过来源或数字验证，无法安全返回。请补充可验证资料后重试。"
        citations = []
    elif not compliance_passed:
        answer = "当前请求或生成内容未通过合规检查，已停止输出。"
        citations = []
    final_answer = answer + suitability + risk

    # 综合置信度
    verification_conf = state.get(STATE_VERIFICATION, {}).get("confidence", CONFIDENCE_MEDIUM)
    result_count = len([
        result
        for result in state.get(STATE_RETRIEVAL_RESULTS, [])
        if not result.get(RR_DENIED)
    ])

    if not compliance_passed:
        confidence = CONFIDENCE_LOW
    elif verification_conf == CONFIDENCE_HIGH and result_count >= 3:
        confidence = CONFIDENCE_HIGH
    else:
        confidence = CONFIDENCE_MEDIUM

    return _with_state_updates(state, {
        STATE_FINAL_ANSWER: final_answer,
        STATE_CITATIONS: citations,
        STATE_CONFIDENCE: confidence,
        STATE_RISK_DISCLOSURE: risk + suitability,
    })


def persist_conversation_turn(state: AssistantState) -> AssistantState:
    """Persist the visible turn and durable audit outbox event atomically."""
    _get_conversation_store().insert_turn(state)
    return state


# ══════════════════════════════════════════════════════════════════════
# 4.10 Audit Log — 全链路追踪记录
# ══════════════════════════════════════════════════════════════════════


def audit_log(state: AssistantState) -> AssistantState:
    """记录追踪日志——覆盖 Query → Retrieve → Reason → Verify → Compose 全链路

    实现委托给 src/utils/audit.py 的 AuditLogger（AuditEntry 模型的权威构建者），
    避免与该模块重复维护同一份字段拼装逻辑。
    """
    from src.utils.audit import AuditLogger, audit_entry_to_trail

    audit_entry = AuditLogger().log(state)
    conversation_store = _get_conversation_store()
    try:
        _get_audit_store().insert(audit_entry)
        if conversation_store.get_outbox_status(audit_entry.request_id) is not None:
            conversation_store.mark_outbox_processed(audit_entry.request_id)
    except Exception as exc:
        if conversation_store.get_outbox_status(audit_entry.request_id) is not None:
            conversation_store.mark_outbox_failed(audit_entry.request_id, str(exc))
        raise

    return _with_state_updates(state, {
        STATE_AUDIT_TRAIL: audit_entry_to_trail(audit_entry),
    })
