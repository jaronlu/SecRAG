"""Agent Graph 节点实现

每个节点接收 AssistantState，返回部分更新的 AssistantState。
"""

import json
import re
import time
from functools import lru_cache
from typing import Any, Callable, Literal, cast

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

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
    MAX_TOOL_ITERATIONS,
    META_CHUNK_ID,
    META_DATE,
    META_SOURCE,
    META_STOCK_CODE,
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
    SOURCE_REPORT,
    STATE_AMBIGUITY,
    STATE_AUDIT_TRAIL,
    STATE_CHAT_HISTORY,
    STATE_CITATIONS,
    STATE_CLIENT_ID,
    STATE_COMPLIANCE,
    STATE_CONFIDENCE,
    STATE_CONVERSATION_SUMMARY,
    STATE_DATA_PERMISSIONS,
    STATE_DEPARTMENT,
    STATE_ENTITIES,
    STATE_FINAL_ANSWER,
    STATE_INTENT,
    STATE_INTERMEDIATE_STEPS,
    STATE_MESSAGES,
    STATE_ORIGINAL_QUERY,
    STATE_QUERY_TYPE,
    STATE_REASON_ATTEMPTS,
    STATE_REASON_MESSAGE_START,
    STATE_REASON_STARTED_PERF_COUNTER,
    STATE_RESOLVED_QUERY,
    STATE_RETRIEVAL_ATTEMPTS,
    STATE_RETRIEVAL_FILTERED_CHUNKS,
    STATE_RETRIEVAL_PLAN,
    STATE_RETRIEVAL_RESULTS,
    STATE_RETRIEVAL_TOTAL_CHUNKS,
    STATE_REWRITTEN_QUERY,
    STATE_RISK_DISCLOSURE,
    STATE_THREAD_ID,
    STATE_TOOL_CALLS,
    STATE_TOOL_ITERATIONS,
    STATE_TOOL_MESSAGE_CURSOR,
    STATE_USER_ID,
    STATE_USER_ROLE,
    STATE_VERIFICATION,
)
from src.schemas.typed_dicts import IntermediateStep, RetrievalPlanStep, ToolCallDict
from src.utils.compliance import (
    INVESTMENT_ADVICE_PATTERNS,
    TARGET_PRICE_PATTERN,
    ComplianceChecker,
)
from src.utils.verifier import CitationExtractor, ComprehensiveVerifier


def _build_llm():
    """根据 config 选择 LLM 后端（复用 rag/chain.py 的同名模式）"""
    # 延迟导入 provider 依赖，避免项目启动时必须安装所有 LLM 后端
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


# 模块级 LLM 实例，所有节点共享（避免每个节点重复创建）
llm = _build_llm()


def _get_audit_store():
    """懒加载审计存储（首次调用时才导入并创建实例）"""
    # 审计库只在最终落盘时用到，延迟导入可减少启动依赖
    from src.utils.audit import SQLiteAuditStore

    return SQLiteAuditStore(config.audit_db_path)


def _get_conversation_store():
    """懒加载对话存储"""
    # 对话历史同样不需要在模块导入阶段就初始化连接
    from src.utils.conversation import SQLiteConversationStore

    return SQLiteConversationStore(config.conversation_db_path)


# 共享规则关键词（verify 与 compliance_check 共用，避免重复定义漂移）
_ADVICE_KEYWORDS = INVESTMENT_ADVICE_PATTERNS
# 以下实例初始化开销低且被多个节点复用，放在模块顶层避免重复创建
_COMPLIANCE_CHECKER = ComplianceChecker()
_CITATION_EXTRACTOR = CitationExtractor()
_VERIFIER = ComprehensiveVerifier()


def _format_evidence_metadata(metadata: dict[str, Any]) -> str:
    """把元数据格式化成结构化证据字符串，供 prompt 使用"""
    # 只保留存在的字段，避免 prompt 里出现大量空白键值对
    fields = (
        ("institution", "机构"),
        ("rating", "评级"),
        (META_DATE, "来源日期"),
        (META_STOCK_CODE, "股票代码"),
        (META_SOURCE, "来源"),
    )
    values = [f"{label}={metadata[key]}" for key, label in fields if metadata.get(key)]
    return f"结构化证据：{'；'.join(values)}" if values else ""


def _structure_answer(content: str) -> str:
    """Normalize visible answer text into a stable Markdown structure."""
    answer = content.strip()
    # 去掉 LLM 可能输出的英文前缀
    for prefix in ("Answer:", "回答：", "回答:"):
        if answer.startswith(prefix):
            answer = answer[len(prefix) :].lstrip()
            break
    # 截断到 Citations / Audit Trail 之前，避免内部标记泄露给用户
    for marker in ("\nCitations:", "\nAudit Trail:"):
        if marker in answer:
            answer = answer.split(marker, 1)[0].rstrip()
    # 如果已经结构化，或为空，直接返回
    if not answer or answer.startswith("## 结论"):
        return answer

    # 按空行拆成结论和明细，统一套回固定 Markdown 结构
    conclusion, separator, details = answer.partition("\n\n")
    structured = f"## 结论\n\n{conclusion.strip()}"
    if separator and details.strip():
        structured += f"\n\n## 关键结果\n\n{details.strip()}"
    return structured


def _normalize_plan_step(step: object, default_query: str = "") -> RetrievalPlanStep | None:
    """把 LLM 返回的原始检索步骤规范化成 RetrievalPlanStep，失败返回 None"""
    # LLM 输出不可信，先做类型防御，避免后续 .get() 或属性访问崩溃
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


def load_conversation_context(state: AssistantState) -> dict[str, Any]:
    """Load recent visible messages and entity summary for the current owner."""
    # 只读取当前线程 + 当前用户可见的历史，避免串号或跨用户泄露
    history, summary = _get_conversation_store().load_context(
        thread_id=state[STATE_THREAD_ID],
        user_id=state[STATE_USER_ID],
    )
    return {
        STATE_CHAT_HISTORY: history,
        STATE_CONVERSATION_SUMMARY: summary,
    }


def resolve_followup_query(state: AssistantState) -> dict[str, Any]:
    """Resolve pronoun-style follow-ups using stored entity context only."""
    # 仅依赖当前会话摘要做指代消解，不引入外部上下文，降低幻觉风险
    query = state[STATE_ORIGINAL_QUERY]
    summary = state.get(STATE_CONVERSATION_SUMMARY, "")
    followup_markers = ("它", "这个", "该产品", "该公司", "那", "上述", "前面")
    resolved = (
        f"基于会话实体（{summary}），{query}"
        if summary and any(marker in query for marker in followup_markers)
        else query
    )
    return {STATE_RESOLVED_QUERY: resolved}


# ══════════════════════════════════════════════════════════════════════
# 4.2 Query Understand — 意图分类、实体抽取、查询重写、歧义检测
# ══════════════════════════════════════════════════════════════════════


def query_understand(state: AssistantState) -> dict[str, Any]:
    """查询理解：意图分类、实体抽取、查询重写、歧义检测"""
    # 优先使用“指代消解后的查询”，否则回退到原始查询
    effective_query = state.get(STATE_RESOLVED_QUERY) or state[STATE_ORIGINAL_QUERY]
    prompt = f"""请分析以下行业业务查询：

【用户查询】{effective_query}
【用户角色】{state[STATE_USER_ROLE]}
【用户部门】{state[STATE_DEPARTMENT]}

请以 JSON 格式返回：{{
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
        # LLM 未按格式返回 JSON 时，不阻断流程，用默认值兜底
        result = {
            "intent": "unknown",
            "query_type": "unknown",
            "entities": {},
            "rewritten_query": effective_query,
            "ambiguity": [],
        }

    return {
        STATE_INTENT: result.get("intent", "unknown"),
        STATE_QUERY_TYPE: result.get("query_type", "unknown"),
        STATE_ENTITIES: result.get("entities", {}),
        STATE_REWRITTEN_QUERY: result.get("rewritten_query", effective_query),
        STATE_AMBIGUITY: result.get("ambiguity", []),
    }


# ══════════════════════════════════════════════════════════════════════
# 4.3 Planner — 根据角色权限生成多源检索计划
# ══════════════════════════════════════════════════════════════════════


def planner(state: AssistantState) -> dict[str, Any]:
    """检索计划生成：根据意图、角色、查询类型生成多步检索计划"""
    # 先查当前角色允许访问的数据源，作为后续计划的权限边界
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
        # LLM 未返回合法 JSON 时，退化为单源检索，保证流程继续
        parsed_plan = (
            [
                {
                    PLAN_SOURCE: allowed_sources[0],
                    PLAN_QUERY: state[STATE_REWRITTEN_QUERY],
                    PLAN_TOP_K: 3,
                }
            ]
            if allowed_sources
            else []
        )

    # 按角色权限过滤：去掉 LLM 可能越权生成的 source
    raw_steps = parsed_plan if isinstance(parsed_plan, list) else []
    filtered_plan: list[RetrievalPlanStep] = []
    for raw_step in raw_steps:
        step = _normalize_plan_step(raw_step, state[STATE_REWRITTEN_QUERY])
        if step is not None and step.get(PLAN_SOURCE) in allowed_sources:
            # 研报通常按股票组织，补股票代码过滤可提高召回精度
            if step.get(PLAN_SOURCE) == SOURCE_REPORT:
                stock_code = str(state.get(STATE_ENTITIES, {}).get(META_STOCK_CODE, ""))
                if stock_code:
                    filters = dict(step.get(PLAN_FILTERS) or {})
                    # 只保留代码部分，去掉沪市 .SH / 深市 .SZ 等后缀
                    filters[META_STOCK_CODE] = stock_code.split(".", maxsplit=1)[0]
                    step[PLAN_FILTERS] = filters
            filtered_plan.append(step)

    return {STATE_RETRIEVAL_PLAN: filtered_plan}


# ══════════════════════════════════════════════════════════════════════
# 4.4 Retrieve — 按计划并行执行多源检索
# ══════════════════════════════════════════════════════════════════════


def retrieve(state: AssistantState) -> dict[str, Any]:
    """使用 HybridRetriever 按角色权限执行一轮检索计划。"""
    # 先把 state 里可能被合并过的计划重新标准化，避免旧结构残留
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

    # 把本轮结果累加到已有结果上，支持后续重写查询后再检索一轮
    accumulated = state.get(STATE_RETRIEVAL_RESULTS, []) + results
    return {
        STATE_RETRIEVAL_RESULTS: accumulated,
        STATE_RETRIEVAL_ATTEMPTS: state.get(STATE_RETRIEVAL_ATTEMPTS, 0) + 1,
        STATE_RETRIEVAL_TOTAL_CHUNKS: state.get(STATE_RETRIEVAL_TOTAL_CHUNKS, 0) + len(results),
    }


# ══════════════════════════════════════════════════════════════════════
# 4.5 Grade and Filter — 按相似度排序，保留 top-10
# ══════════════════════════════════════════════════════════════════════


def grade_and_filter(state: AssistantState) -> dict[str, Any]:
    """相关性评分与过滤：按 score 排序，保留前 GRADE_TOP_K 条"""
    results = state.get(STATE_RETRIEVAL_RESULTS, [])
    if not results:
        return {}

    # 先把无权限结果摘出来；保留它们是为了后续可明确提示用户“部分结果无权查看”
    denied = [result for result in results if result.get(RR_DENIED)]
    filtered = []
    seen_evidence = set()
    # 先按相似度降序，优先保留高相关证据
    for result in sorted(results, key=lambda x: x.get(RR_SCORE, 0), reverse=True):
        if result.get(RR_DENIED) or result.get(RR_SCORE, 0) < RETRIEVAL_MIN_SCORE:
            continue
        metadata = result.get(RR_METADATA, {})
        # 以来源 + 内容指纹去重，避免同一证据反复占据上下文窗口
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

    return {
        STATE_RETRIEVAL_RESULTS: filtered + denied,
        STATE_RETRIEVAL_FILTERED_CHUNKS: len(filtered),
    }


# ══════════════════════════════════════════════════════════════════════
# 4.6 Reason — LLM 推理 + 工具调用（ReAct Agent）
# ══════════════════════════════════════════════════════════════════════


def _build_reason_system_prompt(state: AssistantState) -> str:
    """Build the role-aware prompt from current retrieval evidence."""
    results = [
        result for result in state.get(STATE_RETRIEVAL_RESULTS, []) if not result.get(RR_DENIED)
    ]

    # 构建 context：只把前 5 条结果喂给 LLM，控制 prompt 长度
    context_parts = []
    for index, result in enumerate(results[:5]):
        metadata = result[RR_METADATA]
        metadata_evidence = _format_evidence_metadata(metadata)
        context_part = f"[来源{index + 1}] {metadata.get(META_TITLE, '未知')}\n{result[RR_CONTENT]}"
        if metadata_evidence:
            context_part += f"\n{metadata_evidence}"
        context_parts.append(context_part)
    context = "\n\n".join(context_parts) or (
        "当前轮没有可用文档检索结果；如问题可由授权工具回答，应调用工具并仅依据成功工具输出作答。"
    )

    # 角色化 prompt：不同岗位的权限边界和回答风格不同
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
只有当前轮存在文档检索结果时，回答才允许并必须附对应的 [来源N]；纯工具回答不得编造文档引用。
数字必须来自检索结果或成功的工具输出，禁止编造。
纯工具回答只能复述成功工具输出中实际存在的字段和值。不得补充工具未返回的字段含义、
数据来源、更新频率、覆盖范围、趋势判断或后续能力；优先直接使用字段名和值，保持简洁。
回答正文必须使用结构化 Markdown：
1. 以 `## 结论` 开头，用 1-2 句话直接回答问题；
2. 有明细时增加 `## 关键结果`：记录型/对比型数据使用 Markdown 表格，操作流程使用有序列表，
   概念解释使用 What / Why / How 要点；
3. 只有存在限制、缺失字段或适用边界时才增加 `## 注意事项`；
4. 不重复输出“字段列表 + 同字段完整表格”，除非用户明确询问字段结构；
5. 不在正文输出 `Answer:`、`Citations:`、`Audit Trail:`、置信度或内部工具调用细节。
6. 只回答用户询问的字段，不主动扩展日期等未询问 metadata；`来源日期` 不得改写为报告日期或发布日期。
{"投顾/销售角色：不得主动建议买卖或生成目标价；可以中性引用有 [来源N] 且可验证的历史研报既有目标价。" if role in (ROLE_ADVISOR, ROLE_INSTITUTIONAL_SALES) else ""}
{"合规角色：引用必须精确到条款/条文号。" if role == ROLE_COMPLIANCE else ""}"""

    return system_prompt


def _excluded_retrieval_sources(state: AssistantState) -> tuple[str, ...]:
    """Return sources already satisfied by usable outer-graph retrieval results."""
    has_usable_results = any(
        not result.get(RR_DENIED) for result in state.get(STATE_RETRIEVAL_RESULTS, [])
    )
    if not has_usable_results:
        return ()
    return tuple(
        sorted({
            step.get(PLAN_SOURCE, "")
            for step in state.get(STATE_RETRIEVAL_PLAN, [])
            if step.get(PLAN_SOURCE)
        })
    )


def _reason_tools(state: AssistantState):
    """Resolve the tools visible and executable for the current state."""
    from src.agents.tools import get_tools_for_role

    return get_tools_for_role(
        state.get(STATE_USER_ROLE, ROLE_OPERATIONS),
        excluded_retrieval_sources=set(_excluded_retrieval_sources(state)),
    )


@lru_cache(maxsize=64)
def _get_bound_reason_model(role: str, excluded_sources: tuple[str, ...]):
    """Cache immutable tool bindings; prompts remain state-dependent."""
    from src.agents.tools import get_tools_for_role

    tools = get_tools_for_role(role, excluded_retrieval_sources=set(excluded_sources))
    return llm.bind_tools(tools)


def prepare_reason(state: AssistantState) -> dict[str, Any]:
    """Start one verification-aware ReAct attempt."""
    messages = list(state.get(STATE_MESSAGES, []))
    reason_attempt = state.get(STATE_REASON_ATTEMPTS, 0) + 1
    query = state.get(STATE_RESOLVED_QUERY) or state[STATE_ORIGINAL_QUERY]
    issues = state.get(STATE_VERIFICATION, {}).get("issues", [])
    if reason_attempt > 1:
        issue_text = "\n".join(f"- {issue}" for issue in issues) or "- 上次回答未通过验证"
        user_content = (
            f"原问题：{query}\n\n上次回答未通过验证：\n{issue_text}\n\n请只依据可验证证据修正回答。"
        )
    else:
        user_content = query

    message_start = len(messages)
    return {
        STATE_MESSAGES: [HumanMessage(content=user_content)],
        STATE_REASON_ATTEMPTS: reason_attempt,
        STATE_TOOL_ITERATIONS: 0,
        STATE_REASON_MESSAGE_START: message_start,
        STATE_TOOL_MESSAGE_CURSOR: message_start,
        STATE_REASON_STARTED_PERF_COUNTER: time.perf_counter(),
        STATE_FINAL_ANSWER: "",
    }


def call_reason_model(state: AssistantState) -> dict[str, Any]:
    """Invoke the cached role-aware model binding for the current ReAct attempt."""
    messages = list(state.get(STATE_MESSAGES, []))
    start = min(max(state.get(STATE_REASON_MESSAGE_START, 0), 0), len(messages))
    attempt_messages = messages[start:]
    if not attempt_messages:
        raise ValueError("ReAct 推理缺少当前尝试的输入消息")

    role = state.get(STATE_USER_ROLE, ROLE_OPERATIONS)
    bound_model = _get_bound_reason_model(role, _excluded_retrieval_sources(state))
    response = bound_model.invoke([
        SystemMessage(content=_build_reason_system_prompt(state)),
        *attempt_messages,
    ])
    if not isinstance(response, AIMessage):
        raise TypeError("ReAct 模型必须返回 AIMessage")
    return {STATE_MESSAGES: [response]}


def authorize_reason_tool_call(
    request: ToolCallRequest,
    execute: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
) -> ToolMessage | Command[Any]:
    """Enforce role and source permissions again at the tool execution boundary."""
    state = cast(AssistantState, request.state)
    allowed_names = {tool.name for tool in _reason_tools(state)}
    tool_name = request.tool_call["name"]
    if tool_name not in allowed_names:
        return ToolMessage(
            content="当前角色或检索计划无权调用该工具。",
            name=tool_name,
            tool_call_id=request.tool_call["id"],
            status="error",
        )
    return execute(request)


def record_tool_results(state: AssistantState) -> dict[str, Any]:
    """Append only ToolMessages produced since the previous tool audit cursor."""
    messages = list(state.get(STATE_MESSAGES, []))
    cursor = min(max(state.get(STATE_TOOL_MESSAGE_CURSOR, 0), 0), len(messages))
    tool_calls = list(state.get(STATE_TOOL_CALLS, []))
    for message in messages[cursor:]:
        if isinstance(message, ToolMessage):
            output = message.content if isinstance(message.content, str) else str(message.content)
            tool_calls.append(
                ToolCallDict(
                    tool=message.name or "unknown",
                    output=output,
                    success=message.status != "error",
                )
            )
    return {
        STATE_TOOL_CALLS: tool_calls,
        STATE_TOOL_ITERATIONS: state.get(STATE_TOOL_ITERATIONS, 0) + 1,
        STATE_TOOL_MESSAGE_CURSOR: len(messages),
    }


def route_reason_model(state: AssistantState) -> Literal["tools", "finalize", "limit"]:
    """Route the ReAct loop after a model response."""
    messages = list(state.get(STATE_MESSAGES, []))
    if not messages or not isinstance(messages[-1], AIMessage):
        raise ValueError("ReAct 路由缺少 AIMessage")
    if not messages[-1].tool_calls:
        return "finalize"
    if state.get(STATE_TOOL_ITERATIONS, 0) >= MAX_TOOL_ITERATIONS:
        return "limit"
    return "tools"


def _reason_trace(state: AssistantState, *, success: bool = True) -> list[IntermediateStep]:
    started = state.get(STATE_REASON_STARTED_PERF_COUNTER, time.perf_counter())
    step = IntermediateStep(
        step="reason",
        duration_ms=max((time.perf_counter() - started) * 1000, 0.0),
        success=success,
    )
    return list(state.get(STATE_INTERMEDIATE_STEPS, [])) + [step]


def finalize_reason(state: AssistantState) -> dict[str, Any]:
    """Store the final model response and close the current ReAct attempt."""
    messages = list(state.get(STATE_MESSAGES, []))
    if not messages or not isinstance(messages[-1], AIMessage) or messages[-1].tool_calls:
        raise ValueError("ReAct 结束时必须存在无未决工具调用的 AIMessage")
    content = messages[-1].content
    final_content = _structure_answer(content if isinstance(content, str) else str(content))
    return {
        STATE_FINAL_ANSWER: final_content,
        STATE_INTERMEDIATE_STEPS: _reason_trace(state),
    }


def tool_limit_response(state: AssistantState) -> dict[str, Any]:
    """Fail closed when the model exceeds the per-attempt tool loop limit."""
    messages = list(state.get(STATE_MESSAGES, []))
    last_message = messages[-1] if messages else None
    pending_calls = last_message.tool_calls if isinstance(last_message, AIMessage) else []
    answer = _structure_answer("工具调用次数达到上限，无法安全完成当前请求。")
    limit_tool_messages = [
        ToolMessage(
            content="工具调用次数达到上限，已停止执行。",
            name=tool_call["name"],
            tool_call_id=tool_call["id"],
            status="error",
        )
        for tool_call in pending_calls
    ]
    limit_messages: list[BaseMessage] = list(limit_tool_messages)
    limit_messages.append(AIMessage(content=answer))
    tool_calls = list(state.get(STATE_TOOL_CALLS, []))
    tool_calls.extend(
        ToolCallDict(
            tool=message.name or "unknown",
            output=str(message.content),
            success=False,
        )
        for message in limit_tool_messages
    )
    return {
        STATE_MESSAGES: limit_messages,
        STATE_TOOL_CALLS: tool_calls,
        STATE_TOOL_MESSAGE_CURSOR: len(messages) + len(limit_messages),
        STATE_FINAL_ANSWER: answer,
        STATE_INTERMEDIATE_STEPS: _reason_trace(state, success=False),
    }


# ══════════════════════════════════════════════════════════════════════
# 4.7 Citation Extraction and Verification
# ══════════════════════════════════════════════════════════════════════


def extract_citations(state: AssistantState) -> dict[str, Any]:
    """Extract citations only from this turn's usable retrieval results."""
    # 用“消解/重写后的查询”做引用匹配，比原始查询更贴近本轮实际意图
    query = (
        state.get(STATE_RESOLVED_QUERY)
        or state.get(STATE_REWRITTEN_QUERY)
        or state.get(STATE_ORIGINAL_QUERY, "")
    )
    citations = _CITATION_EXTRACTOR.extract(
        state.get(STATE_RETRIEVAL_RESULTS, []),
        query=query,
    )
    return {STATE_CITATIONS: citations}


def verify(state: AssistantState) -> dict[str, Any]:
    """Run source, number, consistency, and hallucination verification."""
    verification = _VERIFIER.verify(
        answer=state.get(STATE_FINAL_ANSWER, ""),
        citations=state.get(STATE_CITATIONS, []),
        retrieval_results=state.get(STATE_RETRIEVAL_RESULTS, []),
        tool_calls=state.get(STATE_TOOL_CALLS, []),
    )

    role = state.get(STATE_USER_ROLE)
    issues = list(verification.get("issues", []))
    attributed_target_price = _has_attributed_target_price(state)
    # 投顾/销售岗额外拦截业务建议关键词，防止越权输出
    if role in (ROLE_ADVISOR, ROLE_INSTITUTIONAL_SALES):
        for pattern in _ADVICE_KEYWORDS:
            if pattern == TARGET_PRICE_PATTERN and attributed_target_price:
                continue
            if pattern in state.get(STATE_FINAL_ANSWER, ""):
                issues.append(f"投顾/销售角色不得输出业务建议: {pattern}")
    if issues:
        verification.update(passed=False, issues=issues, confidence=CONFIDENCE_LOW)
    return {STATE_VERIFICATION: verification}


# ══════════════════════════════════════════════════════════════════════
# 4.8 Compliance Check — 敏感词检测、业务建议拦截、风险提示
# ══════════════════════════════════════════════════════════════════════


def _has_attributed_target_price(state: AssistantState) -> bool:
    answer = state.get(STATE_FINAL_ANSWER, "")
    if TARGET_PRICE_PATTERN not in answer or not re.search(r"\[来源\d+\]", answer):
        return False
    return any(
        TARGET_PRICE_PATTERN in result.get(RR_CONTENT, "")
        for result in state.get(STATE_RETRIEVAL_RESULTS, [])
        if not result.get(RR_DENIED)
    )


def compliance_check(state: AssistantState) -> dict[str, Any]:
    """合规检查：敏感词、业务建议、风险提示、适当性"""
    # 最终答案 + 角色 + 客户号一起送入，便于做适当性和角色化拦截
    answer = state.get(STATE_FINAL_ANSWER, "")
    compliance = _COMPLIANCE_CHECKER.check(
        answer,
        user_role=state.get(STATE_USER_ROLE),
        client_id=state.get(STATE_CLIENT_ID),
        allow_attributed_target_price=(
            state.get(STATE_VERIFICATION, {}).get("passed", False)
            and _has_attributed_target_price(state)
        ),
    )

    return {STATE_COMPLIANCE: compliance}


def permission_denied_response(state: AssistantState) -> dict[str, Any]:
    """Terminate before LLM reasoning when every retrieval result is denied."""
    # 在进入 LLM 推理前短路返回，既省成本，也避免把无权限内容继续带入后续节点
    return {
        STATE_FINAL_ANSWER: _structure_answer("当前角色无权限访问完成该请求所需的数据源。"),
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
    }


# ══════════════════════════════════════════════════════════════════════
# 4.9 Compose — 引用标注、置信度、风险提示
# ══════════════════════════════════════════════════════════════════════


def compose(state: AssistantState) -> dict[str, Any]:
    """生成最终回答：带引用、置信度、风险提示"""
    answer = state.get(STATE_FINAL_ANSWER, "")
    citations = state.get(STATE_CITATIONS, [])

    # 风险提示 + 适当性警告
    compliance = state.get(STATE_COMPLIANCE, {})
    risk = compliance.get("risk_disclosure", "")
    suitability = compliance.get("suitability_warning", "")
    verification_passed = state.get(STATE_VERIFICATION, {}).get("passed", False)
    compliance_passed = compliance.get("passed", False)
    # 验证不通过时，直接替换为安全提示，并清空引用，避免继续传播不可靠答案
    if not verification_passed:
        answer = "当前答案未通过来源或数字验证，无法安全返回。请补充可验证资料后重试。"
        citations = []
    # 合规不通过时，停止输出，但保留风险提示/适当性警告作为最终兜底说明
    elif not compliance_passed:
        answer = "当前请求或生成内容未通过合规检查，已停止输出。"
        citations = []
    final_answer = _structure_answer(answer) + suitability + risk

    # 综合置信度：合规失败直接低；高置信 + 至少 3 条有效结果才算高
    verification_conf = state.get(STATE_VERIFICATION, {}).get("confidence", CONFIDENCE_MEDIUM)
    result_count = len([
        result for result in state.get(STATE_RETRIEVAL_RESULTS, []) if not result.get(RR_DENIED)
    ])

    if not verification_passed or not compliance_passed:
        confidence = CONFIDENCE_LOW
    elif verification_conf == CONFIDENCE_HIGH and result_count >= 3:
        confidence = CONFIDENCE_HIGH
    else:
        confidence = CONFIDENCE_MEDIUM

    return {
        STATE_FINAL_ANSWER: final_answer,
        STATE_CITATIONS: citations,
        STATE_CONFIDENCE: confidence,
        STATE_RISK_DISCLOSURE: risk + suitability,
    }


def persist_conversation_turn(state: AssistantState) -> dict[str, Any]:
    """Persist the visible turn and durable audit outbox event atomically."""
    # 将本轮可见对话写入 SQLite；返回空更新，不改变当前 state
    _get_conversation_store().insert_turn(state)
    return {}


# ══════════════════════════════════════════════════════════════════════
# 4.10 Audit Log — 全链路追踪记录
# ══════════════════════════════════════════════════════════════════════


def audit_log(state: AssistantState) -> dict[str, Any]:
    """记录追踪日志——覆盖 Query → Retrieve → Reason → Verify → Compose 全链路

    实现委托给 src/utils/audit.py 的 AuditLogger（AuditEntry 模型的权威构建者），
    避免与该模块重复维护同一份字段拼装逻辑。
    """
    # AuditLogger 负责把整条 state 组装成结构化审计事件
    from src.utils.audit import AuditLogger, audit_entry_to_trail

    audit_entry = AuditLogger().log(state)
    conversation_store = _get_conversation_store()
    try:
        # 先写入审计库；若存在对话 outbox，则同步标记为已处理
        _get_audit_store().insert(audit_entry)
        if conversation_store.get_outbox_status(audit_entry.request_id) is not None:
            conversation_store.mark_outbox_processed(audit_entry.request_id)
    except Exception as exc:
        # 审计写入失败时，至少把 outbox 标成失败，避免丢失问题线索
        if conversation_store.get_outbox_status(audit_entry.request_id) is not None:
            conversation_store.mark_outbox_failed(audit_entry.request_id, str(exc))
        raise

    return {STATE_AUDIT_TRAIL: audit_entry_to_trail(audit_entry)}


# ══════════════════════════════════════════════════════════════════════
# Data Flow
# ══════════════════════════════════════════════════════════════════════

#
# User Query
#   -> load_conversation_context
#   -> resolve_followup_query
#   -> query_understand
#   -> planner
#   -> retrieve
#   -> grade_and_filter
#   -> reason (may call tools)
#   -> extract_citations
#   -> verify
#   -> compliance_check
#   -> compose
#   -> persist_conversation_turn
#   -> audit_log
#   -> Final Answer
