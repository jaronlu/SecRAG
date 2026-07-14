# SecRAG — 面试问答录（问题 → 源码 → 解读）

> 使用方式：每个问题先自己回答，再对照"源码定位"读代码，最后看"解读"校准表述。

---

## Q1：你的 Agent 是怎么工作的？整体流程是什么？

### 源码定位

`src/agents/graph.py` `build_agent_graph()` 第 104-189 行

### 关键代码

```python
graph = StateGraph(AssistantState)

graph.add_node("load_conversation_context", ...)
graph.add_node("resolve_followup_query", ...)
graph.add_node("query_understand", ...)
graph.add_node("planner", ...)
graph.add_node("retrieve", ...)
graph.add_node("grade_and_filter", ...)
graph.add_node("permission_denied_response", ...)
graph.add_node("reason", ...)
graph.add_node("extract_citations", ...)
graph.add_node("verify", ...)
graph.add_node("compliance_check", ...)
graph.add_node("compose", ...)
graph.add_node("persist_conversation_turn", ...)
graph.add_node("audit_log", ...)

graph.add_edge(START, "load_conversation_context")
graph.add_edge("load_conversation_context", "resolve_followup_query")
graph.add_edge("resolve_followup_query", "query_understand")
graph.add_edge("query_understand", "planner")
graph.add_edge("planner", "retrieve")
graph.add_edge("retrieve", "grade_and_filter")
# 条件路由：grade_and_filter → retrieve / permission_denied_response / reason
graph.add_conditional_edges("grade_and_filter", should_retry_retrieval, {...})
graph.add_edge("permission_denied_response", "persist_conversation_turn")
graph.add_edge("reason", "extract_citations")
graph.add_edge("extract_citations", "verify")
# 条件路由：verify → retry / continue
graph.add_conditional_edges("verify", should_reason_again, {...})
# 条件路由：compliance_check → pass / block
graph.add_conditional_edges("compliance_check", is_compliant, {...})
graph.add_edge("compose", "persist_conversation_turn")
graph.add_edge("persist_conversation_turn", "audit_log")
graph.add_edge("audit_log", END)
```

### 解读

整个流程分三个阶段：

**阶段一：查询理解**（4 个节点）
- `load_conversation_context`：从 SQLite 加载当前用户的会话历史和实体摘要
- `resolve_followup_query`：如果用户说"它"、"这个"，补全上下文
- `query_understand`：LLM 做结构化分析，输出意图、实体、重写后的查询
- `planner`：根据角色权限生成多源检索计划

**阶段二：检索+推理**（3 个节点 + 条件路由）
- `retrieve`：按计划执行检索
- `grade_and_filter`：评分、去重、保留 top-K
- 条件判断：结果不足就重新检索，全部被拒就走权限拒绝，否则进入推理

**阶段三：验证+回答**（5 个节点 + 条件路由）
- `reason`：ReAct Agent 推理+工具调用
- `extract_citations`：提取引用
- `verify`：校验来源和数字
- `compliance_check`：合规检查
- `compose`：生成最终回答

---

## Q2：你为什么用 LangGraph 而不是 LangChain Chain？

### 源码定位

`src/agents/graph.py` 第 104-189 行（条件路由部分）

### 关键代码

```python
# 检索不足时重新检索
graph.add_conditional_edges(
    "grade_and_filter",
    should_retry_retrieval,
    {
        "continue": "reason",
        "retrieve": "planner",
        "denied": "permission_denied_response",
    },
)

# 验证失败时重新推理
graph.add_conditional_edges(
    "verify",
    should_reason_again,
    {
        "continue": "compliance_check",
        "retry": "reason",
    },
)

# 合规拦截
graph.add_conditional_edges(
    "compliance_check",
    is_compliant,
    {
        "pass": "compose",
        "block": "compose",
    },
)
```

### 解读

有三个地方需要动态决策：

1. **检索不足要重试**：`should_retry_retrieval` 判断结果数量和质量，不够就回到 planner 重新规划，最多 `DEFAULT_MAX_HOPS` 次
2. **验证失败要重推**：`should_reason_again` 判断 verify 是否通过，失败且未超限就回到 reason 重新推理，最多 `MAX_REASON_ATTEMPTS` 次
3. **合规拦截要返回**：`is_compliant` 判断合规检查，block 也走 compose，但附加拦截提示

如果用固定链，这些分支逻辑只能塞进节点内部，状态流转不清晰，调试困难。LangGraph 的条件路由把决策逻辑集中在一个函数里，图结构清晰表达"什么条件下走哪条路"。

---

## Q3：你的检索是怎么做的？为什么不用一次检索就回答？

### 源码定位

`src/agents/nodes.py` `planner()` 第 265-326 行
`src/agents/nodes.py` `retrieve()` 第 334-359 行
`src/retrieval/hybrid_retriever.py` `retrieve()` 第 51-79 行

### 关键代码

```python
# planner：生成多源检索计划
def planner(state: AssistantState) -> AssistantState:
    allowed_sources = ROLE_ALLOWED_SOURCES.get(state[STATE_USER_ROLE], [])
    prompt = f"""根据以下查询理解结果，生成检索计划：
    ...
    可用数据源（基于角色权限）：
    - product_search: 理财产品说明书、产品合同、风险揭示书
    - regulation_search: 规则法规、内部制度、处罚案例
    - report_search: 研报摘要、晨会纪要、策略周报
    - faq_search: 常见问题解答、操作流程
    ...
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    # 按角色权限过滤计划
    for raw_step in parsed_plan:
        if step.get(PLAN_SOURCE) in allowed_sources:
            filtered_plan.append(step)
```

```python
# retrieve：按计划执行检索
def retrieve(state: AssistantState) -> AssistantState:
    retriever = HybridRetriever(
        user_role=state[STATE_USER_ROLE],
        data_permissions=state.get(STATE_DATA_PERMISSIONS, [PERMISSION_PUBLIC]),
    )
    results = retriever.retrieve(plan=normalized_plan)
    accumulated = state.get(STATE_RETRIEVAL_RESULTS, []) + results
    return _with_state_updates(state, {
        STATE_RETRIEVAL_RESULTS: accumulated,
        STATE_RETRIEVAL_ATTEMPTS: state.get(STATE_RETRIEVAL_ATTEMPTS, 0) + 1,
    })
```

```python
# HybridRetriever：按计划执行多源检索
def retrieve(self, plan: list[RetrievalPlanStep]) -> list[RetrievalResult]:
    for step in self._filter_plan_by_role(plan):
        if step.get(PLAN_DENIED):
            results.append(self._denied_result(step))
            continue
        source = step.get(PLAN_SOURCE)
        retriever = self._get_retriever(source)
        retrieved = retriever.retrieve(
            query=step.get(PLAN_QUERY, ""),
            top_k=step.get(PLAN_TOP_K, DEFAULT_TOP_K),
            filters=step.get(PLAN_FILTERS),
        )
        results.extend(self._filter_results_by_role(retrieved))
    return results
```

### 解读

检索分两步，不是一步：

**第一步：planner 生成计划**
- 把用户的模糊问题（比如"这个产品怎么样"）重写成结构化查询
- 根据用户角色决定可以访问哪些数据源
- 输出检索计划：每个计划包含 source、query、top_k、filters

**第二步：retrieve 按计划执行**
- `HybridRetriever` 遍历检索计划
- 每个 source 有独立的检索器（ProductRetriever / RegulationRetriever / ReportRetriever / FAQRetriever）
- 执行前再次过滤权限（`_filter_plan_by_role`）
- 检索结果再次过滤 chunk 级权限（`_filter_results_by_role`）

**为什么不用一次检索**：金融知识分散在多个数据源（研报、公告、法规、产品说明书），同一个问题可能需要跨多个源检索。而且不同角色看到的源不一样，必须先确定"能看什么"，再决定"去哪检索"。

---

## Q4：你怎么处理权限？怎么防止越权访问？

### 源码定位

`src/retrieval/hybrid_retriever.py` `_filter_plan_by_role()` 第 81-100 行
`src/retrieval/hybrid_retriever.py` `_filter_results_by_role()` 第 116-157 行
`src/agents/nodes.py` `planner()` 第 310-322 行

### 关键代码

```python
# 第一层：planner 节点过滤检索计划
def planner(state: AssistantState) -> AssistantState:
    allowed_sources = ROLE_ALLOWED_SOURCES.get(state[STATE_USER_ROLE], [])
    for raw_step in parsed_plan:
        step = _normalize_plan_step(raw_step, state[STATE_REWRITTEN_QUERY])
        if step is not None and step.get(PLAN_SOURCE) in allowed_sources:
            filtered_plan.append(step)
```

```python
# 第二层：HybridRetriever 过滤计划
def _filter_plan_by_role(self, plan: list[RetrievalPlanStep]) -> list[RetrievalPlanStep]:
    for step in plan:
        source = step.get(PLAN_SOURCE)
        if source in self.allowed_sources:
            filtered.append(step)
        else:
            filtered.append(
                RetrievalPlanStep(
                    source=source or "",
                    query=step.get(PLAN_QUERY, ""),
                    top_k=step.get(PLAN_TOP_K, 0),
                    denied=True,
                    reason=f"角色 {self.user_role} 无权限访问 {source}",
                )
            )
    return filtered
```

```python
# 第三层：chunk 级权限过滤
def _filter_results_by_role(self, results: list[RetrievalResult]) -> list[RetrievalResult]:
    for result in results:
        metadata = result.get(RR_METADATA, {})
        permission_level = metadata.get(META_PERMISSION_LEVEL, PERMISSION_PUBLIC)
        if permission_level not in self.data_permissions:
            filtered.append(self._permission_denied_result(...))
            continue
        allowed_roles = metadata.get(META_ALLOWED_ROLES)
        if not allowed_roles:
            if permission_level == PERMISSION_PUBLIC:
                filtered.append(result)
            else:
                filtered.append(self._permission_denied_result(...))
            continue
        if self.user_role in allowed:
            filtered.append(result)
        else:
            filtered.append(self._permission_denied_result(...))
    return filtered
```

### 解读

三层过滤，不是一层：

**第一层：planner 节点**
- 检索计划生成后，只保留当前角色允许的 source
- 即使 LLM 偷偷生成越权 source，也会被过滤掉
- 这是"预防性"过滤，从源头控制

**第二层：HybridRetriever**
- 执行检索前再次检查计划
- 越权 source 转为显式 `denied` 结果，不会执行检索
- 这是"执行前"过滤，双重保险

**第三层：chunk 级过滤**
- 检索结果返回后，检查每个 chunk 的 `permission_level` 和 `allowed_roles`
- 非公开数据必须有 `allowed_roles`，否则默认拒绝
- 这是"结果级"过滤，防止数据源配置错误导致泄露

**面试时要强调**：很多 RAG 系统是先检索再过滤，这是不安全的，因为检索过程本身就可能暴露敏感内容。我的做法是从计划生成阶段就限制，执行层再次验证，结果层再次过滤。

---

## Q5：你的 ReAct Agent 是怎么实现的？为什么嵌套在 LangGraph 里？

### 源码定位

`src/agents/nodes.py` `reason()` 第 402-505 行
`src/agents/tools.py` 第 1-135 行

### 关键代码

```python
# reason 节点：内嵌 LangChain ReAct Agent
def reason(state: AssistantState) -> AssistantState:
    # 构建上下文（检索结果前 5 条）
    context_parts = []
    for index, result in enumerate(results[:5]):
        context_parts.append(f"[来源{index + 1}] {metadata.get(META_TITLE, '未知')}\n{result[RR_CONTENT]}")
    context = "\n\n".join(context_parts)

    # 角色化 system prompt
    role_instructions = {
        ROLE_ADVISOR: "你是机构投顾助手。基于内部知识库为客户提供准确的产品/规则/市场信息。",
        ROLE_INSTITUTIONAL_SALES: "你是机构销售助手。为机构客户提供研究支持和市场洞察。",
        ...
    }

    # 创建 ReAct Agent
    agent = create_agent(
        model=llm,
        tools=get_tools_for_role(role, excluded_retrieval_sources=excluded_sources),
        system_prompt=system_prompt,
    )
    response = agent.invoke({
        "messages": [HumanMessage(content=state.get(STATE_RESOLVED_QUERY) or state[STATE_ORIGINAL_QUERY])]
    })

    # 提取工具调用记录
    tool_calls: list[ToolCallDict] = list(state.get(STATE_TOOL_CALLS, []))
    for message in response[STATE_MESSAGES]:
        if isinstance(message, ToolMessage):
            tool_calls.append(ToolCallDict(
                tool=message.name or "unknown",
                output=message.content,
                success=message.status != "error",
            ))
```

```python
# 工具定义：8 个工具按角色动态注册
tools = [
    product_search, regulation_search, report_search, faq_search,
    calculator, suitability_check, market_data_tool,
    sql_query_tool, financial_ratios_tool, rerank_tool,
]

def get_tools_for_role(user_role: str, excluded_retrieval_sources: set[str] | None = None):
    allowed_sources = set(ROLE_ALLOWED_SOURCES.get(user_role, [SOURCE_FAQ]))
    return [
        tool_item for tool_item in tools
        if (_RETRIEVAL_TOOL_SOURCES.get(tool_item.name) in allowed_sources
            and _RETRIEVAL_TOOL_SOURCES.get(tool_item.name) not in excluded_sources)
            or tool_item.name not in _RETRIEVAL_TOOL_SOURCES
    ]
```

### 解读

双层 Agent 架构：

**外层 LangGraph**：负责工作流编排——权限检查、检索、验证、合规。这些是领域逻辑，需要条件路由和状态持久化。

**内层 ReAct Agent**：负责工具调用。reason 节点需要 LLM 自主决定调用哪个工具、调用几次、怎么组合结果。LangChain 的 `create_agent` 已经实现了完整的 ReAct 循环（思考→调用工具→观察→再思考），直接复用比自己实现一遍更稳定。

**工具按角色动态注册**：不是所有用户都能调用所有工具。投顾可以调用产品检索和适当性检查，合规可以调用法规检索，技术可以调用 SQL 查询。`get_tools_for_role` 根据用户角色和已排除的检索源动态决定可见工具。

**技术债**：当前 `create_agent` 每次调用都重新编译 agent graph，性能有浪费。Phase 2 计划：把 ReAct 循环建模为 LangGraph 子图，这样工具调用状态也会被 checkpoint 持久化。

---

## Q6：你怎么处理检索结果不足的情况？

### 源码定位

`src/agents/graph.py` `should_retry_retrieval()` 第 65-79 行
`src/agents/nodes.py` `retrieve()` 第 334-359 行

### 关键代码

```python
# 条件路由：判断是否需要重新检索
def should_retry_retrieval(state: AssistantState) -> str:
    attempts = state.get(STATE_RETRIEVAL_ATTEMPTS, 0)
    results = state.get(STATE_RETRIEVAL_RESULTS, [])
    usable = [result for result in results if not result.get("denied")]

    if results and not usable:
        return "denied"                      # 全部被拒 → 权限拒绝
    if attempts >= DEFAULT_MAX_HOPS:
        return "continue"                    # 已达最大跳数 → 继续推理
    if not results or len(usable) < CONFIDENCE_HIGH_MIN_RESULTS:
        return "retrieve"                    # 结果不足 → 重新检索
    return "continue"                        # 结果充足 → 继续推理
```

```python
# retrieve 节点：累加结果，计数器+1
def retrieve(state: AssistantState) -> AssistantState:
    results = retriever.retrieve(plan=normalized_plan)
    accumulated = state.get(STATE_RETRIEVAL_RESULTS, []) + results
    return _with_state_updates(state, {
        STATE_RETRIEVAL_RESULTS: accumulated,
        STATE_RETRIEVAL_ATTEMPTS: state.get(STATE_RETRIEVAL_ATTEMPTS, 0) + 1,
    })
```

### 解读

不是一次检索就放弃，而是多跳检索：

**判断逻辑**（`should_retry_retrieval`）：
1. 如果有结果但全部被 `denied` → 走权限拒绝分支，直接返回"无权限"
2. 如果检索次数已达上限 → 停止重试，用现有结果继续推理
3. 如果没有结果，或可用结果 < 高置信度阈值 → 回到 planner 重新规划检索路径
4. 否则 → 结果充足，继续推理

**累加机制**：`retrieve` 节点每次执行都累加到 `retrieval_results`，不会覆盖上一轮的结果。这样多跳检索的结果是累积的。

**最大跳数**：由 `DEFAULT_MAX_HOPS` 控制，防止无限循环。

---

## Q7：你怎么处理验证失败？验证通过就一定可信吗？

### 源码定位

`src/agents/nodes.py` `verify()` 第 527-544 行
`src/agents/nodes.py` `compose()` 第 596-635 行
`src/agents/graph.py` `should_reason_again()` 第 82-88 行

### 关键代码

```python
# verify 节点：多维度校验
def verify(state: AssistantState) -> AssistantState:
    verification = _VERIFIER.verify(
        answer=state.get(STATE_FINAL_ANSWER, ""),
        citations=state.get(STATE_CITATIONS, []),
        retrieval_results=state.get(STATE_RETRIEVAL_RESULTS, []),
        tool_calls=state.get(STATE_TOOL_CALLS, []),
    )
    # 投顾/销售额外检测业务建议关键词
    role = state.get(STATE_USER_ROLE)
    issues = list(verification.get("issues", []))
    if role in (ROLE_ADVISOR, ROLE_INSTITUTIONAL_SALES):
        for pattern in _ADVICE_KEYWORDS:
            if pattern in state.get(STATE_FINAL_ANSWER, ""):
                issues.append(f"投顾/销售角色不得输出业务建议: {pattern}")
    if issues:
        verification.update(passed=False, issues=issues, confidence=CONFIDENCE_LOW)
    return _with_state_updates(state, {STATE_VERIFICATION: verification})
```

```python
# compose 节点：验证不通过时拒绝回答
def compose(state: AssistantState) -> AssistantState:
    verification_passed = state.get(STATE_VERIFICATION, {}).get("passed", False)
    compliance_passed = state.get(STATE_COMPLIANCE, {}).get("passed", False)
    if not verification_passed:
        answer = "当前答案未通过来源或数字验证，无法安全返回。请补充可验证资料后重试。"
        citations = []
    elif not compliance_passed:
        answer = "当前请求或生成内容未通过合规检查，已停止输出。"
        citations = []
    # 综合置信度
    if not compliance_passed:
        confidence = CONFIDENCE_LOW
    elif verification_conf == CONFIDENCE_HIGH and result_count >= 3:
        confidence = CONFIDENCE_HIGH
    else:
        confidence = CONFIDENCE_MEDIUM
```

```python
# 条件路由：验证失败时重新推理
def should_reason_again(state: AssistantState) -> str:
    verification = state.get(STATE_VERIFICATION, {})
    attempts = state.get(STATE_REASON_ATTEMPTS, 0)
    if not verification.get("passed", False) and attempts < MAX_REASON_ATTEMPTS:
        return "retry"
    return "continue"
```

### 解读

验证不是一次性的，有重试机制：

**verify 节点做什么**：
- 来源校验：引用是否在检索结果中存在
- 数字校验：回答中的数字是否与检索结果一致
- 一致性校验：回答是否与上下文矛盾
- 幻觉检测：是否有编造内容的迹象
- 业务建议检测：投顾/销售角色是否输出了"买入"/"卖出"/"目标价"等建议

**验证失败的处理**：
- `should_reason_again` 判断：如果验证失败且未超限，回到 reason 重新推理
- `compose` 节点：如果验证最终不通过，直接拒绝回答，返回"未通过验证"的提示

**验证通过也不一定可信**：
- 置信度分三档：HIGH / MEDIUM / LOW
- HIGH 需要：验证通过 + 至少 3 条可用结果 + 验证置信度为 HIGH
- 否则是 MEDIUM
- 合规不通过直接 LOW

**面试时要诚实说**：验证本身依赖 LLM，不是 100% 可靠。我的策略是"fail-closed"——不确定就不要返回，宁可拒绝回答，也不返回不可信内容。

---

## Q8：你的合规检查是怎么做的？和验证有什么区别？

### 源码定位

`src/agents/nodes.py` `compliance_check()` 第 552-563 行
`src/utils/compliance.py` `ComplianceChecker`
`src/agents/nodes.py` `compose()` 第 596-635 行

### 关键代码

```python
# compliance_check 节点
def compliance_check(state: AssistantState) -> AssistantState:
    answer = state.get(STATE_FINAL_ANSWER, "")
    compliance = _COMPLIANCE_CHECKER.check(
        answer,
        user_role=state.get(STATE_USER_ROLE),
        client_id=state.get(STATE_CLIENT_ID),
    )
    return _with_state_updates(state, {STATE_COMPLIANCE: compliance})
```

```python
# compose 节点：验证和合规分开处理
def compose(state: AssistantState) -> AssistantState:
    verification_passed = state.get(STATE_VERIFICATION, {}).get("passed", False)
    compliance_passed = state.get(STATE_COMPLIANCE, {}).get("passed", False)
    if not verification_passed:
        answer = "当前答案未通过来源或数字验证，无法安全返回。..."
    elif not compliance_passed:
        answer = "当前请求或生成内容未通过合规检查，已停止输出。..."
    # 合规通过但验证不通过 → 返回拒绝
    # 合规不通过但验证通过 → 返回答案 + 拦截提示
    # 两者都通过 → 返回答案 + 引用 + 置信度
```

### 解读

**验证和合规是两个独立的检查**：

| 维度 | 验证（verify） | 合规（compliance_check） |
|------|---------------|------------------------|
| 检查对象 | 回答的内容准确性 | 回答的业务合规性 |
| 检查内容 | 引用是否存在、数字是否一致、是否有幻觉 | 敏感词、业务建议、风险提示、适当性 |
| 不通过后果 | 拒绝回答 | 返回答案但附加拦截提示 |
| 重试机制 | 验证失败可重试（回到 reason） | 合规不通过不重试，直接返回 |

**为什么分开**：验证是"内容对不对"，合规是"能不能说"。验证失败说明检索或推理有问题，需要重新推理；合规失败说明内容本身有问题，重新推理也没用，直接拦截。

**面试时要强调**：这是"fail-closed"设计——验证不通过直接拒绝，合规不通过也停止输出，不会把不可信或不合规的内容返回给用户。

---

## Q9：你的系统怎么处理多轮对话？上下文是怎么管理的？

### 源码定位

`src/agents/nodes.py` `load_conversation_context()` 第 185-197 行
`src/agents/nodes.py` `resolve_followup_query()` 第 200-208 行
`src/agents/state.py` 第 36-38 行

### 关键代码

```python
# 加载会话上下文
def load_conversation_context(state: AssistantState) -> AssistantState:
    history, summary = _get_conversation_store().load_context(
        thread_id=state[STATE_THREAD_ID],
        user_id=state[STATE_USER_ID],
    )
    return _with_state_updates(state, {
        STATE_CHAT_HISTORY: history,
        STATE_CONVERSATION_SUMMARY: summary,
    })
```

```python
# 消解指代式追问
def resolve_followup_query(state: AssistantState) -> AssistantState:
    query = state[STATE_ORIGINAL_QUERY]
    summary = state.get(STATE_CONVERSATION_SUMMARY, "")
    followup_markers = ("它", "这个", "该产品", "该公司", "那", "上述", "前面")
    resolved = f"基于会话实体（{summary}），{query}" if summary and any(
        marker in query for marker in followup_markers
    ) else query
    return _with_state_updates(state, {STATE_RESOLVED_QUERY: resolved})
```

```python
# 状态定义
class AssistantState(TypedDict):
    chat_history: list[ConversationMessageDict]        # 历史消息（SQLite 持久化）
    conversation_summary: str                          # 实体摘要（SQLite 持久化）
    resolved_query: str                                # 消解后的查询
```

### 解读

多轮对话管理分三步：

**第一步：加载上下文**
- `load_conversation_context` 从 SQLite 加载当前线程的历史消息和实体摘要
- 历史消息用于 ReAct Agent 的上下文
- 实体摘要用于消解指代

**第二步：消解指代**
- `resolve_followup_query` 检测用户输入是否包含指代词（"它"、"这个"、"上述"）
- 如果包含，把实体摘要拼接到查询前面，形成完整的查询
- 例如：用户问"它年化收益率多少？"，系统补全为"基于会话实体（产品A：开放式净值型理财产品），它年化收益率多少？"

**第三步：状态持久化**
- `persist_conversation_turn` 节点在每轮对话结束后把可见消息写入 SQLite
- 下次对话时从 SQLite 加载，不是存在内存里

**面试时要提**：这里用的是 SQLite 持久化，不是 LangGraph Checkpointer。原因是需要跨 session 持久化，而 Checkpointer 默认是内存级的。

---

## Q10：你的状态为什么这么多字段？怎么管理的？

### 源码定位

`src/agents/state.py` `AssistantState` 第 24-76 行

### 关键代码

```python
class AssistantState(TypedDict):
    # 用户上下文
    user_id: str
    user_role: str
    department: str
    data_permissions: list[str]
    client_id: Optional[str]
    thread_id: str
    turn_id: str
    turn_index: int

    # 会话上下文
    chat_history: list[ConversationMessageDict]
    conversation_summary: str
    resolved_query: str

    # 查询理解
    original_query: str
    rewritten_query: str
    intent: str
    entities: QueryEntities
    ambiguity: list[str]
    query_type: str

    # 检索计划
    retrieval_plan: list[RetrievalPlanStep]
    retrieval_attempts: int

    # 检索结果
    retrieval_results: Annotated[list[RetrievalResult], "concatenate"]
    retrieval_total_chunks: int
    retrieval_filtered_chunks: int

    # 推理过程
    messages: Annotated[Sequence[BaseMessage], add_messages]
    tool_calls: list[ToolCallDict]
    intermediate_steps: list[IntermediateStep]
    reason_attempts: int

    # 验证/合规
    verification: VerificationResult
    compliance: ComplianceResult

    # 最终回答
    final_answer: str
    citations: list[CitationDict]
    confidence: str
    risk_disclosure: str

    # 追踪
    audit_trail: AuditTrail
```

### 解读

30+ 字段，分六个组：

**用户上下文**（7 个字段）：身份信息，来自 token，不信任请求体
**会话上下文**（3 个字段）：线程和摘要，SQLite 持久化
**查询理解**（6 个字段）：意图/实体/重写，LLM 生成
**检索计划**（2 个字段）：计划列表 + 多跳计数器
**检索结果**（3 个字段）：结果列表用 `concatenate` reducer 多轮累加
**推理过程**（4 个字段）：消息历史用 `add_messages` reducer 自动追加
**验证/合规**（2 个字段）：硬约束结果
**最终回答**（4 个字段）：对外返回的内容
**追踪**（1 个字段）：全链路审计日志

**管理方式**：
- 每个节点只返回自己负责的字段更新
- 通过 `_with_state_updates` 合并，避免状态漂移
- `add_messages` reducer：消息历史自动追加
- `concatenate` reducer：检索结果多轮累加

---

## Q11：你的 FastAPI 接口有什么特殊设计？

### 源码定位

`src/api/main.py` 第 231-291 行
`src/api/auth.py` `authenticate_user` / `build_assistant_initial_state`

### 关键代码

```python
# Agent 接口：身份绑定，不信任请求体
@app.post(API_ROUTE_ASSISTANT_QA, response_model=AssistantQAResponse)
async def assistant_qa(
    request: AssistantQARequest,
    user: AuthenticatedUser = Depends(authenticate_user),
):
    # 从 token 派生身份，不信任请求体里的角色参数
    initial_state = build_assistant_initial_state(
        request, user,
        thread_id=thread_id,
        turn_id=turn_id,
        turn_index=thread.get("turn_count", 0),
    )
    app = _get_agent_app()
    config: RunnableConfig = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": AGENT_RECURSION_LIMIT,
    }
    result = app.invoke(initial_state, config=config)
    return AssistantQAResponse(...)
```

```python
# 错误处理：LLM 不可用时返回 503
except Exception as exc:
    if _is_provider_unavailable(exc):
        raise HTTPException(
            status_code=503,
            detail="LLM provider unavailable. Check OPENAI_API_BASE, ..."
        )
    raise HTTPException(status_code=500, detail="内部处理错误")
```

### 解读

**身份绑定**：`authenticate_user` 从 token 派生用户身份和角色，不信任请求体里的任何身份参数。这防止了请求体伪造角色的攻击。

**懒加载**：`_get_agent_app()` 首次请求时才构建 Agent Graph，避免启动时建立 ChromaDB 连接。

**Checkpointer**：用 `configurable={"thread_id": thread_id}` 把请求和 LangGraph checkpoint 关联，实现多轮对话的状态恢复。

**错误分级**：LLM provider 不可用返回 503（服务不可用），其他错误返回 500。审计日志记录所有异常。

---

## Q12：你遇到过什么难点？怎么解决的？

### 推荐回答方向（按你的实际代码）

**难点 1：中文语义检索**
- 代码：`src/rag/indexer.py` `_get_embed_model()` 使用 `paraphrase-multilingual-MiniLM-L12-v2`
- 解决：选多语言 embedding 模型，同时支持中英文检索

**难点 2：对话历史超长**
- 代码：`src/chat/session.py`（RepoAgent）`_compress_history()` 或 SecRAG 的 SQLite 持久化 + 摘要
- 解决：超出 token 预算时用 LLM 生成摘要，保留最近的消息原文

**难点 3：非 Python 代码分析**
- 代码：`src/main.py` `_do_analyze()` 中 Python 走 Tree-sitter，其他语言走 LLM fallback
- 解决：精确解析和 LLM 分析结合，不是一刀切

**难点 4：权限绕过**
- 代码：`src/retrieval/hybrid_retriever.py` 双层过滤
- 解决：planner 层过滤计划 + retriever 层过滤执行 + chunk 级过滤，三层防护

---

## Q13：你项目最大的技术债是什么？怎么改进？

### 源码定位

`src/agents/nodes.py` `reason()` 第 402-410 行注释

### 关键代码

```python
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
```

### 解读

当前最大的技术债是 `reason` 节点的实现方式：

**问题**：每次调用都重新编译 ReAct Agent graph，性能浪费。而且嵌套的 Agent 状态和外层 LangGraph checkpoint 隔离，无法完整持久化工具调用过程。

**改进方案**：Phase 2 把 ReAct 循环建模为 LangGraph 子图：
```
reason → [tool_calls?] → tool_executor → reason
```
这样：
- Agent 的工具调用状态也会被 checkpoint 持久化
- 不需要每次重新编译
- 整个系统在统一的 StateGraph 范式内，调试和追踪更一致

**面试时要强调**：我不仅写了代码，还对技术债有清醒的认识和重构计划。

---

## Q14：如果让你重新设计这个系统，你会改什么？

### 推荐回答

1. **ReAct 循环改为 LangGraph 子图**：解决当前嵌套的技术债，状态管理更一致
2. **加入 KV Cache**：高频 query 的 LLM 响应可以缓存，减少 API 调用和延迟
3. **工具接口迁移到 MCP**：当前工具是硬编码的 LangChain `@tool`，换成 MCP 协议后可以动态发现和加载工具
4. **查询理解用结构化输出**：当前 `json.loads` 解析 LLM 返回，失败时 fallback 到默认值。应该用 LangChain 的 `with_structured_output` 强制 schema 验证
5. **检索效果评估常态化**：当前有 `scripts/evaluate_retrieval.py`，但样例数据太少，需要扩充到 20-50 条覆盖不同场景

---

## Q15：你的系统怎么处理幻觉？

### 源码定位

`src/agents/nodes.py` `extract_citations()` 第 513-524 行
`src/agents/nodes.py` `verify()` 第 527-544 行
`src/agents/nodes.py` `compose()` 第 596-635 行

### 关键代码

```python
# extract_citations：从检索结果提取引用
def extract_citations(state: AssistantState) -> AssistantState:
    citations = _CITATION_EXTRACTOR.extract(
        state.get(STATE_RETRIEVAL_RESULTS, []),
        query=query,
    )
    return _with_state_updates(state, {STATE_CITATIONS: citations})

# verify：校验引用和数字
def verify(state: AssistantState) -> AssistantState:
    verification = _VERIFIER.verify(
        answer=state.get(STATE_FINAL_ANSWER, ""),
        citations=state.get(STATE_CITATIONS, []),
        retrieval_results=state.get(STATE_RETRIEVAL_RESULTS, []),
        tool_calls=state.get(STATE_TOOL_CALLS, []),
    )

# compose：验证不通过时拒绝回答
def compose(state: AssistantState) -> AssistantState:
    if not verification_passed:
        answer = "当前答案未通过来源或数字验证，无法安全返回。..."
```

### 解读

不是靠 prompt 防幻觉，是靠流程做硬约束：

**四层约束**：
1. `extract_citations`：从检索结果提取引用，不依赖 LLM "自觉"标注
2. `verify`：校验引用是否真实存在、数字是否一致、是否有幻觉迹象
3. `compliance_check`：检测敏感词和业务建议
4. `compose`：任何一层不通过都拒绝回答

**fail-closed 原则**：不确定就不要返回。宁可返回"无法安全返回"，也不返回可能幻觉的内容。

**诚实说局限性**：验证本身依赖 LLM，不是 100% 可靠。我的策略是"把幻觉风险转化为可追踪的决策"——每一步都有日志，出问题可以追溯。
