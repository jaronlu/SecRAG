# SecRAG — Agent Graph 真实结构

> 源码依据：`src/agents/graph.py`、`src/agents/nodes.py`、`src/agents/state.py`

---

## 一、整体流程

```
Query
  → load_conversation_context
  → resolve_followup_query
  → query_understand
  → planner
  → retrieve
  → grade_and_filter
  ↓ 条件路由 ↓
    ├─ retrieve          # 检索不足，回到 planner 重新规划（最多 DEFAULT_MAX_HOPS 次）
    ├─ permission_denied_response  # 全部检索结果被权限拒绝
    └─ reason
      → extract_citations
      → verify
      ↓ 条件路由 ↓
        ├─ retry          # 验证失败，重新推理（最多 MAX_REASON_ATTEMPTS 次）
        └─ compliance_check
          ↓ 条件路由 ↓
            ├─ pass → compose
            └─ block → compose  # 附合规拦截提示
              → persist_conversation_turn
              → audit_log
              → END
```

---

## 二、节点职责

| 节点 | 职责 | 输入 | 输出 |
|------|------|------|------|
| `load_conversation_context` | 从 SQLite 加载当前用户的会话历史和实体摘要 | `thread_id`, `user_id` | `chat_history`, `conversation_summary` |
| `resolve_followup_query` | 消解指代式追问（"它"/"这个"/"上述"），补全上下文 | `original_query`, `conversation_summary` | `resolved_query` |
| `query_understand` | LLM 结构化分析：意图分类、实体抽取、查询重写、歧义检测 | `resolved_query`, `user_role`, `department` | `intent`, `query_type`, `entities`, `rewritten_query`, `ambiguity` |
| `planner` | 根据角色权限生成多源检索计划（product / regulation / report / faq） | `rewritten_query`, `user_role`, `entities` | `retrieval_plan` |
| `retrieve` | HybridRetriever 按计划执行检索，累加结果，计数器+1 | `retrieval_plan`, `user_role`, `data_permissions` | `retrieval_results`, `retrieval_attempts` |
| `grade_and_filter` | 按 score 排序，去重，保留 top-K，保留 denied 结果 | `retrieval_results` | `retrieval_results`（过滤后） |
| `permission_denied_response` | 全部结果被拒时提前终止，生成权限拒绝回答 | — | `final_answer`, `citations`, `confidence` |
| `reason` | 内嵌 LangChain ReAct Agent：LLM 推理 + 工具调用（8 个工具按角色动态注册） | `retrieval_results`, `resolved_query`, `system_prompt` | `messages`, `tool_calls`, `final_answer` |
| `extract_citations` | 从本轮可用检索结果中提取引用标注 | `retrieval_results`, `resolved_query` | `citations` |
| `verify` | 来源校验、数字校验、一致性校验、幻觉检测；投顾/销售额外检测业务建议关键词 | `final_answer`, `citations`, `retrieval_results`, `tool_calls` | `verification` |
| `compliance_check` | 敏感词检测、业务建议拦截、风险提示、适当性警告 | `final_answer`, `user_role`, `client_id` | `compliance` |
| `compose` | 综合验证和合规结果生成最终回答；不通过时返回拒绝文本 | `final_answer`, `verification`, `compliance` | `final_answer`, `citations`, `confidence`, `risk_disclosure` |
| `persist_conversation_turn` | 将本轮可见对话写入 SQLite | `thread_id`, `turn_id`, `messages` | — |
| `audit_log` | 全链路追踪写入 SQLite（结构化 JSON，可对接 ELK/Loki） | `state` | `audit_trail` |

---

## 三、状态字段说明

`AssistantState` 关键字段：

| 分组 | 字段 | 说明 |
|------|------|------|
| 用户上下文 | `user_id`, `user_role`, `department`, `data_permissions`, `client_id` | 身份绑定，不来自请求体，来自 token |
| 会话上下文 | `thread_id`, `turn_id`, `turn_index`, `chat_history`, `conversation_summary` | SQLite 持久化 |
| 查询理解 | `original_query`, `rewritten_query`, `intent`, `entities`, `ambiguity`, `query_type` | LLM 结构化输出 |
| 检索计划 | `retrieval_plan`, `retrieval_attempts` | 多跳检索计数器 |
| 检索结果 | `retrieval_results`（reducer: concatenate） | 多轮累加 |
| 推理过程 | `messages`, `tool_calls`, `intermediate_steps`, `reason_attempts` | ReAct Agent 产物 |
| 验证/合规 | `verification`, `compliance` | 硬约束 |
| 最终回答 | `final_answer`, `citations`, `confidence`, `risk_disclosure` | 对外返回 |
| 追踪 | `audit_trail` | 全链路日志 |

---

## 四、条件路由详情

### 4.1 grade_and_filter → retrieve / permission_denied_response / reason

```python
def should_retry_retrieval(state: AssistantState) -> str:
    attempts = state.get("retrieval_attempts", 0)
    results = state.get("retrieval_results", [])
    usable = [r for r in results if not r.get("denied")]

    if results and not usable:
        return "denied"                      # 全部被拒 → 权限拒绝
    if attempts >= DEFAULT_MAX_HOPS:
        return "continue"                    # 已达最大跳数 → 继续推理
    if not results or len(usable) < CONFIDENCE_HIGH_MIN_RESULTS:
        return "retrieve"                    # 结果不足 → 重新检索
    return "continue"                        # 结果充足 → 继续推理
```

### 4.2 verify → retry / continue

```python
def should_reason_again(state: AssistantState) -> str:
    verification = state.get("verification", {})
    attempts = state.get("reason_attempts", 0)
    if not verification.get("passed", False) and attempts < MAX_REASON_ATTEMPTS:
        return "retry"                       # 验证失败但未超限 → 重新推理
    return "continue"                        # 通过或已达上限 → 继续
```

### 4.3 compliance_check → pass / block

```python
def is_compliant(state: AssistantState) -> str:
    compliance = state.get("compliance", {})
    if compliance.get("passed", False):
        return "pass"
    return "block"                           # 合规不通过，仍走 compose 附拦截提示
```

---

## 五、双层 Agent 架构

```
外层：LangGraph StateGraph（工作流编排）
  └─ 14 个节点，条件路由控制流转

内层：LangChain ReAct Agent（嵌套在 reason 节点）
  └─ create_agent(model=llm, tools=get_tools_for_role(role))
     └─ 8 个工具按角色动态注册
        ├─ 知识检索：product_search / regulation_search / report_search / faq_search
        ├─ 计算器：calculator
        ├─ 适当性检查：suitability_check
        ├─ 市场数据：market_data_tool
        ├─ SQL 查询：sql_query_tool
        ├─ 财务指标：financial_ratios_tool
        └─ 重排：rerank_tool
```

**注意**：`reason` 节点当前使用 `langchain.agents.create_agent` 嵌套在 LangGraph 内，每次调用重新编译 agent graph，存在性能浪费。Phase 2 重构计划：将 ReAct 循环建模为 Graph 子图，全程在 StateGraph 范式内。

---

## 六、权限过滤链路

```
请求到达
  → FastAPI authenticate_user（token 派生身份）
  → build_assistant_initial_state（注入 user_role / data_permissions）
    → planner 节点：ROLE_ALLOWED_SOURCES 过滤生成检索计划
      → retrieve 节点：HybridRetriever._filter_plan_by_role 二次过滤
        → 各检索器内部：metadata 级 allowed_roles / permission_level 过滤
```

双层过滤确保：即使 LLM 在 planner 阶段生成越权 source，执行层仍会拦截。

---

## 七、关键常量

| 常量 | 值 | 说明 |
|------|-----|------|
| `DEFAULT_MAX_HOPS` | — | 检索最大跳数 |
| `CONFIDENCE_HIGH_MIN_RESULTS` | — | 高置信度所需最小结果数 |
| `MAX_REASON_ATTEMPTS` | — | 推理最大重试次数 |
| `DEFAULT_TOP_K` | — | 默认检索条数 |
| `GRADE_TOP_K` | — | 过滤后保留条数 |
| `RETRIEVAL_MIN_SCORE` | — | 最低相似度分数 |
| `AGENT_RECURSION_LIMIT` | — | LangGraph 递归深度上限 |

---

## 八、技术债与重构计划

| 编号 | 问题 | 计划 |
|------|------|------|
| 1 | `reason` 节点每次调用重新编译 ReAct Agent | Phase 2：ReAct 循环建模为 Graph 子图 |
| 2 | 无 KV Cache | 评估在 LLM 调用层加缓存 |
| 3 | 无 MCP | 工具接口可迁移到 MCP 协议 |
| 4 | 样例数据规模小 | 扩充 20-50 条评估集 |
| 5 | UI 偏调试型 | 拆成 Answer / Citations / Audit Trail / Raw JSON 分区 |
