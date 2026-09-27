---
type: workflow tutorial
title: 问答请求执行链路：从 HTTP 到最终答案
description: 本页跟踪 POST /v1/assistant/qa 的一次完整执行，解释认证、会话、LangGraph 状态图、检索、ReAct、验证、合规、持久化和审计，以及各个提前结束或重试分支。
tags: [workflow, fastapi, langgraph, agent, qa]
verified:
  - by: openwiki/0.5.2
    at: 2026-09-25T04:47:56.817Z
sources:
  - id: openwiki-source-61267d3d2b88d5be53534466
    resource: repo://docs/architecture-overview.json
  - id: openwiki-source-ce706aa9fc0c231bbb5791c7
    resource: repo://src/agents/graph.py
  - id: openwiki-source-1204a4ec52aa8e3c70a8eac9
    resource: repo://src/agents/nodes.py
  - id: openwiki-source-9abd0efc90fa978f061bb160
    resource: repo://src/api/main.py
  - id: openwiki-source-e532544007c5ed049c805ecd
    resource: repo://src/retrieval/hybrid_retriever.py
  - id: openwiki-source-d7fe4b257987f8cbf763fe5e
    resource: repo://src/utils/audit.py
  - id: openwiki-source-ca13b5edb6eb87b3be9baecf
    resource: repo://src/utils/conversation.py
generated: { by: "codex", at: "2026-09-25T04:47:56.817Z" }
---

# 问答请求执行链路：从 HTTP 到最终答案

可以把一次问答想成一条“传送带”：API 先把请求装进 `AssistantState`，LangGraph 节点依次加工这份状态，最后只把安全的结果返回给用户。

先区分两张图的用途：`docs/architecture-overview.svg` 是面试和架构阅读用的总览图，只保留认证、问答 API、查询理解、检索、ReAct、验证输出这条主链；本页流程图则展开真实运行时的澄清、权限拒绝、检索重试、工具上限、验证重推和合规阻断分支。总览图中的“ReAct 推理”对应本页的 `prepare_reason → call_reason_model → execute_reason_tools / finalize_reason` 子图，“验证与输出”对应引用提取、验证、合规和 `compose`。

```text
POST /v1/assistant/qa
  │
  ├─ Bearer 认证 + 限流 + 创建/检查会话
  ├─ 构造 AssistantState
  ├─ 加载历史 → 消解追问 → 查询理解
  │                           ├─ 有歧义 → 澄清 → 保存 → 结束
  │                           └─ 无歧义 → Planner → 检索 → 相关性过滤
  │                                             ├─ 全部越权 → 拒绝 → 保存 → 审计
  │                                             ├─ 结果不足 → 重新 Planner/检索（有限次）
  │                                             └─ 结果可用 → ReAct 推理/工具
  ├─ 提取引用 → 四类验证
  │                 ├─ 失败且未超限 → 回到 ReAct
  │                 └─ 继续 → 合规检查 → 组织答案
  └─ 保存会话 → 写审计 → 返回 answer/citations/confidence/compliance
```

这张详细图表达的是“控制流”，不是每个基础设施调用的物理拓扑：向量库、BM25/RRF、Embedding 和 SQLite 由相应节点或服务按需使用，最终状态仍沿着外层 StateGraph 汇聚到持久化和审计。

## 1. 请求先经过 FastAPI，而不是直接进入 LLM

请求入口在 `src/api/main.py` 的 `assistant_qa`。FastAPI 依赖注入先调用 `authenticate_user`，要求 `Authorization: Bearer <token>`。随后 API：

1. 按 `user_id` 做每分钟 30 次的滑动窗口限流；
2. 创建新线程，或确认传入的 `thread_id` 属于当前用户；
3. 检查线程的角色和 `client_id` 没有被换掉；
4. 生成本轮 `turn_id`，调用 `build_assistant_initial_state`；
5. 懒加载带 checkpointer 的 Agent Graph；
6. 用 `thread_id` 作为 LangGraph 的 configurable key，并设置递归上限。

如果语义缓存命中，API 会在进入 Agent Graph 前直接返回缓存的答案、引用和置信度；这条路径仍然带当前线程号和用户角色的响应信息。

LLM 调用超时返回 504；连接、鉴权、限流或服务端错误等 provider 不可用情况返回 503；其他未分类异常返回 500。这些错误分支都在 API 层完成，不让异常响应变成半截答案。

## 2. 节点都围绕同一份 `AssistantState`

`build_assistant_initial_state` 会初始化身份、会话、原始问题、检索计划、检索结果、消息、工具调用、验证、合规、最终答案和审计轨迹等字段。

图中的节点不会互相直接调用；它们接收当前状态，返回要合并的字段。`_traced_node` 还会记录节点名、耗时和成功标记，最后这些信息进入审计执行路径。

这就是阅读代码时的主线：先看一个节点读取了哪些 state key，再看它返回了哪些 key，而不是只看函数之间的普通 Python 调用关系。

## 3. 会话上下文和追问消解

图从 `load_conversation_context` 开始。它只读取当前 `thread_id + user_id` 可见的消息，并限制历史 token 预算，保留最近内容。然后 `resolve_followup_query` 使用会话摘要中的实体，把“这个产品”“前面那家公司”等追问改写成包含上下文的查询。

这一步不会把别的用户或别的线程的内容混进来。会话不存在、已删除或角色/客户上下文不匹配时，存储层抛错，API 转成 404 或 409。

## 4. 查询理解：先清洗，再让模型结构化

`query_understand` 先做三件不依赖 LLM 的事：

- 截断超长查询；
- 标记可能的 Prompt Injection；
- 记录 PII 发现和语言。

然后让 LLM 返回固定 JSON：意图、查询类型、实体、重写查询和歧义列表。JSON 解析失败时，会回退到 `unknown + 原查询`，保证流程还有机会继续。

如果 `ambiguity` 非空，`should_clarify` 把流程路由到 `clarify`。这个节点最多列出三个需要补充的信息，生成澄清问题后直接进入会话保存，跳过 Planner、检索、推理和验证。

## 5. Planner 生成“去哪查、查什么”

没有歧义时，`planner` 根据重写查询、意图、实体和用户角色生成检索计划。计划中的每一步通常包含：

```json
{
  "source": "product_search",
  "query": "产品风险等级",
  "top_k": 5,
  "filters": {"product_type": "fund"}
}
```

Planner 生成的 JSON 仍然是不可信输入。代码会把每一步规范化为 `RetrievalPlanStep`，过滤角色不允许的数据源，并把时间范围、研报股票代码等条件合并到 filters。多跳重试时，Planner 还会从已有结果的 metadata 中提取实体，避免第二轮重复同一个查询。

## 6. 检索和相关性过滤

`retrieve` 创建带角色和数据权限的 `HybridRetriever`，按计划执行向量检索、可选 BM25/RRF 融合和结果级权限过滤；本轮结果会累加到状态中，同时递增检索次数和 chunk 计数。

`grade_and_filter` 再做一次本轮结果整理：

1. 丢掉低于阈值的候选；
2. 按 source + chunk 内容去重；
3. 尝试使用 BGE Reranker；
4. 保留前 `GRADE_TOP_K` 条，并记录 reranker 是否真的应用。

这里要区分两种“没有结果”：

- 有结果但全部是 `denied`：进入 `permission_denied_response`，在 LLM 推理前短路，避免把无权内容送进上下文；
- 没有足够的可用结果：`should_retry_retrieval` 回到 `planner`，允许有限次多跳检索；达到 `DEFAULT_MAX_HOPS` 后继续向下，最终由验证/置信度反映证据不足。

## 7. ReAct 子图：模型需要时才调用工具

结果足够时，外层图进入 `reason` 子图。它的固定结构是：

```text
prepare_reason
  → call_reason_model
      ├─ 有 tool_calls → execute_reason_tools → record_tool_results → call_reason_model
      ├─ 无 tool_calls → finalize_reason
      └─ 达到工具次数上限 → tool_limit_response
```

`prepare_reason` 把检索证据、角色说明和安全约束装进系统 Prompt；`call_reason_model` 调用按角色绑定的模型；模型若提出工具调用，`ToolNode` 会在 `authorize_reason_tool_call` 中再次做角色授权、超时和熔断处理。

当模型不再提出工具调用时，`finalize_reason` 把最后一条 AIMessage 规整成可见答案。如果工具循环超过 `MAX_TOOL_ITERATIONS`，`tool_limit_response` 直接返回“无法安全完成”的结构化提示，而不是无限重试。

## 8. 引用、验证和有限重推

`extract_citations` 只从本轮非拒绝的检索结果生成最多五条引用。随后 `verify` 调用综合验证器检查来源、数字、一致性和幻觉，并额外检查投顾/销售角色的投资建议表达。

验证通过后继续合规；验证失败时，`should_reason_again` 在 `MAX_REASON_ATTEMPTS` 以内把流程送回 `reason`，让模型基于同一轮证据重新组织答案。达到上限后不再重推，`compose` 会把答案替换为无法安全返回的提示并清空引用。

## 9. 合规、最终组装和持久化

`compliance_check` 检测敏感信息、投资建议、目标价、法规条款精度和高风险产品适当性。无论合规通过还是失败，图都会进入 `compose`；区别是失败时只返回合规阻断提示，并附带必要的风险/适当性说明。

`compose` 最终计算置信度：验证或合规失败为 low；验证高置信、至少三条有效结果且 reranker 真正应用，才可能为 high；其他成功结果通常为 medium。

之后：

1. `persist_conversation_turn` 把用户消息、助手答案、解析查询、实体和引用写入 SQLite；
2. `audit_log` 构造完整 `AuditEntry`，记录检索计划、工具调用、节点耗时、验证和合规结果；
3. 审计库写失败时写入 outbox，不阻断已通过安全检查的回答；
4. API 只返回答案、引用、置信度和合规结果，不把内部 `audit_trail` 暴露给前端。

## 10. 对照测试定位每个分支

想快速验证理解，可以从这些测试开始：

- `tests/test_api_main.py`：API 错误映射、线程配置和响应不泄露审计；
- `tests/test_agents.py`：检索重试、验证重推、合规路由、图编译和 Prompt Injection；
- `tests/test_hybrid_retriever.py`：数据源权限、chunk 元数据权限和错误结果；
- `tests/test_conversation.py`：线程隔离、上下文不匹配、幂等写入。

把测试中的状态构造和断言，与 `src/agents/graph.py` 的边连接对照起来，通常比从头读完所有节点更快理解这条执行链。
