---
type: architecture tutorial
title: 状态、权限与安全边界
description: 本页用一条问答请求说明 AssistantState 如何贯穿 SecRAG 的各个节点，并区分认证、检索权限、工具授权、答案验证、合规检查、会话隔离和审计各自负责什么。
tags: [architecture, state, authorization, security, audit]
verified:
  - by: openwiki/0.5.2
    at: 2026-09-24T17:03:23.068Z
sources:
  - id: openwiki-source-ce706aa9fc0c231bbb5791c7
    resource: repo://src/agents/graph.py
  - id: openwiki-source-1204a4ec52aa8e3c70a8eac9
    resource: repo://src/agents/nodes.py
  - id: openwiki-source-cb2873cb59409086c3aa128e
    resource: repo://src/agents/state.py
  - id: openwiki-source-440a8f53b847f49df7895d34
    resource: repo://src/agents/tools.py
  - id: openwiki-source-53bdf62a9d0ee4ca3a837299
    resource: repo://src/api/auth.py
  - id: openwiki-source-e532544007c5ed049c805ecd
    resource: repo://src/retrieval/hybrid_retriever.py
  - id: openwiki-source-d7fe4b257987f8cbf763fe5e
    resource: repo://src/utils/audit.py
  - id: openwiki-source-004dc7c0b0c3948dc335f697
    resource: repo://src/utils/compliance.py
  - id: openwiki-source-ca13b5edb6eb87b3be9baecf
    resource: repo://src/utils/conversation.py
  - id: openwiki-source-fc350d8da22ace16e2b8e8c4
    resource: repo://src/utils/verifier.py
generated: { by: "codex", at: "2026-09-24T17:03:23.068Z" }
---

# 状态、权限与安全边界

先记住一个核心观点：SecRAG 不是“先生成答案，再把敏感文字删掉”。权限和安全检查分布在请求的多个阶段；任何一层发现不能安全继续，都会返回拒绝、重试或安全兜底。

## 1. `AssistantState` 是整条链路的共享内存

一次问答会创建一份 `AssistantState`。它可以看成一个大字典，但字段是固定的，节点只负责读取自己需要的部分并返回增量更新。

主要字段可以分成六组：

| 内容 | 例子 | 谁会读写 |
| --- | --- | --- |
| 用户上下文 | `user_id`、`user_role`、`data_permissions`、`client_id` | 认证入口、检索器、工具授权、合规 |
| 查询理解 | `original_query`、`resolved_query`、`entities`、`ambiguity` | 会话节点、查询理解、Planner |
| 检索 | `retrieval_plan`、`retrieval_results`、`retrieval_attempts` | Planner、`HybridRetriever`、结果过滤 |
| 推理 | `messages`、`tool_calls`、`reason_attempts` | ReAct 子图和工具记录节点 |
| 质量控制 | `citations`、`verification`、`compliance`、`confidence` | 引用提取、验证、合规、组装 |
| 结果与追踪 | `final_answer`、`intermediate_steps`、`audit_trail` | 组装、会话持久化、审计 |

因此，节点之间不是靠隐式全局变量传递信息，而是围绕同一份状态协作。`src/agents/state.py` 是字段的集中定义，`src/schemas/constants.py` 中的 `STATE_*` 常量则约束键名。

## 2. 第一层：认证把请求身份绑定到服务端状态

`src/api/auth.py` 的 `authenticate_user` 只接受 `Authorization: Bearer <token>`，再从服务端的 demo token 表得到 `user_id`、角色和部门。请求体不能自行声明“我是 technical”来获得权限。

认证成功后，`build_assistant_initial_state` 把身份、角色允许的数据权限、线程号、轮次号和原始问题放进 `AssistantState`。从这一刻开始，后面的节点都使用状态里的身份，而不是重新相信用户输入。

本地演示 token（例如 `demo-advisor`、`demo-tech`）只是开发方便，不能当作生产身份系统。

## 3. 第二层：检索计划和检索结果都做权限检查

权限检查至少有两道：

1. **计划级检查**：`HybridRetriever._filter_plan_by_role` 根据角色允许的数据源过滤 Planner 生成的计划。越权的数据源不会被悄悄丢掉，而会变成带 `denied=True` 的显式结果。
2. **结果级检查**：检索到 chunk 后，`_filter_results_by_role` 检查 `permission_level` 和 `allowed_roles`。非公开 chunk 缺少 `allowed_roles` 时默认拒绝；角色不在列表中时只返回安全的拒绝占位，不把原文带入后续上下文。

因此，即使 LLM 误生成了越权检索计划，执行层仍会再次拦截。`tests/test_hybrid_retriever.py` 覆盖未知角色、未知数据源、结果级角色标签和非公开数据的拒绝行为。

## 4. 第三层：ReAct 工具在真正执行前再授权

`src/agents/tools.py` 注册产品、法规、研报、FAQ、计算、行情、SQL、财务指标等工具。`get_tools_for_role` 先按角色和本轮排除的数据源决定工具是否可见。

但“模型看得见工具”不等于“工具一定能执行”。ReAct 子图使用 `authorize_reason_tool_call` 在执行边界再次确认工具名是否允许；随后还用线程超时保护工具调用，并对超时或异常工具设置短暂熔断。这样可以避免一个失败或失控的工具拖住整条请求。

## 5. 查询和文档内容都被视为不可信输入

`query_understand` 会先截断查询，并用 `_detect_injection` 标记诸如“忽略以上指令”的 Prompt Injection。命中时不会擅自删除用户问题，而是把标记留在状态中，让后续 Prompt 加固。

检索到的文档也不是指令来源。`_harden_context` 会把疑似注入的文档包裹成“不可信文档内容”，提醒模型只能把它当证据，不能执行其中的命令。这个边界很重要：知识库里的文字可能来自外部文件，不能因为被检索到就自动获得控制权。

## 6. 生成答案后仍要经过四类验证

`extract_citations` 只从本轮允许使用的检索结果提取引用；随后 `ComprehensiveVerifier` 做四类检查：

- **来源验证**：答案里的 `[来源N]` 必须存在，并且引用的 source/chunk 属于本轮结果。
- **数字验证**：答案中的数字必须能在检索证据或成功的工具输出中找到。
- **一致性验证**：阻止“买入/卖出”“看多/看空”等明显互相矛盾的结论同时出现。
- **幻觉检测**：逐句比较答案与证据；证据覆盖不足时判定失败。

验证失败且还没达到最大推理次数时，图会回到 `reason` 重新生成；超过上限则继续走安全分支。`compose` 对未通过验证的答案直接清空引用，改成“无法安全返回”的提示。

## 7. 合规检查决定“能不能对外显示”

`compliance_check` 使用 `ComplianceChecker` 检测敏感信息、投资建议模式、目标价表达和适当性风险。投顾/销售角色输出“建议买入”等业务建议会被拦截；合规角色缺少法规条款引用时也会标记失败；投顾在有客户上下文时遇到高风险产品，会附加适当性提示。

合规失败不会把原答案原样送给用户。`compose` 会清空引用并返回合规阻断提示，同时保留必要的风险或适当性说明。也就是说，合规是最终输出闸门，不是 UI 层的装饰。

## 8. 会话、用户可见响应和审计记录彼此分离

`SQLiteConversationStore` 只按当前 `user_id` 读取线程和消息；线程的角色或 `client_id` 发生变化时会拒绝继续使用。写入时会保存用户消息、助手答案、解析后的查询、实体和引用，并用 `request_id` 做幂等保护。

用户收到的是 `AssistantQAResponse` 中的答案、引用、置信度和合规结果；内部 `audit_trail` 不通过 API 返回。`audit_log` 会把 Query → Retrieve → Reason → Verify → Compose 的节点路径、工具调用、来源、验证和合规结果写入 SQLite 审计库。

如果审计库暂时写失败，回答不会被强行阻断：会话 outbox 和本地 `data/audit_outbox.jsonl` 保留待重试记录，并在审计状态中标记写入失败。这是“用户体验不中断”和“审计问题不丢失”之间的折中。

## 9. 读代码时建议按这条安全路线检查

遇到一个新节点，可以依次问：

1. 它从 `AssistantState` 读取了谁的身份和哪些证据？
2. 它是否把外部输入当成了指令？
3. 它输出的数据会不会绕过下一层权限或验证？
4. 失败时是重试、拒绝、降级，还是把不可信内容继续向下传？
5. 最终结果是否进入会话和审计，且没有把内部审计信息泄露给用户？

按这条路线阅读 `src/api/auth.py`、`src/agents/graph.py`、`src/agents/nodes.py`、`src/retrieval/hybrid_retriever.py`、`src/utils/verifier.py`、`src/utils/compliance.py` 和 `src/utils/conversation.py`，就能从“字段流动”和“安全边界”两个角度理解整个系统。
