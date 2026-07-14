# SecRAG 项目掌握与面试学习指南

> 目标：逐步掌握项目的业务背景、核心原理、源码实现、技术亮点、工程取舍和技术债，最终能够在面试中用 30 秒、2 分钟和 5 分钟三个版本准确表达。
>
> 事实边界：本文优先以当前仓库源码和可执行验证为依据。设计目标与当前实现不一致时，会明确区分“已实现”“部分实现”和“规划中”。

## 一、项目结论

SecRAG 是一个面向券商内部投研场景的 Agentic RAG 原型。它的主要价值不是让模型生成更长的答案，而是通过角色权限、多源检索、工具调用、引用验证、合规检查和审计日志，把知识问答变成一条受约束、可追溯的工作流。

适合将其作为以下岗位的作品集项目：

- AI 应用工程师
- RAG 工程师
- Agent 工程师
- Python 后端转 AI 应用工程师

面试中的准确定位应是：

> 这是一个具备完整工程链路的架构验证型项目，不是已经经过大规模数据、并发和生产合规验收的生产系统。

## 二、逐步学习路线

| 阶段 | 主题 | 必须掌握的内容 | 面试产出 |
|---|---|---|---|
| 1 | 项目定位与整体架构 | 业务问题、普通 RAG 的不足、完整工作流 | 30 秒项目介绍 |
| 2 | 数据摄入 | Loader、Chunk、Embedding、稳定 ID、增量更新 | 讲清数据如何进入知识库 |
| 3 | 检索与权限 | Chroma、领域检索器、角色过滤、chunk 权限 | 讲清为什么不会直接越权检索 |
| 4 | Agent Graph | State、节点、条件路由、重试和安全终态 | 讲清为什么使用 LangGraph |
| 5 | ReAct 与工具 | 工具选择、角色化工具注册、金融工具 | 讲清 Agent 和工作流的边界 |
| 6 | 可信输出 | 引用、数字验证、幻觉检测、合规检查 | 讲清“验证”和“合规”的区别 |
| 7 | 会话与审计 | SQLite 会话、Checkpointer、Outbox、幂等 | 讲清状态持久化和审计一致性 |
| 8 | API 与工程化 | FastAPI、身份绑定、异常处理、配置 | 讲清系统如何对外提供服务 |
| 9 | 评估与技术债 | 检索指标、权限验收、测试、性能缺口 | 能主动评价项目槽点 |
| 10 | 模拟面试 | 30 秒、2 分钟、5 分钟表达和压力追问 | 形成完整面试话术 |

## 三、第一课：项目解决什么问题

### 3.1 What：它是什么

SecRAG 是券商内部投研知识问答 Agent，处理的知识包括：

- 研报和公告
- 法规和内部制度
- 产品说明书和风险揭示书
- FAQ 和操作流程
- 行情、财务和结构化数据库数据

### 3.2 Why：为什么不能只做普通 RAG

普通 RAG 通常是：

```text
用户问题 -> 向量检索 -> 拼接上下文 -> LLM 生成
```

金融内部知识场景还有额外约束：

1. 不同角色能看到的文档不同。
2. 数字错误可能直接造成业务和合规风险。
3. 回答必须能定位到原始文档和 chunk。
4. 投顾、销售、合规的输出边界不同。
5. 系统需要保留查询、检索、推理和输出的审计记录。

因此系统不能只依赖 Prompt 要求模型“谨慎”，而要把约束编码进工作流。

### 3.3 How：当前真实工作流

核心图定义位于 `src/agents/graph.py`：

```text
认证身份
  -> load_conversation_context
  -> resolve_followup_query
  -> query_understand
  -> planner
  -> retrieve
  -> grade_and_filter
       |-> 检索不足：回到 planner
       |-> 全部越权：permission_denied_response
       `-> 结果可用：reason
  -> extract_citations
  -> verify
       |-> 验证失败且未超限：回到 reason
       `-> 继续
  -> compliance_check
  -> compose
  -> persist_conversation_turn
  -> audit_log
  -> END
```

这个流程包含 14 个业务节点。设计文档中出现的“6 节点”或“9 步工作流”是概括或旧版本，不应在面试中当成当前源码事实。

## 四、普通 RAG 与 SecRAG 的区别

| 维度 | 普通 RAG | SecRAG 当前实现 |
|---|---|---|
| 控制流 | 一次检索后生成 | 条件路由和有限重试 |
| 权限 | 常见做法是结果返回后过滤 | 身份、计划、执行、chunk 多层约束 |
| 数据来源 | 通常只有向量库 | 多领域向量检索和结构化工具 |
| 数字处理 | LLM 直接计算或复述 | Decimal 计算器、SQL、数字验证 |
| 引用 | 文本附带来源即可 | 引用必须属于本轮检索结果 |
| 合规 | 依赖 Prompt | 独立合规节点和 fail-closed 输出 |
| 可观测性 | 普通应用日志 | 节点耗时、工具调用、检索和审计记录 |
| 多轮对话 | 拼接历史消息 | 会话存储、实体摘要和追问消解 |

## 五、当前技术亮点

### 5.1 权限不是事后装饰

权限链路包括：

1. FastAPI 根据 Bearer token 派生用户身份和角色，不信任请求体角色。
2. Planner 只接受当前角色允许的数据源。
3. HybridRetriever 在执行检索前再次检查数据源权限。
4. 检索结果根据 `permission_level` 和 `allowed_roles` 做 chunk 级过滤。
5. 非公开数据缺少 `allowed_roles` 时默认拒绝。

关键源码：

- `src/api/auth.py`
- `src/agents/nodes.py::planner`
- `src/retrieval/hybrid_retriever.py`

面试表达重点：这是 fail-closed 设计。配置缺失时拒绝访问，而不是默认放行。

### 5.2 LangGraph 条件工作流

系统使用条件路由表达两类循环：

- 检索结果不足时重新规划，最多执行 `DEFAULT_MAX_HOPS` 次。
- 答案验证失败时重新推理，最多执行 `MAX_REASON_ATTEMPTS` 次。

同时存在明确安全终态：

- 所有结果都被权限拒绝时，不进入 LLM 推理。
- 验证或合规最终不通过时，Composer 不返回原始答案。

### 5.3 验证和合规职责分离

验证回答的是：答案是否有证据支撑。

- 引用是否属于本轮检索结果
- 数字是否存在于文档或成功工具输出中
- 是否出现简单矛盾
- 回答句子是否被证据覆盖

合规回答的是：即使答案有证据，是否允许向当前用户输出。

- 是否包含敏感信息
- 是否包含投资建议
- 合规角色是否引用到具体条款
- 高风险产品是否需要适当性提示

关键源码：

- `src/utils/verifier.py`
- `src/utils/compliance.py`
- `src/agents/nodes.py::compose`

### 5.4 数据摄入具有增量更新能力

摄入链路不只是“解析后写入向量库”，还包括：

- 文件哈希和 metadata 哈希
- 稳定的 `doc_id` 和 `chunk_id`
- 文档版本号
- 未变化文档跳过
- 变化文档 upsert
- 旧 chunk 删除
- 入库失败状态记录
- 全量扫描时归档已删除文档

关键源码：

- `src/ingestion/identity.py`
- `src/ingestion/pipeline.py`
- `src/ingestion/registry.py`

### 5.5 金融工具重视安全和精度

计算器使用 AST 白名单和 Decimal，避免 `eval` 安全风险及 float 精度问题。

SQL 工具使用 `sqlglot` 解析 AST，并限制为：

- 单条 SELECT
- 单表查询
- 表和字段白名单
- 禁止 JOIN、子查询、UNION、CTE
- 自动添加或限制 LIMIT
- SQLite 只读连接

关键源码：

- `src/tools/calculator.py`
- `src/tools/sql_query.py`

### 5.6 会话与审计使用 Outbox 思路

保存一轮对话时，会在同一个 SQLite 事务中写入：

- 用户消息
- Agent 回答
- 回合摘要
- 审计 Outbox 事件

后续 `audit_log` 写入审计库，并将 Outbox 标记为 processed；失败时保留 pending 状态和错误信息。这比“先保存会话，再随手写一条日志”更能处理部分失败和重试。

关键源码：

- `src/utils/conversation.py::insert_turn`
- `src/agents/nodes.py::audit_log`

## 六、面试中必须主动承认的槽点

### 6.1 HybridRetriever 名称大于实现

当前 `HybridRetriever` 的实际能力是：根据 Planner 的多数据源计划，调用不同领域的 Chroma 向量检索器。

它没有实现典型 Hybrid Search 所需的：

- BM25 或其他稀疏检索
- 向量与稀疏结果融合
- RRF 或加权融合

准确表述应是“角色感知的多源向量检索”，不能直接宣称已经完成稀疏和稠密混合检索。

### 6.2 Reranker 没有进入标准检索链路

项目有 `rerank_tool`，但正常检索流程只根据 Chroma 相似度排序。Reranker 是否被调用取决于 LLM，不是稳定的检索阶段能力。

此外，Reranker 运行依赖 `FlagEmbedding`，但该包当前没有列入 `pyproject.toml`。

所以 README 中“ChromaDB + BGE Reranker 对召回结果语义重排”的说法偏超前。

### 6.3 多跳检索更接近有限重试

当前 Graph 会在结果数量不足时回到 Planner，但 Planner 没有接收“上一轮缺少什么证据”或“已检索过哪些路径”的结构化反馈，可能重复生成相同计划。

更成熟的多跳检索应包括：

- 问题分解
- 子问题依赖关系
- 已获得证据摘要
- 缺失证据类型
- 去重后的下一跳查询

### 6.4 验证器主要是规则和词法启发式

当前幻觉检测使用字符/token 重叠，数字验证使用字符串存在性检查，一致性验证依赖关键词组合。

优点是成本低、可解释；缺点是：

- 同义改写可能被误判为无证据
- 数字虽然出现，但语义关系可能错误
- 引用存在不代表引用真正蕴含结论
- 复杂矛盾无法通过关键词识别

生产方案可增加 claim extraction、NLI、结构化字段校验和人工评估集。

### 6.5 Checkpointer 不持久化

LangGraph 当前使用 `InMemorySaver`，进程重启后 Graph checkpoint 会丢失。SQLite 保存的是用户可见会话和摘要，两者职责不同。

生产环境可使用持久化 Checkpointer，并明确 checkpoint、conversation store 和 audit store 的生命周期。

### 6.6 身份认证只是 Demo

当前使用硬编码 token 到用户的映射，能够证明“角色由服务端派生”，但不具备生产身份系统需要的：

- JWT 验签和过期时间
- SSO/OIDC
- token 撤销
- 用户和权限动态配置
- 租户隔离

### 6.7 评估证据不足且当前结果不可复现

仓库评估集只有 5 条，无法证明泛化效果。

2026-07-14 当前环境验证结果：

- 单元测试：202 passed
- Ruff：通过
- 检索评估：当前运行结果为 0
- 权限 smoke check：`technical_langgraph_faq` 失败

当前直接错误是：`_detect_device()` 在所有 macOS 环境中都选择 `mps`，但当前运行环境不支持 MPS，检索异常被转换为空错误结果。

因此 README 中记录的历史检索指标只能表述为“曾记录的本地结果”，不能说成当前版本稳定可复现的验收结果。

### 6.8 配置和设计存在漂移

- `DEFAULT_EMBEDDING_MODEL` 是 `BAAI/bge-m3`，Settings 默认值却是 `BAAI/bge-small-zh-v1.5`。
- 设计文档存在“6 节点”“9 步”等旧描述，当前源码是 14 个节点。
- 设计中的 Redis、PostgreSQL、持久化 Checkpointer 等仍主要是生产路线，而非当前落地事实。

## 七、30 秒面试表达

> 我做了一个面向券商内部投研场景的 Agentic RAG 原型。普通 RAG 通常检索一次就直接生成，但金融场景还需要处理角色权限、数字准确性、引用和合规，所以我用 LangGraph 把查询理解、多源检索、工具推理、引用验证、合规检查和审计拆成条件工作流。系统会在检索不足时有限重试，验证失败时重新推理，越权或合规失败则 fail closed。当前项目已经能证明完整架构和工程链路，但真正的稀疏稠密混合检索、生产认证和大规模评估仍需要完善。

## 八、2 分钟面试表达框架

### 1. 背景

券商内部知识分散在研报、公告、法规、产品文档和结构化财务数据中。传统文档问答不能解决权限、数字准确性、引用和合规问题。

### 2. 方案

使用 LangGraph 构建条件工作流：

```text
查询理解 -> 权限感知检索 -> 工具增强推理 -> 引用 -> 验证 -> 合规 -> 审计
```

### 3. 三个重点

1. 权限在服务端身份、Planner、执行层和 chunk 层多次校验。
2. 回答经过来源、数字、一致性和幻觉四类验证。
3. 对话和审计通过 SQLite、幂等 request ID 和 Outbox 思路保持可追溯。

### 4. 结果边界

项目有完整测试、样例数据和评估入口，但仍属于原型。评估集规模、真实 Hybrid Search、持久化 Checkpointer 和生产身份系统尚未完成。

## 九、高频面试考点

1. RAG、Agentic RAG、Graph RAG 有什么区别？
2. 为什么使用 LangGraph，而不是普通 Chain？
3. State 中为什么使用 TypedDict 和 reducer？
4. 条件路由如何避免死循环？
5. Planner 由 LLM 生成时，如何防止越权计划？
6. 为什么权限过滤不能只放在结果返回阶段？
7. 向量检索、BM25 和 Reranker 分别解决什么问题？
8. 当前 HybridRetriever 是否是真正的 Hybrid Search？
9. 引用存在为什么不代表答案一定可信？
10. 数字验证为什么不能只判断字符串是否存在？
11. 验证与合规有什么区别？
12. Checkpointer 与 ConversationStore 有什么区别？
13. Outbox 解决什么一致性问题？
14. SQL 工具如何防止注入和越权访问？
15. 如何评估检索、生成、权限、合规和端到端性能？
16. 如果并发达到 50 QPS，当前架构最先出现什么瓶颈？
17. 如何将 ChromaDB、SQLite 和 InMemorySaver 替换为生产组件？
18. 项目中最值得重构的部分是什么？

## 十、源码阅读顺序

第一遍只理解主流程：

1. `README.md`
2. `src/api/main.py`
3. `src/api/auth.py`
4. `src/agents/state.py`
5. `src/agents/graph.py`
6. `src/agents/nodes.py`

第二遍理解 RAG 链路：

1. `src/ingestion/identity.py`
2. `src/ingestion/chunkers.py`
3. `src/ingestion/pipeline.py`
4. `src/retrieval/vector_retriever.py`
5. `src/retrieval/hybrid_retriever.py`

第三遍理解可信与工程化：

1. `src/agents/tools.py`
2. `src/tools/`
3. `src/utils/verifier.py`
4. `src/utils/compliance.py`
5. `src/utils/conversation.py`
6. `src/utils/audit.py`
7. `scripts/evaluate_retrieval.py`
8. `tests/`

## 十一、第一课自测

完成本课后，应能够不看文档回答：

1. 这个项目解决的核心业务问题是什么？
2. 为什么它不能只使用一次向量检索加 LLM？
3. 当前 Graph 的主流程是什么？
4. 哪些失败会触发重试，哪些失败会直接拒绝？
5. 项目最强的三个技术亮点是什么？
6. 为什么不能宣称已经完成真正的混合检索？
7. 当前项目为什么仍然只能定位为原型？

建议练习：用自己的话完成一次 30 秒介绍，不照抄模板。后续学习应根据表达中暴露的知识缺口继续进入数据摄入、检索、Agent Graph 和可信输出模块。

