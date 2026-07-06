# SecRAG — 机构投研知识问答 Agent

> 个人独立项目，非任何客户定制开发，不涉及任何客户保密信息。

## 这是什么

机构内部投研知识平台的 Agentic RAG 系统。目标是把知识库从"文档检索器"升级成结构上可信任的投研助手——不是靠模型自觉，而是靠工作流设计把权限、验证、追踪变成硬约束。

行业知识问答场景，核心难点从来不是"检索"，而是：

- 信息分散在研报、公告、法规、财报、内部制度里
- 同一问题，投顾 / 机构销售 / 合规看到的材料权限不同
- 数字错了会触发合规风险，引用错了会引发客户投诉
- 普通 RAG 系统无法区分"检索到的内容"和"模型脑补的内容"

## 系统架构

不是"提问 → 检索 → 生成答案"一步到位，而是拆成六个可追踪的节点，用 LangGraph StateGraph 编排：

| 节点 | 职责 |
|------|------|
| Planner | 生成检索计划 |
| Retriever | 按角色权限执行检索 |
| Reasoner | LLM 推理 + 工具调用 |
| Verifier | 来源校验、数字校验 |
| Composer | 生成带引用的最终回答 |
| Auditor | 记录追踪日志 |

```
Query → Planner → Retriever → Reasoner → Verifier → Composer → Auditor → Answer
```

节点间用条件路由控制流转，而非固定线性链路——例如检索结果不足时会回退到 Planner 重新规划检索路径。

## 技术亮点

- **基于角色的检索权限过滤（RBAC）**：投顾、机构销售、合规、运营、技术 5 种角色，权限直接决定检索路径和可见结果，不是事后过滤
- **混合检索 + 语义重排**：ChromaDB 向量检索为主，BGE Reranker 对召回结果做语义重排
- **LangGraph StateGraph 编排**：节点间条件路由，而非固定 Chain
- **FastAPI 问答接口**：面向内部使用场景的服务化封装

## 快速开始

```bash
# 1. 安装依赖
uv sync --all-extras

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env，至少填入 OPENAI_API_KEY（或切换 LLM_PROVIDER=ollama 使用本地模型）

# 3. 入库示例数据
uv run python scripts/ingest.py src/data/samples/product product
uv run python scripts/ingest.py src/data/samples/regulation regulation
uv run python scripts/ingest.py src/data/samples/faq faq
uv run python scripts/ingest.py src/data/samples/report research_report

# 4. 启动服务
uv run uvicorn src.api.main:app --port 8000

# 5. 另开一个终端，跑 demo 脚本验证两个核心接口
uv run python scripts/demo.py
```

`scripts/demo.py` 会依次调用 `/v1/qa`（单轮 RAG）和 `/v1/assistant/qa`（完整 Agent 工作流，含权限拒绝场景），打印回答、引用、置信度和追踪信息。

### Assistant API 身份绑定

`/v1/assistant/qa` 不再接受请求体里的用户角色作为可信输入。调用时必须传入
`Authorization: Bearer <demo-token>`，服务端根据 token 派生用户身份和角色。

可用 demo token：

```text
demo-advisor
demo-sales
demo-compliance
demo-ops
demo-tech
```

curl 示例：

```bash
curl -X POST http://127.0.0.1:8000/v1/assistant/qa \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer demo-tech' \
  -d '{"query":"系统操作流程怎么查？"}'
```

Swagger UI 中点击 `Authorize`，填入 `Bearer demo-tech` 后再调用接口。

### 检索效果评估

```bash
uv run python scripts/evaluate_retrieval.py
```

基于 `scripts/evaluate_retrieval.sample.json` 中的小样本评估集，输出 recall@5/recall@10/mrr/precision@5/coverage/permission_block_accuracy。样本量很小，仅用于验证评估流程可跑通，不代表生产环境效果。

当前本地小样本结果：

```text
samples: 5
recall@5: 0.700
recall@10: 0.800
mrr: 0.640
precision@5: 0.240
coverage: 1.000
permission_block_accuracy: 1.000
```

### 权限验收

```bash
uv run python scripts/check_permissions.py
```

当前本地验收结果：

```text
technical_langgraph_faq: PASS
operations_langgraph_faq_blocked: PASS
sales_langgraph_report_blocked: PASS
```

## 运行测试

```bash
uv run pytest
uv run ruff check .
```

CI 在每次 push / PR 时自动跑 lint + 测试。

## 可验证的代码事实

这个项目的可信度建立在代码本身，不是设计文档：

- 独立 commit 逐步推进，功能按模块落地（数据管道 → 基础 RAG → Agent 编排 → 检索优化 → 业务工具）
- `src/agents/`：`graph.py`（图构建）、`nodes.py`（六节点实现）、`state.py`（状态定义）、`tools.py`（工具定义）
- `tests/` 100+ 单元测试，覆盖节点逻辑、条件路由与图构建

## 职场竞争力评估

这个项目适合作为 AI 应用工程、RAG 工程、后端转 AI 工程方向的作品集项目。它不是简单聊天壳子，而是围绕角色权限、引用、审计、合规边界构建了一条可验证的 Agentic RAG 链路。

当前加分点：

- **业务约束明确**：同一问题在不同角色下有不同检索源和可见结果，不是普通“上传 PDF 问答”。
- **工程链路完整**：包含 FastAPI、LangGraph、Chroma、ingest、retriever、agent graph、tools、tests、demo UI 和启动脚本。
- **真实问题修复记录清晰**：已修过请求体伪造角色、工具绕过权限、chunk 级 `allowed_roles` 过滤、低相关内容误答、sources 去重等问题。
- **测试意识较强**：包含节点单测、权限 smoke check、metadata 测试和评估脚本入口。

当前短板：

- **评估指标还不够强**：需要用 `scripts/evaluate_retrieval.py` 跑出一组稳定的小样本 recall/precision/permission_block_accuracy 结果。
- **UI 仍偏调试型**：目前结果展示更像 JSON 调试页，后续应拆成 Answer、Citations、Audit Trail 和 Raw JSON。
- **LLM 依赖外部网络**：StepFun 或兼容 API 不稳定时会影响现场演示，需要明确网络要求或提供标注清楚的离线 demo mode。
- **样例数据规模较小**：当前能证明流程，但不能证明效果，需要扩充 20-50 条覆盖不同角色和拒答场景的小评估集。
- **设计文档发布方式未闭环**：`docs/design` 是本机外部路径软链接，公开仓库前需要内置精简设计文档或在 README 里说明详细设计不随仓库发布。

下一步最能提分的三件事：

1. 把 `scripts/check_permissions.py` 的结果写进 README，形成固定权限验收说明。
2. 跑通 `scripts/evaluate_retrieval.py`，记录本地小样本评估结果。
3. 优化 UI 展示结构，减少 raw JSON 暴露，把答案、引用和审计信息分区展示。

## 技术栈

- **Agent 编排**：LangGraph
- **LLM 接口**：LangChain
- **向量检索**：ChromaDB
- **语义重排**：BGE Reranker
- **服务接口**：FastAPI

## 项目边界说明

这是一个验证架构可行性的个人项目，不是生产系统。需要明确的是：

- 引用准确率、幻觉率、响应延迟等指标，在项目设计阶段的 PRD 里作为目标值提出，**未经生产环境实测验证**，这里不引用这些数字作为已达成的成果
- 内部知识库的数据源（研报、公告、法规）目前使用的是模拟/公开数据，未接入任何机构的真实内部数据

## License

MIT
