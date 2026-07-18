# SecRAG

面向机构内部投研场景的 Agentic RAG 原型。系统通过角色权限、多源检索、工具调用、
引用验证、合规检查和审计记录，将知识问答组织为一条受约束、可追踪的工作流。

> 这是个人独立完成的架构验证项目，不是生产系统，不包含任何客户数据或客户定制代码。

## 核心能力

- **身份与权限绑定**：服务端根据 Bearer token 派生用户和角色，不信任请求体中的身份信息。
- **多层检索权限**：Planner、检索执行层和文档 chunk 元数据共同限制可访问的数据源与内容。
- **可控 Agent 工作流**：LangGraph 负责条件路由、有限重试和明确的拒绝终态。
- **ReAct 工具调用**：检索、计算器、适当性检查、行情、SQL 和财务指标工具按角色动态开放。
- **可信输出**：回答依次经过引用提取、来源与数字验证、合规检查，再生成最终响应。
- **会话与审计**：SQLite 保存会话、审计记录和入库任务状态。
- **增量入库**：支持稳定文档 ID、内容哈希、版本管理、更新跳过和旧 chunk 清理。

## 工作流

外层 StateGraph 负责业务流程，`reason` 节点内部是一次编译的 ReAct 子图：

```text
认证身份
  -> 加载会话 -> 消解追问 -> 查询理解 -> 生成检索计划
  -> 执行检索 -> 过滤结果
       |-> 结果不足：返回 Planner，有限重试
       |-> 全部越权：生成权限拒绝响应
       `-> 结果可用：进入 ReAct 子图
  -> 提取引用 -> 验证
       |-> 验证失败且未超限：重新推理
       `-> 继续
  -> 合规检查 -> 组织回答 -> 保存会话 -> 写入审计 -> END
```

权限不是生成答案后的文本过滤。身份、检索计划、检索执行、chunk 元数据和工具调用都包含独立校验；
非公开内容缺少 `allowed_roles` 时默认拒绝。

## 技术栈

- Python 3.11+
- FastAPI
- LangGraph / LangChain
- ChromaDB
- Sentence Transformers
- SQLite
- Pydantic

## 快速开始

### 1. 安装依赖

项目使用 [uv](https://docs.astral.sh/uv/) 管理 Python 环境：

```bash
uv sync --all-extras
```

### 2. 配置模型

```bash
cp .env.example .env
```

默认使用 OpenAI-compatible provider，需要配置：

```dotenv
LLM_PROVIDER=openai
OPENAI_API_BASE=https://your-provider.example/v1
OPENAI_MODEL=your-model
OPENAI_API_KEY=your-key
```

也可以切换到本地 Ollama：

```dotenv
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=llama3.1:8b
```

首次运行 embedding 时可能需要下载 `BAAI/bge-small-zh-v1.5`。

### 3. 入库示例数据

```bash
uv run python scripts/ingest.py data/raw/demo_knowledge_base/samples/product product
uv run python scripts/ingest.py data/raw/demo_knowledge_base/samples/regulation regulation
uv run python scripts/ingest.py data/raw/demo_knowledge_base/samples/faq faq
uv run python scripts/ingest.py data/raw/demo_knowledge_base/samples/report research_report
```

入库是增量操作。如需在一次完整目录扫描中归档已经删除的文档，可增加 `--full-scan`。

### 4. 启动服务

```bash
uv run uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

可用入口：

| 入口 | 用途 |
|---|---|
| `http://127.0.0.1:8000/` | 问答和文档入库 UI |
| `http://127.0.0.1:8000/docs` | OpenAPI / Swagger UI |
| `POST /v1/assistant/qa` | 唯一问答接口 |
| `/v1/assistant/threads*` | 会话创建、消息查询和删除 |
| `/v1/admin/ingestion/*` | technical 角色的入库管理接口 |

### 5. 运行演示

另开一个终端：

```bash
uv run python scripts/demo.py
```

演示脚本覆盖授权查询和权限拒绝场景，并打印回答、引用、置信度和合规状态。完整审计只在
服务端持久化，不通过问答接口返回。

## 身份验证

问答、会话和入库接口都要求 `Authorization: Bearer <token>`。仓库内置以下 demo token：

| Token | 角色 |
|---|---|
| `demo-advisor` | advisor |
| `demo-sales` | institutional_sales |
| `demo-compliance` | compliance |
| `demo-ops` | operations |
| `demo-tech` | technical |

请求示例：

```bash
curl -X POST http://127.0.0.1:8000/v1/assistant/qa \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer demo-tech' \
  -d '{"query":"系统操作流程怎么查？"}'
```

这些固定 token 只用于本地演示，不能替代生产环境中的 IdP、签名 token 和授权策略。

## 数据与评估

仓库包含两组可复现数据：

- `data/raw/demo_knowledge_base/`：权限、检索和入库流程使用的最小样例。
- `data/raw/real_securities_data/`：来自公开来源的财报、研报和结构化证券数据样本。

检索评估：

```bash
uv run python scripts/evaluate_retrieval.py
```

该命令使用 `scripts/evaluate_retrieval.sample.json`，输出 `recall@5`、`recall@10`、`MRR`、
`precision@5`、覆盖率和权限拦截准确率。样本量很小，只验证评估链路，不代表生产效果。

权限冒烟检查：

```bash
uv run python scripts/check_permissions.py
```

重新获取公开证券数据需要临时安装抓取依赖并访问外部数据源：

```bash
uv run --with akshare --with efinance --with baostock \
  python scripts/fetch_real_securities_data.py
```

外部数据接口可能限流、断连或变更，仓库中的固定样本用于保证本地解析与入库验证不依赖实时抓取。

## 开发验证

```bash
uv run ruff check .
uv run pytest
```

当前测试覆盖 Agent 节点与路由、身份和权限、检索、数据摄入、会话、合规、工具以及 API。

## 项目结构

```text
src/
  agents/       LangGraph 工作流、状态和 Agent 工具
  api/          FastAPI 路由、身份绑定和 Web UI
  ingestion/    文档解析、切片、增量入库和任务状态
  rag/          基础 RAG 链
  retrieval/    多源向量检索和权限过滤
  tools/        计算、行情、SQL、财务指标与重排工具
  utils/        引用验证、合规、会话、审计和追踪
scripts/        入库、演示、检查和评估脚本
tests/          自动化测试
```

## 当前边界

- 标准检索链路是角色感知的多源向量检索，不包含 BM25、RRF 等稀疏检索融合。
- Reranker 作为 Agent 工具提供，是否调用由推理过程决定，不是标准检索阶段的固定步骤。
- LangGraph checkpointer 使用内存存储；服务重启后不会恢复图执行状态。
- 会话、审计和入库任务使用本地 SQLite，后台入库基于单机进程，不支持多实例任务调度。
- demo token、样例数据和小规模评估集只能证明流程，不能证明生产安全性、吞吐量或回答质量。
- OpenAI-compatible provider 和公开数据抓取依赖外部服务；Ollama 模式仍需本地模型与 embedding 模型。

## License

MIT
