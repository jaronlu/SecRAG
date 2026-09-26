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
- **持仓与关注池**：每位用户维护自己的持仓与关注标的，所有读写按 `user_id` 隔离，越权按不存在处理。
- **每日增量扫描与事件分级**：对持仓标的扫描抓取产物生成事件卡片，按 P0 / P1 / P2 分级，每张卡片保留判定依据。

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

开启 Langfuse（见下节）后，演示的每次请求在 Langfuse 控制台有一条完整观测视图：

- **Agent 链路**：根 trace（`agent.request`）之下，会话加载、查询理解、检索计划、检索、
  结果过滤、ReAct 推理、引用验证、合规检查、回答编排等每个节点一个 span，父子关系就是
  真实执行顺序。
- **节点耗时**：每个节点 span 的 `duration_ms` 元数据给出耗时，可直接定位慢在检索还是推理。
- **模型调用次数**：查询理解和 ReAct 推理产生的每次 LLM 调用是一条 generation 观测，
  trace 内 generation 的条数即本次请求的模型调用次数（含验证失败后的重试）。
- **token 用量与成本**：每次 LLM 调用记录 prompt / completion / total token，Langfuse 按
  模型定价汇总整个 trace 的 token 与成本（需在 Langfuse 项目中定义模型价格，未定义时
  只显示 token 数）。

注意隐私边界：Langfuse 只接收链路元数据。问题原文、模型完整回答、文档与引用原文不会进入
Langfuse，仍然只能在受控本地 SQLite 审计系统中查看——观测平台看耗时和结构，审计系统看内容。

### 6. 链路追踪（Langfuse，可选）

SecRAG 用 [Langfuse](https://langfuse.com) 观测 Agent / LLM 链路（节点耗时、token、模型
调用），默认关闭。职责边界：权限、引用、合规审计仍在本地 SQLite；QPS、延迟、错误率与
缓存指标仍在 Prometheus（`/metrics`），Langfuse 不重复建设。

配置项（`cp .env.example .env` 后按需修改）：

| 配置项 | 默认值 | 说明 |
|---|---|---|
| `LANGFUSE_ENABLED` | `false` | 总开关。开启时 `LANGFUSE_HOST`、`LANGFUSE_PUBLIC_KEY`、`LANGFUSE_SECRET_KEY` 必填，缺失会在启动时报错 |
| `LANGFUSE_HOST` | `https://cloud.langfuse.com` | Langfuse 实例地址，自托管时改为自有地址（仅接受 http/https） |
| `LANGFUSE_PUBLIC_KEY` | 空 | 项目公钥，Langfuse 控制台 → Settings → API Keys 创建 |
| `LANGFUSE_SECRET_KEY` | 空 | 项目私钥，与公钥成对，不要提交进仓库 |
| `LANGFUSE_SAMPLE_RATE` | `1.0` | 采样比例 0.0~1.0，`1.0` 全量。开发可全量，生产按比例；失败请求不参与采样，保证错误可查 |
| `LANGFUSE_CAPTURE_CONTENT` | `false` | 内容捕获，默认关闭。仅 `APP_ENV=development` 生效，且内容先经统一 PII 脱敏；其他环境强制关闭 |

默认脱敏是两层防线：

1. **metadata 白名单**：Langfuse 只接受固定标量字段——`request_id`、`thread_id`、节点名、
   模型名、耗时、token 用量、检索数量、重试次数、验证/合规结果等。新增字段必须显式登记到
   `LangfuseTraceMetadata` 白名单（`src/utils/langfuse_adapter.py`），未登记的键一律丢弃。
2. **导出层兜底脱敏**：LangChain callback 自动挂到 span 上的 input / output 内容属性，
   导出前默认整段删除；只有开发环境显式开启 `LANGFUSE_CAPTURE_CONTENT` 才保留，且先经
   `redact_pii` 统一脱敏。用户原始问题、模型完整回答、文档与 chunk 原文、工具原始参数、
   SQL、客户 ID、持仓明细和 PII 默认不离开进程。

可靠性（fail-open）：Langfuse 未配置、超时、鉴权失败或服务不可用时，问答、审计与合规
链路照常完成，错误只写本地日志并累计 Prometheus 计数器
`secrag_langfuse_export_errors_total`（按 timeout / auth / exception 分类）与
`secrag_langfuse_dropped_total`（按丢弃原因），不会抛进业务路径。

查看 trace：

1. `.env` 中设置 `LANGFUSE_ENABLED=true` 并填入密钥，重启服务后发一次问答请求
   （`uv run python scripts/demo.py` 即可）。
2. 打开 `LANGFUSE_HOST` 的 Web UI → **Traces**，按名称 `agent.request` 过滤（失败请求为
   `agent.request.error`）。
3. trace 详情即上一节的节点 span 树；LLM 调用是 generation 观测（含模型名与 token 用量），
   工具调用有独立 span。
4. trace metadata 中的 `request_id` 与本地 SQLite 审计记录共用同一 ID，`thread_id` 与 API
   响应一致；需要核对回答原文时，用这个 `request_id` 回本地审计库查询。

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

抓取脚本的行为：成分股列表在运行时从数据源解析，代码中没有硬编码股票代码表；按内容哈希跳过
已入库记录，重复执行不重复下载；单只标的失败只写入 `.fetch_failures.json` 而不中断整批；批次
结束后在产物目录留下 `.fetch_state.json` 水位。

外部数据接口可能限流、断连或变更，仓库中的固定样本用于保证本地解析与入库验证不依赖实时抓取。

### 每日扫描与分级

对每位用户的持仓与关注标的扫描上述产物，生成事件卡片并分级。目前只有 Python 调用入口，尚未
提供 CLI 或 HTTP 接口。

```python
from pathlib import Path

from src.jobs.daily_scan import SQLiteDailyScanStore, run_daily_scan
from src.portfolio.store import SQLitePortfolioStore

summary = run_daily_scan(
    output_dir=Path("data/raw/real_securities_data"),
    portfolio_store=SQLitePortfolioStore("data/portfolio.db"),
    scan_store=SQLiteDailyScanStore("data/scan.db"),
    user_ids=["u1"],
)
print(summary["events_inserted"], summary["grade_counts"])
```

同一交易日重复调用不会重复生成卡片：幂等键由事件本身计算，不包含运行日期，因此在之后的日子
重跑也不会让同一事件第二次出现。P2 卡片同样入库但标记为 `filtered`，保留被滤除的原因；只有
P0 / P1 供下游消费。每次运行在 `scan.db` 留下一条按用户记录的水位。

## 开发验证

```bash
uv run ruff check .
uv run pytest
```

当前测试覆盖 Agent 节点与路由、身份和权限、检索、数据摄入、会话、合规、工具、API、持仓存储
以及每日扫描与事件分级。

## 项目结构

```text
src/
  agents/       LangGraph 工作流、状态和 Agent 工具
  api/          FastAPI 路由、身份绑定和 Web UI
  ingestion/    文档解析、切片、增量入库和任务状态
  jobs/         每日增量扫描与事件分级
  portfolio/    持仓与关注池持久化
  rag/          基础 RAG 链
  retrieval/    多源向量检索和权限过滤
  tools/        计算、行情、SQL、财务指标与重排工具
  utils/        引用验证、合规、会话、审计和追踪
scripts/        入库、演示、检查和评估脚本
tests/          自动化测试
```

## 当前边界

- 标准检索链路是角色感知的多源向量检索，并在每个检索步骤叠加 BM25 与 RRF 融合；BM25 索引构建失败时静默降级为纯向量结果，降级过程不记录原因。
- Reranker 作为 Agent 工具提供，是否调用由推理过程决定，不是标准检索阶段的固定步骤。
- LangGraph checkpointer 使用内存存储；服务重启后不会恢复图执行状态。
- 会话、审计和入库任务使用本地 SQLite，后台入库基于单机进程，不支持多实例任务调度。
- demo token、样例数据和小规模评估集只能证明流程，不能证明生产安全性、吞吐量或回答质量。
- OpenAI-compatible provider 和公开数据抓取依赖外部服务；Ollama 模式仍需本地模型与 embedding 模型。
- 事件分级阈值的默认值是启发式起点，未经真实标注数据标定，不应直接当作生产判据使用。
- `uv sync` 会卸载任何未写入 `pyproject.toml` 的包。开发所需的 `pytest-asyncio` 与 `ruff` 位于 optional-dependencies，需要 `uv sync --extra dev` 才会安装，裸跑 `uv sync` 会把它们移除。
- `src/ingestion/loaders.py` 经 langchain-unstructured 间接依赖 spacy 模型 `en_core_web_sm`，该模型尚未纳入依赖声明；`uv sync` 之后 `tests/test_loader.py` 需要另行安装它。
- 抓取脚本依赖的三个 provider 契约（akshare 成分股列名、cninfo orgId 解析、efinance 与 baostock 返回值）未在联网环境实机核对；不符之处体现为失败清单，不会提前报错。

## License

MIT
