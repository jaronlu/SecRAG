# TODO

## Langfuse 真实数据场景接入

目标：为 SecRAG 增加 Langfuse Agent 链路追踪，同时保证真实金融数据不被发送到第三方观测平台，且 Langfuse 故障不影响问答、审计和合规流程。

### 实施任务

- [x] 增加 Langfuse 依赖并更新 `uv.lock`。
- [x] 增加配置项：
  - [x] `LANGFUSE_ENABLED`
  - [x] `LANGFUSE_HOST`
  - [x] `LANGFUSE_PUBLIC_KEY`
  - [x] `LANGFUSE_SECRET_KEY`
  - [x] `LANGFUSE_SAMPLE_RATE`
  - [x] `LANGFUSE_CAPTURE_CONTENT`，默认关闭
- [x] 新增统一 Langfuse adapter，封装 client、callback、trace 和 span 创建。
- [x] 在 API 请求入口创建根 trace，并关联内部 `request_id`、`thread_id` 和执行状态。
- [x] 通过 `RunnableConfig.callbacks` 将追踪上下文传入 LangGraph。
- [x] 为检索、工具调用、验证、合规和响应编排节点补充 span。
- [x] 记录模型名、耗时、token 用量、重试次数、检索数量、验证结果和合规结果等结构化 metadata。
- [x] 默认禁止发送以下内容：
  - [x] 原始用户问题
  - [x] 模型完整回答
  - [x] 文档内容、引用原文和 chunk 文本
  - [x] 工具原始参数、SQL、客户 ID、持仓明细和 PII
- [x] 如需调试内容，只允许开发环境显式开启，并经过统一脱敏器处理。
- [x] Langfuse 未配置、超时、鉴权失败或服务不可用时，业务链路继续完成；错误只写本地日志和指标。
- [x] 保留现有职责边界：
  - [x] Langfuse：Agent / LLM 运行链路、耗时、token 和成本观察
  - [x] SQLite：权限、引用、合规和业务审计
  - [x] Prometheus：QPS、延迟、错误率和缓存指标
- [x] 在 `.env.example`、`docker-compose.yml` 和 `README.md` 补充配置与运行说明。

### 测试与验收

- [x] Langfuse 关闭时，问答和现有测试行为不变。
- [x] Langfuse callback 能从 API 请求传播到 LangGraph、LLM 和工具调用。
- [x] 测试确认原始问题、回答、文档片段、SQL、客户信息和持仓数据不会进入 Langfuse payload。
- [x] 模拟 Langfuse 网络超时、鉴权失败和写入异常，问答仍返回正确业务结果。
- [x] 验证 trace 与内部 `request_id` 可关联，但不会暴露用户身份。
- [x] 增加 Langfuse 不可用时的本地告警或 Prometheus 计数。
- [x] 运行 `uv run python -m pytest -q` 和 `uv run python -m ruff check .`。
- [x] 更新面试演示：展示一次请求的 Agent 链路、节点耗时、模型调用次数和 token 成本；原文仍只在受控本地审计系统查看。

### 设计决策

- [ ] 先确认使用 Langfuse Cloud 还是自托管实例。
- [ ] 先确认真实环境的数据驻留、网络出口和密钥管理要求。
- [x] 明确 trace metadata 的允许字段白名单，禁止通过“新增字段”绕过脱敏策略。
  已落地：`LangfuseTraceMetadata` 白名单（TypedDict）+ `filter_metadata` 标量过滤，
  见 `src/utils/langfuse_adapter.py:72-92` 与 `src/utils/langfuse_adapter.py:421`；
  未登记的键在入口即丢弃，新增字段必须修改 adapter 源码才能通过。
- [x] 明确采样策略：开发环境可全量，生产环境按比例采样并保留错误请求。
  已落地：`LANGFUSE_SAMPLE_RATE` 在 adapter 请求边界做头部采样，错误请求不参与采样、
  未采样的失败请求在收尾时补建错误 trace，见 `src/utils/langfuse_adapter.py:445` 与
  `src/utils/langfuse_adapter.py:525`；SDK 采样固定 1.0，采样决策集中在 adapter。

记录日期：2026-09-24
