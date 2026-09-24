# TODO

## Langfuse 真实数据场景接入

目标：为 SecRAG 增加 Langfuse Agent 链路追踪，同时保证真实金融数据不被发送到第三方观测平台，且 Langfuse 故障不影响问答、审计和合规流程。

### 实施任务

- [ ] 增加 Langfuse 依赖并更新 `uv.lock`。
- [ ] 增加配置项：
  - [ ] `LANGFUSE_ENABLED`
  - [ ] `LANGFUSE_HOST`
  - [ ] `LANGFUSE_PUBLIC_KEY`
  - [ ] `LANGFUSE_SECRET_KEY`
  - [ ] `LANGFUSE_SAMPLE_RATE`
  - [ ] `LANGFUSE_CAPTURE_CONTENT`，默认关闭
- [ ] 新增统一 Langfuse adapter，封装 client、callback、trace 和 span 创建。
- [ ] 在 API 请求入口创建根 trace，并关联内部 `request_id`、`thread_id` 和执行状态。
- [ ] 通过 `RunnableConfig.callbacks` 将追踪上下文传入 LangGraph。
- [ ] 为检索、工具调用、验证、合规和响应编排节点补充 span。
- [ ] 记录模型名、耗时、token 用量、重试次数、检索数量、验证结果和合规结果等结构化 metadata。
- [ ] 默认禁止发送以下内容：
  - [ ] 原始用户问题
  - [ ] 模型完整回答
  - [ ] 文档内容、引用原文和 chunk 文本
  - [ ] 工具原始参数、SQL、客户 ID、持仓明细和 PII
- [ ] 如需调试内容，只允许开发环境显式开启，并经过统一脱敏器处理。
- [ ] Langfuse 未配置、超时、鉴权失败或服务不可用时，业务链路继续完成；错误只写本地日志和指标。
- [ ] 保留现有职责边界：
  - [ ] Langfuse：Agent / LLM 运行链路、耗时、token 和成本观察
  - [ ] SQLite：权限、引用、合规和业务审计
  - [ ] Prometheus：QPS、延迟、错误率和缓存指标
- [ ] 在 `.env.example`、`docker-compose.yml` 和 `README.md` 补充配置与运行说明。

### 测试与验收

- [ ] Langfuse 关闭时，问答和现有测试行为不变。
- [ ] Langfuse callback 能从 API 请求传播到 LangGraph、LLM 和工具调用。
- [ ] 测试确认原始问题、回答、文档片段、SQL、客户信息和持仓数据不会进入 Langfuse payload。
- [ ] 模拟 Langfuse 网络超时、鉴权失败和写入异常，问答仍返回正确业务结果。
- [ ] 验证 trace 与内部 `request_id` 可关联，但不会暴露用户身份。
- [ ] 增加 Langfuse 不可用时的本地告警或 Prometheus 计数。
- [ ] 运行 `uv run python -m pytest -q` 和 `uv run python -m ruff check .`。
- [ ] 更新面试演示：展示一次请求的 Agent 链路、节点耗时、模型调用次数和 token 成本；原文仍只在受控本地审计系统查看。

### 设计决策

- [ ] 先确认使用 Langfuse Cloud 还是自托管实例。
- [ ] 先确认真实环境的数据驻留、网络出口和密钥管理要求。
- [ ] 明确 trace metadata 的允许字段白名单，禁止通过“新增字段”绕过脱敏策略。
- [ ] 明确采样策略：开发环境可全量，生产环境按比例采样并保留错误请求。

记录日期：2026-09-24
