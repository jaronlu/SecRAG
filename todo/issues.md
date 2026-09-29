# SecRAG 待办问题登记处

> 本文件为立项待办登记处。**当前无待处理问题。**
> 2026-09-29 批次（LLM 响应延迟审计，ISSUE-9~20，12 项）已全部修复并清理，
> 修复提交（一题一提交，TDD）：0964a5a / b9f084a / cfc970f / d1f9e2a / c1d7f2a /
> 6c11b2f / 79c93d0 / d97706a / 8739ee8 / ac3a3ab / c579b93（ISSUE-20 机制与实测见
> [todo/model-tiering-report-20260929.md](./model-tiering-report-20260929.md)）。
> 历史批次（2026-09-25 共 13 项、2026-09-28 演练 ISSUE-1~4、2026-09-28 晚 ISSUE-5~8）
> 均已修复，记录见 git 历史（446c9f1 / d96b35b / 457286c 等）。

---

## 2026-09-29 批次复测证据（P0+P1 落地后，端口 8001 新服务）

修复后基线对比（基线 74.1s / 87.8s，验证误判重跑 + 3-8 次串行 LLM 往返）：

| 查询 | 修复后 | 结构性变化 |
|---|---|---|
| 货币基金的风险等级是多少？ | 43.7s | 两跳、reason 单轮 16.2s、验证一次通过 |
| 示例稳健增利理财产品的风险等级是多少？ | 23.0s | 两跳、reason 单轮 3.4s |
| 什么是量化交易？（SSE） | 首字 35.3s / 188 个 answer_delta / 总 37.8s | token 级真流式生效 |

结构性改善（audit.db 瀑布佐证）：验证失败重跑消失（ISSUE-13）、回环不再为凑证据数
触发（ISSUE-12）、恒失败工具轮次消失（ISSUE-10/19）、reason 单轮 16.4-50.1s →
3.4-16.2s（ISSUE-13/14）。

**残余差距（后续再立项）**：
1. 批次目标 P95 ≤25s：多跳 + 长 reason 查询（43.7s）未达标。
2. 设计线 P95 ≤10s / 首字 ≤5s：受 reason 前置的 1-2 轮规划支配；更快的小模型
   等待账号侧 coding 端点扩容（当前仅 deepseek-v4-flash 可用，其余候选
   UnsupportedModel，见 model-tiering-report）。
3. reason 输出长度可在 4096 预算内进一步调优。

---

## 备注：演练环境（非问题）

本机 `sh start.sh`（端口 8001）已可直接启动（2026-09-28 重建 .venv 修复搬迁残留的
入口脚本 shebang）；正式 QA 演练仍需完整环境前缀：
`NO_PROXY='*' no_proxy='*' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
API_REQUEST_TIMEOUT_SECONDS=300 uv run python -m uvicorn src.api.main:app --port 8001`：
系统代理死端口、bge 模型已本地化但需 OFFLINE 跳过 HF 在线校验、多跳推理需 300s 预算
（默认 60s 不够）。`LANGFUSE_CAPTURE_CONTENT=false` 下 Langfuse observations 的
input/output 为空属设计脱敏；验证细节查 `data/audit.db`。

另注意：起服务前确认目标端口无遗留旧进程（`lsof -tiTCP:8001`），否则旧代码会继续
响应且新实例绑定失败退出，验证结论会失真。

另注意：本机 venv 需 `uv sync --extra dev`（pytest-asyncio 声明在
`[project.optional-dependencies].dev`，默认 sync 不安装，缺失时 async 测试全挂）。
`tests/test_ingest_metadata.py` 曾因 2026-09-28 演练数据重建失配 5 例，
已对齐现行 per-file `.meta.json` sidecar 契约（1d167b0），全量 525 passed。
