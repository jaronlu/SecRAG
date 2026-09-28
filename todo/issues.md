# SecRAG 待立项问题清单

> 本文件为立项待办登记处。
> 2026-09-25 批次（9 P1 + 4 P2）与 2026-09-28 批次（Langfuse 全链路演练，4 项）均已修复并清理本文件。
> 2026-09-28 批次修复记录：ISSUE-1 63f171a（date_day 迁移）、ISSUE-2 22b17e9（planner 年份语义）、
> ISSUE-3 f544373（0 召回短路）、ISSUE-4 0ed0994+630b4fc（demo 场景）；E2E 与审计复核见各提交说明。

---

## 备注：演练环境（非问题）

本机起服务需 `NO_PROXY='*' no_proxy='*' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
API_REQUEST_TIMEOUT_SECONDS=300 uv run python -m uvicorn src.api.main:app --port 8000`：
系统代理死端口、bge 模型已本地化但需 OFFLINE 跳过 HF 在线校验、多跳推理需 300s 预算
（默认 60s 不够）。`LANGFUSE_CAPTURE_CONTENT=false` 下 Langfuse observations 的
input/output 为空属设计脱敏；验证细节查 `data/audit.db`。

另注意：起服务前确认 8000 端口无遗留旧进程（`lsof -tiTCP:8000`），否则旧代码会继续
响应且新实例绑定失败退出，验证结论会失真。
