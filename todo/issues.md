# SecRAG 待立项问题清单

> 本文件为立项待办登记处。当前无待处理问题。
> 2026-09-25 批次（9 P1 + 4 P2）、2026-09-28 演练批次（ISSUE-1~4）、
> 2026-09-28 晚批次（ISSUE-5~7，另含 ISSUE-8 原子提交清理）均已修复并清理本文件。
> 2026-09-28 晚批次修复提交：446c9f1（ISSUE-5 流式进度节点名对齐）、
> d96b35b（ISSUE-6 demo 场景 2 拒绝可达：改问 advisor 角色外的内部制度内容）、
> 457286c（ISSUE-7 legacy UI 删除 + dist 缺失 fail-fast）。

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
