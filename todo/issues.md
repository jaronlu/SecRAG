# SecRAG 待立项问题清单

> 本文件为立项待办登记处。
> 2026-09-25 批次（9 P1 + 4 P2）与 2026-09-28 批次（Langfuse 全链路演练，ISSUE-1~4）均已修复并清理本文件。
> 2026-09-28 晚登记（UI 修复后续 + 演练发现）：ISSUE-5~7 待处理；
> ISSUE-8（原子提交清理）同批修复：9426c48（gitignore fetch state）、09ee972（docs/design 停止跟踪）、
> e21da8f（markdown 渲染修复）、63a7d66（2026 演练数据刷新）。

---

## P1 — 演练前建议处理

### ISSUE-5 流式问答 7 步进度指示器不亮（演示体验）

- **症状**：UI 提问后 "AI 思考中..."，7 步（查询理解→组织回答）全程灰色，服务端完成后答案才一次性出现。实测 111s；audit.db 显示 2026-09-28T14:52:44Z 已完成（5 citations、confidence=medium），UI 侧无任何中间反馈。
- **方向**：`/v1/assistant/qa/stream` 的 SSE 进度事件要么后端没发（StreamingResponse 生成器），要么前端 `api.ts` 的 fetch stream 只处理了最终答案事件、没消费进度事件。与 markdown 渲染是两个独立问题（markdown 渲染已修复，见 todo 外的 frontend 改动）。
- **涉及**：`src/api/main.py`（stream 端点）、`frontend/src/api.ts`、`frontend/src/components/StreamingProgress.tsx`
- **复现**：demo-advisor 提问"示例稳健增利理财产品的风险等级是多少？"，浏览器 Network 面板观察 `/qa/stream` 事件流。

### ISSUE-6 demo.py 场景 2 权限拒绝失配（演练阻塞，需三选一决策）

- **症状**：demo-tech 问"内部研究摘要里对新能源板块怎么看？"不再返回设计中的干净拒绝（"无权限" + citations=[] + flags=[permission_denied]）；3 次脚本重试 + 1 次手动均失败——或以公开内容作答，或 fail-closed 兜底文案。
- **根因**：权限是结果级过滤（`src/retrieval/hybrid_retriever.py` `_filter_results_by_role`），拒绝路由只在全部结果被拒时触发（`src/agents/graph.py` `should_retry_retrieval` 的 `results and not usable`）；演练数据重建抓入的 1,662 条券商研报 chunk 全是 public（内部摘要仅 8 条 internal），usable 恒非空 → 拒绝分支不可达。不是权限模型回归，是 demo 断言与新数据规模失配。
- **三选一**：① 换一个只存在于受限内容里的问题；② planner 对"内部研究摘要"类查询做 doc_type/permission 收敛（改动大，需设计讨论）；③ 放宽 `scripts/demo.py` 场景 2 断言接受公开回答。
- **证据**：`data/audit.db` audit_entries.payload_json；chroma `embedding_metadata`（doc_type=research_report 按 permission_level 分组：internal 8 / public 1,662）。

## P2 — 清理决策

### ISSUE-7 legacy UI 删除（评估已完成，待拍板）

- **结论**：可删，-2,400+ 行。React（ChatPage + AdminPage）功能覆盖 ⊇ admin.html；对 ui.html 唯一缺口是 raw JSON 调试视图；legacy 无流式。全仓库仅 `src/api/main.py`、`src/api/ui.py` 引用，tests/scripts/docs 零引用。
- **两步走**：① 先补 ChatPage raw JSON 调试面板（或接受损失，改用 curl/Langfuse）；② 删 `src/api/ui.html`、`src/api/admin.html`、`src/api/ui.py`，`main.py` 去掉 `/legacy` 路由与 legacy 兜底分支（dist 缺失改为 fail-fast 报错提示先 build），`start.sh` 去掉 Legacy 一行。
- **行为变化**：dist 缺失时从"静默兜底旧 UI"变为"启动报错"。

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
