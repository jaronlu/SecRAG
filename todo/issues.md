# SecRAG 待立项问题清单

> 本文件为立项待办登记处。上一批审查问题（2026-09-25，9 P1 + 4 P2）已全部修复并清理本文件。
> 本批问题由 2026-09-28 Langfuse 全链路真实演练发现；证据来自 `data/audit.db`、
> Chroma 元数据实测与 Langfuse traces，均可复核。

---

## ISSUE-1 存量向量库缺 `date_day` 字段，任何日期过滤检索必然 0 召回

- **优先级**: P1（迁移债）
- **类别**: 数据迁移 / 检索
- **发现日期**: 2026-09-28

**现象**：带时间范围的查询（如"贵州茅台2025年研报的核心观点是什么？"）检索环节 `total_chunks=0`，
最终返回 fail-closed 兜底答案。

**证据**（2026-09-28 实测）：

- 本地 Chroma collection `securities_docs`（28388 chunks，入库时间 2026-07-11）的
  `embedding_metadata` 全部 key 中**没有 `date_day`**，只有字符串型 `date`
  （全库值分布：`2026-03-21` 13751 条、`2026-03-10` 13395 条、`2026-05-25` 594 条、
  `2026-04-26` 488 条，另有 `2025`、`2024.0` 等脏格式）。
- 该问题对应 2026-09-25 战役记忆中的部署注意项："date_day 只写在新入库数据上，旧 Chroma
  数据需重跑入库，否则时间范围过滤静默查不到旧文档（fail-closed，非泄漏）"。本次演练确认该债已在实际负载中爆雷。

**根因**：date_day 数值化改造（2026-09-25，新增 `src/utils/dates.py`）只对增量入库生效，
本地存量 collection 未重建。

**修复方向**：全量重跑入库（`scripts/ingest.py` 重建 collection），或写一次性迁移脚本从现有
`date` 元数据补写 `date_day`（注意 `2025`、`2024.0` 等非 ISO 格式的解析）。

**验收标准**：`embedding_metadata` 中 `date_day` 覆盖率 100%（或明确记录可豁免的 doc 类型）；
带时间范围的研报查询能召回 600519/000001 研报 chunk。

---

## ISSUE-2 planner 把"报告期年份"映射为发布日期强过滤，研报发布日期晚于报告期时必然漏检

- **优先级**: P1（检索质量；ISSUE-1 修复后依然存在）
- **类别**: Agent planner / 检索语义
- **发现日期**: 2026-09-28

**现象**：query"贵州茅台2025年研报的核心观点是什么？"，planner 生成的检索计划为
`{source: report_search, filters: {$and: [{date_day: {$gte: 20250101}}, {date_day: {$lte: 20251231}}]}, top_k: 5}`。
但贵州茅台 2025 年报研报（`600519_2025_2026.pdf`）的发布日期是 **2026-05-25**
（`date` 元数据，594 chunks；chunk 正文落款"2026 年 05 月"），落在过滤区间之外。
000001 研报同理（`date`=2026-04-26）。即使 ISSUE-1 修复补上 `date_day`，此查询仍为 0 召回。

**证据**：

- `data/audit.db` `audit_entries` 2026-09-28T01:01:29Z 记录：`retrieval.plan[0].filters`
  即上述过滤条件；`total_chunks=0, filtered_chunks=0, sources=[]`；
  `reasoning.execution_path` 显示 planner→retrieve→grade_and_filter 回环 3 次（141.7s）仍未放宽条件。
- Langfuse 同一 trace：`report_search` 调用 10 次、`route_reason_model` 多次重试，全链路可见。

**根因**：planner 把问题中的年份（报告期/标题年份）直接翻译为发布日期硬过滤。
研报的发布日期通常晚于报告期（2025 年报 2026 年发布），语义不等价。

**修复方向**（择一或组合）：

1. planner 对 `research_report` 类检索默认不加发布日期硬过滤，改用 stock_code/标题语义检索，
   日期只作排序偏好；
2. 年份 N 映射为发布日期宽区间 `[N-01-01, N+2-12-31]`；
3. 0 召回时自动去除日期过滤重试一次（复用 grade_and_filter→`should_retry_retrieval`
   现有回环，当前它不覆盖"元数据过滤导致空结果"的场景）。

**验收标准**：query"贵州茅台2025年研报的核心观点是什么？"能召回 600519 研报 chunk 并返回
带有效引用的成功答案（confidence ≥ medium）；补一条"标题年份 ≠ 发布日期"的检索测试。

---

## ISSUE-3 检索 0 结果时 reason model 仍无证据作答，靠验证层兜底

- **优先级**: P2（体验与成本；当前行为安全）
- **类别**: Agent 流程编排
- **发现日期**: 2026-09-28

**现象**：ISSUE-1/2 导致 0 召回后，reason model 仍基于参数知识生成带 `[来源N]` 标注的答案
（编造引用），最终由 `source_verification` 拦截（17 条 issues："引用来源 1/3/5/6/7/9 不存在"、
"答案包含引用标注但无检索结果"），返回"当前答案未通过来源或数字验证"兜底。

**评估**：验证层 fail-closed 拦截**工作正常**（安全边界按设计生效，本条不是安全缺陷）；
但流程上浪费了 141s 多跳推理与 8+ 次 LLM 调用，且对用户的兜底文案不如
"知识库中未找到相关资料"准确。

**修复方向**：retrieve/grade 后 `total_chunks=0` 时短路走"未找到相关资料"路径
（明确提示可调整时间范围等），不进入 reason 生成；或在 planner 0 召回重试仍为空时提前终止。

**验收标准**：构造日期过滤必空的查询，响应应为"未找到资料"类语义而非"验证未通过"，
且 Langfuse trace 中 `call_reason_model` 不再出现无证据生成轮次。

---

## ISSUE-4 demo.py 模糊提问场景与澄清行为不适配

- **优先级**: P3
- **类别**: 演练脚本
- **发现日期**: 2026-09-28

**现象**：`scripts/demo.py` 第一场景提问"这款理财产品风险等级是多少？"（无产品名），
LLM 走澄清追问路径（反问具体产品名，citations=[]），场景断言（必须含 R2、`[来源1]`、
citations ≥ 1）失败报 `DemoValidationError`。

**根因**：脚本问题设计未考虑 Agent 的澄清行为；用具体产品名（"示例稳健增利理财产品"）
提问即可全链路成功（2026-09-28 实测 23s 返回 5 条有效引用、compliance.passed=true）。

**修复方向**：demo 问题改为具体产品名，或断言兼容"澄清追问"响应形态。

---

## 备注：演练环境（非问题）

本机起服务需 `NO_PROXY='*' no_proxy='*' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
API_REQUEST_TIMEOUT_SECONDS=300 uv run python -m uvicorn src.api.main:app --port 8000`：
系统代理死端口、bge 模型已本地化但需 OFFLINE 跳过 HF 在线校验、多跳推理需 300s 预算
（默认 60s 不够）。`LANGFUSE_CAPTURE_CONTENT=false` 下 Langfuse observations 的
input/output 为空属设计脱敏；验证细节查 `data/audit.db`。
