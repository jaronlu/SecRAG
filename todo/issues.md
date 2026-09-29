# SecRAG 待办问题登记处

> 本文件为立项待办登记处。当前批次：**2026-09-29 全链路演练优化（ISSUE-21 ~ ISSUE-28，全部 open）**。
> 上一批次（LLM 响应延迟审计，ISSUE-9~20，12 项）已全部修复并清理，
> 修复提交（一题一提交，TDD）：0964a5a / b9f084a / cfc970f / d1f9e2a / c1d7f2a /
> 6c11b2f / 79c93d0 / d97706a / 8739ee8 / ac3a3ab / c579b93（ISSUE-20 机制与实测见
> [todo/model-tiering-report-20260929.md](./model-tiering-report-20260929.md)）。
> 历史批次（2026-09-25 共 13 项、2026-09-28 演练 ISSUE-1~4、2026-09-28 晚 ISSUE-5~8）
> 均已修复，记录见 git 历史（446c9f1 / d96b35b / 457286c 等）。

---

## 2026-09-29 全链路演练实测证据

演练方式：8001 现役服务（Ark deepseek-v4-flash）+ 2026-09-28 重建数据
（Chroma 30,301 chunks / financial.db 15 行），`Authorization: Bearer demo-advisor`，
`scripts/demo.py` 严格断言 + 自建真实数据场景，节点瀑布取自 `data/audit.db`。

| 场景 | 结果 | 耗时 | 准确性 |
|---|---|---|---|
| S1 示例稳健增利产品风险等级（demo 断言） | 通过 | 30.8s | R2 正确、5 引用、数字验证 5/5、幻觉 0.17 |
| S2 权限拒绝（内部制度，demo 断言） | 通过 | 8.4s | 干净拒绝：citations=[]、permission_denied、low |
| S3 茅台 2026H1 营业收入/归母净利 | 通过（有口径缺陷） | 20.1s | 净利 445.17 亿正确；营收答 922.78 亿属**营业总收入**误标为营业收入（见 ISSUE-22） |
| S4 宁德 2026 EPS 预测 | **3/3 fail-closed** | 46.9/46.8/59.3s | EPS=20.83 在库却未被召回 → 幻觉检测 100% → 拒答（见 ISSUE-21） |
| S5 流式（S1 同题） | 完成 | 首字 38.3s / 总 60.0s | 125 个 answer_delta，token 级流式正常 |

**结构性发现（audit.db node_timings，10 个请求）**：

- `query_understand`（ISSUE-11 合并后的理解+规划单次 LLM 往返）**5.90–16.99s**，
  是首字与端到端的第一大头；同模型基准 small 档总时长仅 2.05s（model-tiering-report §1）。
- `planner` ≤0.3s（已无 LLM 调用）、`retrieve` ≤0.81s、`grade_and_filter` ≤0.7s
  —— 检索侧不是瓶颈，LLM 节点占 95%+。
- 多跳回环二次规划 **4/10 请求**触发，第二轮 9.26–14.64s。
- `reason` 两轮 **5/9 出答案请求**（每轮 3.4–22.3s）；两轮只可能由首轮 verify 失败触发。
- 延迟对比批次前基线（74.1/87.8s）已明显改善，但距设计线端到端 P95 ≤10s 仍差 2–6 倍。

**设计 SLO 引用**（`docs/design/README.md` §性能、`docs/design/implementation-08-evaluation.md` §2）：
端到端 P95 ≤10s、单源检索 P95 ≤3s、3-5 源 ≤5s、Recall@5 ≥80%、Recall@10 ≥90%、
结构化数字精确率 100%、引用准确率 ≥95%、幻觉率 ≤5%、缓存命中率 >80%。
**注意**：设计中**没有**流式首字（TTFT）独立指标；上一批次验收里的"首字 ≤5s"是批次内部
目标，需在设计中显式立项或废弃（见 ISSUE-23）。

---

## 2026-09-29 批次：全链路演练优化（ISSUE-21 ~ ISSUE-28）

### ISSUE-21 (P0) 入库按解析元素碎片化，设计分块尺寸从未生效 → 召回崩塌

- **现象**：30,301 个 chunk 中 **65.6% 长度仅 1–10 字**、87.9% ≤30 字、≥300 字的只有
  20 条（0.1%）；按 `chunk_hash` 去重后有 **43.7% 的重复行**。设计要求的
  研报 500/100、公告 300/50、财报 800/200（impl-01 §4.2）在 PDF/HTML 数据上完全未生效。
- **根因**：`UnstructuredLoader.load()` 按元素（Title/NarrativeText/UncategorizedText…）
  返回**一个 Document 一个元素**（`src/ingestion/loaders.py:29-36`），这些元素被直接送入
  分块器（`src/ingestion/pipeline.py:127-129`），而 `RecursiveCharacterTextSplitter.split_documents`
  对每个 Document **独立切分**（`src/ingestion/chunkers.py:87-88`）——短元素原样通过，
  chunk_size 永不绑定，元素之间也从不合并。
- **直接影响（S4 根因）**：`report_search` 占 30,181/30,301 chunks。对 S4 三条检索查询
  实测 top-20 **全部是 5–11 字的重复页眉/表头碎片，score 全为同一值 0.7090**
  （"宁德时代新能源科技股份有限公司 2026 年半年度报告全文"、"2026E"、"每股收益"），
  含真实数值的 chunk（国信研报 `EPS为20.83/25.96/…`）永远进不了 top_k=5 → 模型无证据
  写数字 → 幻觉检测 100% → fail-closed。同理，年报正表的**标签与数值被切到不同 chunk**
  （chunk 80 是"营业收入 利润总额 …"标签行，chunk 81 是"90,703,260,964.48 …"数值行）。
- **违反设计**：impl-01 §4.2 分块表、"chunk 边界不在标题、表格行、财务指标解释中间硬切"、
  风险表"表格丢失 → 用 Unstructured/Camelot 提取表格"。
- **修复方向**：①入库前按文档聚合元素（保留 category 顺序）后再按 doc_type 应用设计尺寸；
  ②过滤 `Header`/`Footer`/`EmailAddress` 元素（现网 450+414+2 条噪声进索引）；
  ③表格类元素保留结构、表头与单位行同 chunk；④补 chunk 长度分布断言测试。
- **验收**：chunk 长度分布落在设计区间（研报/公告/财报各自区间）；Recall@5/10 可测且达标
  （依赖 ISSUE-28）；S4 类"研报预测数字"问题不再 fail-closed。

### ISSUE-22 (P0) 数字口径混淆：营业总收入被标成营业收入，一手来源未优先

- **现象**：S3 问"贵州茅台2026年半年度报告**披露的营业收入**和归母净利润"，答
  "营业收入 **922.78 亿元**（同比 +1.30%）"，引用券商跟踪报告。
- **证据**：922.78 亿是**营业总收入**——茅台年报正文原文即"上半年，公司**营业总收入**
  922.78亿元，同比增长1.3%"；而年报"主要会计数据"正表（chunk 80/81）的**营业收入**
  是 `90,703,260,964.48` 元 = **907.03 亿元**。差额来自利息收入等口径。
  券商原文写"营收922.78亿元"用词含混，答案把它解析成了"营业收入"。归母净利润
  445.17 亿正确，但同比（-1.95%）在源文中有、答案表中留空。
- **违反设计**：08-evaluation §2"结构化数字精确率 100%"、"引用准确率 ≥95%"。
- **修复方向**：①数字校验绑定**口径标签**（营业总收入 / 营业收入 / 扣非归母净利不得互换）；
  ②同一问题命中一手来源（公告、财报原文）时优先于研报转述；③回答必须标明口径全称。
- **验收**：同口径问题的数字与一手来源一致；口径标签与数值成对校验有测试覆盖。

### ISSUE-23 (P0) query_understand 单次往返 5.9–17.0s，是首字与端到端第一大头

- **现象**：10 个请求的 `query_understand` 实测 5.90 / 7.29 / 8.24 / 10.63 / 11.24 /
  11.69 / 13.59 / 13.64 / 14.73 / 16.99s；而 `scripts/benchmark_models.py` 用同模型
  同端点的 small 档（max_tokens 256）总时长只有 **2.05s**——生产调用慢 3–8 倍。
- **疑点**：`plan_max_tokens=1024`（`src/config.py:84`）+ 较长 prompt（意图/实体/重写/歧义/
  检索计划五段 JSON schema 与规则文本，`src/agents/nodes.py:568-640`）。prefill 与
  completion 各自占多少**没有计量**，无法判断该压 prompt 还是压输出预算。
- **修复方向**：①按请求记录 prompt_tokens / completion_tokens（现无计量）；②压缩 schema
  与规则文本、收紧 `plan_max_tokens`；③评估角色级系统提示复用与 plan-only 轮的精简 prompt。
- **验收**：单轮 query_understand ≤3s；端到端 P95 ≤10s。
- **附带**：设计无 TTFT 指标，"首字 ≤5s"需在设计中显式立项或废弃（本批次实测首字 38.3s）。

### ISSUE-24 (P1) 多跳回环二次规划成本 9.3–14.6s，低召回场景高频触发

- **证据**：10 个请求中 **4 次**出现两轮 `query_understand`（量化交易 37.7s、示例稳健
  22.9s、货币基金 43.7s、S4 59.3s），第二轮 9.26–14.64s。触发条件为
  `should_retry_retrieval` 中 `len(usable) < RETRIEVAL_SUFFICIENT_RESULTS(2)`
  （`src/agents/graph.py:192-217`）。ISSUE-21 的碎片化导致 usable 数天然偏低，
  于是"低召回 → 重规划 → 仍低召回"空烧一轮 LLM。
- **修复方向**：①回环前判断新计划与首轮是否**实质不同**（源/查询/过滤器任一变化）；
  ②低召回优先扩 `top_k` 或换检索策略，而非重跑理解+规划；③回环仅在 0 召回时触发。
- **验收**：回环触发次数下降且 Recall 不退化；回环平均成本 ≤3s。

### ISSUE-25 (P1) reason 二次重推仍普遍，且中间验证结果无留痕

- **证据**：9 个出答案请求中 **5 次**出现两轮 `reason`（每轮 3.4–22.3s）。两轮 `reason`
  只能由首轮 verify 返回 `passed=false` 触发（`src/agents/graph.py:219-227`，
  `MAX_REASON_ATTEMPTS=2`）。S4 三例首轮确为真实无支撑（合理重推）；但 S1（示例稳健
  30.7s）首轮失败、重推后**通过且数字 5/5**，属可疑重推。
- **缺口**：`data/audit.db` 只保存**最终** verification 结果，无中间快照，无法区分
  "验证器误判（ISSUE-13 类残留）"与"真实无支撑"，也就无法回归验证 ISSUE-13 是否彻底。
- **修复方向**：审计/追踪补每轮 verification 快照（`failure_kind` + `issues` + 轮次）；
  依据快照定位误判类重推并清零。
- **验收**：能定位首轮失败原因；误判类重推为 0。

### ISSUE-26 (P1) 答案语义缓存默认关闭，且启用条件已失去出处

- **证据**：`semantic_cache_enabled=False`（`src/config.py:69`）、
  `DEFAULT_CACHE_ENABLED=False`（`src/utils/semantic_cache.py:44`），注释写明
  "重新启用前需满足 issues.md 一.1 的条件"——但该条目已随 484e7aa 批次清理删除，
  当前文件已无此条件。设计线为"缓存命中率 >80%"（README §性能）。
- **修复方向**：二选一——①按 484e7aa 记录的条件补齐实现（缓存绑定身份与授权范围、
  客户上下文、规范化问题、上下文摘要哈希、知识库版本；只缓存明确允许的成功终态；
  命中仍执行会话保存与审计）；②把条件重新登记回本文件再决定是否启用。
- **验收**：启用条件可追溯；启用后同角色多会话隔离与审计语义正确，命中率达标。

### ISSUE-27 (P2) BGE reranker 仍未配置，语义重排能力缺失

- **证据**：`FlagEmbedding` 未安装 → `reranker_available()=False`，`grade_and_filter`
  标记 `reranker_status="unavailable"`（e2e TC-015）。设计要求明确：
  PRD §技术选型"Rerank = BGE-Reranker-v2-m3"（`docs/design/PRD.md:296`）、
  `architecture.md:584-585`（`RERANK_MODEL=BAAI/bge-reranker-v2-m3`）、
  impl-05 §3.5 Rerank Tool 语义排序优化（`implementation-05-financial-tools.md:361-375`）。
- **性质**：当前按 AGENTS.md 设计驱动 #5 走了"明确返回未配置"的合规路径（不冒充），
  但能力确实缺失。ISSUE-21 修复后，rerank 对表格类碎片的分辨率提升是关键收益点。
- **修复方向**：本地化 bge-reranker 并纳入依赖与预热；或明确写入设计为长期降级并标注影响。
- **验收**：E2E 中 rerank 实际执行、`grade_and_filter` 不再因缺失降级。

### ISSUE-28 (P2) 评估数据集不足，设计准入标准无法评估

- **证据**：检索评估集 `scripts/evaluate_retrieval.sample.json` 仅 **5 条**，
  设计要求 ≥100 条查询 + ≥30 条权限负例（08-evaluation §3.1）；答案集
  `scripts/evaluate_answers.dataset.json` 29 条。
- **影响**：Recall@5 ≥80%、Recall@10 ≥90%、端到端 P95 ≤10s、引用准确率 ≥95%、
  幻觉率 ≤5% 全部**无实测产出**，任何"已达标"表述都缺证据。
- **修复方向**：按 08-evaluation §3.1 建评估集（覆盖 product/regulation/report/faq、
  五角色正负例），跑通 `evaluate_retrieval.py` / `evaluate_answers.py` / `evaluate_compliance.py`。
- **验收**：产出与设计同版本的评估产物，可直接对照准入标准判定。

---

## 2026-09-29 批次复测证据（上一批次 P0+P1 落地后，端口 8001 新服务）

修复后基线对比（基线 74.1s / 87.8s，验证误判重跑 + 3-8 次串行 LLM 往返）：

| 查询 | 修复后 | 结构性变化 |
|---|---|---|
| 货币基金的风险等级是多少？ | 43.7s | 两跳、reason 单轮 16.2s、验证一次通过 |
| 示例稳健增利理财产品的风险等级是多少？ | 23.0s | 两跳、reason 单轮 3.4s |
| 什么是量化交易？（SSE） | 首字 35.3s / 188 个 answer_delta / 总 37.8s | token 级真流式生效 |

结构性改善（audit.db 瀑布佐证）：验证失败重跑减少（ISSUE-13）、回环不再为凑证据数
触发（ISSUE-12）、恒失败工具轮次消失（ISSUE-10/19）、reason 单轮 16.4-50.1s →
3.4-16.2s（ISSUE-13/14）。

**残余差距**：①多跳 + 长 reason 查询未达批次目标 P95 ≤25s；②设计线 P95 ≤10s
受 `query_understand` 前置支配（见 ISSUE-23）；③更快小模型等待账号侧 coding 端点扩容
（当前仅 deepseek-v4-flash 可用，其余候选 UnsupportedModel，见 model-tiering-report）。

---

## 备注：演练环境（非问题）

本机 `sh start.sh`（端口 8001）已可直接启动（2026-09-28 重建 .venv 修复搬迁残留的
入口脚本 shebang）；正式 QA 演练仍需完整环境前缀：
`NO_PROXY='*' no_proxy='*' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
API_REQUEST_TIMEOUT_SECONDS=300 uv run python -m uvicorn src.api.main:app --port 8001`：
系统代理死端口、bge 模型已本地化但需 OFFLINE 跳过 HF 在线校验、多跳推理需 300s 预算
（默认 60s 不够）。`LANGFUSE_CAPTURE_CONTENT=false` 下 Langfuse observations 的
input/output 为空属设计脱敏；验证细节查 `data/audit.db`（注意：`total_duration_ms`
在 `payload_json` 内，不在顶层列）。

另注意：起服务前确认目标端口无遗留旧进程（`lsof -tiTCP:8001`），否则旧代码会继续
响应且新实例绑定失败退出，验证结论会失真。

另注意：本机 venv 需 `uv sync --extra dev`（pytest-asyncio 声明在
`[project.optional-dependencies].dev`，默认 sync 不安装，缺失时 async 测试全挂）。
`tests/test_ingest_metadata.py` 曾因 2026-09-28 演练数据重建失配 5 例，
已对齐现行 per-file `.meta.json` sidecar 契约（1d167b0），全量 525 passed。
