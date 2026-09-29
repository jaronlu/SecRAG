# SecRAG 待立项问题清单

> 本文件为立项待办登记处。当前批次：**2026-09-29 LLM 响应延迟审计（ISSUE-9 ~ ISSUE-20，全部 open）**。
> 历史批次（2026-09-25 共 13 项、2026-09-28 演练 ISSUE-1~4、2026-09-28 晚 ISSUE-5~8）
> 均已修复并从本文件清理，记录见 git 历史（446c9f1 / d96b35b / 457286c 等）。

---

## 2026-09-29 批次：LLM 响应延迟审计

立项依据：2026-09-29 全链只读审计（4 路子代理分区排查 + 8001 现役服务实测 +
`data/audit.db` 节点瀑布还原）。结论：慢的根源是**串行 LLM 往返 3–8 次 × 验证误判
触发整段重跑 × 零 token 流式**，检索本身不是瓶颈。

**设计 SLO**（`docs/design/PRD.md` §4.2、`docs/design/implementation-08-evaluation.md`）：
端到端 P95 ≤ 10s、单源检索 ≤3s、多源 ≤5s、≥50 QPS。

**实测基线**（2026-09-29，`Authorization: Bearer demo-advisor`）：
- "货币基金的风险等级是多少？" → 74.1s，验证失败 fail-closed 拒答（confidence=low）
- "示例稳健增利理财产品的风险等级是多少？" → 87.8s，同上
- 节点瀑布：query_understand 3.2–4.5s、planner 3.5–10.4s ×2、retrieve+grade 0.3–1.0s、
  reason 16.4–50.1s ×2（第一轮均失败重跑）。LLM 节点占 95%+。
- 历史审计同构（66.6s / 92.0s 行），检索从未超过 5.4s。

**批次验收总目标**：端到端 P95 从 60–90s 降至 ≤25s（P0+P1 落地后复测），
流式首字 ≤5s；远期对齐设计线 ≤10s 需 ISSUE-20 支撑。

### ISSUE-9 (P0) token 级流式输出

- **现象**：SSE 端点只发节点级 progress，answer 终态整包一次到（`src/api/main.py:727-744`）；
  前端拿到全文后再以 10ms/字符假打字机渲染（`frontend/src/pages/ChatPage.tsx:113-129`）。
  用户感知延迟 = 全链路总时长。
- **证据**：3 处 LLM 调用均为阻塞 `invoke`（`src/agents/nodes.py:520/676/1053`）。
- **修复方向**：reason 节点改 `astream` 产出 token 级增量；SSE 协议新增 answer_delta 事件
  （保留既有 progress/answer 事件兼容）；前端 ChatPage 接真流并移除假打字机。
- **验收**：首字时间 ≤5s；非流式端点行为不变；契约测试覆盖新事件。

### ISSUE-10 (P0) reranker 从未生效：FlagEmbedding 缺失

- **现象**：`FlagEmbedding` 未安装且 pyproject 未声明 → `src/tools/rerank.py:30-34`
  每次必然失败，`grade_and_filter` 每跳 ok=False（实测两跳全失败），置信度被封顶 medium
  （`src/agents/nodes.py:1420-1431`），恒失败的 `rerank_tool` 仍暴露给 LLM 反复调用
  （`src/agents/tools.py:160`）。
- **性质**：违反设计"未实现能力不得用降级行为冒充"（AGENTS.md 设计驱动 #5）；
  证据质量下降连锁导致 reason 更多轮、验证更易失败。
- **修复方向**：安装 `FlagEmbedding`（bge-reranker 本地化）或启动时 fail-fast；
  生效前摘除 `rerank_tool`。
- **验收**：E2E 中 rerank 实际执行且 grade_and_filter ok=True；置信度不再被 reranker
  缺失封顶。

### ISSUE-11 (P0) 合并 query_understand 与 planner 为一次 LLM 往返

- **证据**：两节点串行强依赖、prompt 均小（`src/agents/nodes.py:498-518/652-674`），
  合计一次往返 6.7–14.9s（实测）。
- **修复方向**：单次调用同时产出意图/实体/重写与检索计划；澄清分支逻辑保留。
- **验收**：多跳路径 LLM 调用总数减 1；澄清与拒绝路径回归测试全绿。

### ISSUE-12 (P0) 放宽多跳回环触发阈值

- **证据**：`should_retry_retrieval` 要求可用结果 ≥ `CONFIDENCE_HIGH_MIN_RESULTS=3`
  才停（`src/agents/graph.py:210`、`src/schemas/constants.py:334`）——已有 1–2 条高质量
  结果也强制再跑一轮 planner（+3.5–10.4s）。
- **修复方向**：阈值降为 1–2 或由 planner 自判充分性；保持 0 结果短路（f544373）不变。
- **验收**：audit 瀑布中"已有可用结果仍回环"的轮次消失；引用质量无回归。

### ISSUE-13 (P0) 验证器词面误判触发 reason 整段重跑

- **现象**：实测两条查询 reason 第一轮均失败（50.1s / 35.1s）后重跑，最终仍 fail-closed。
  `ComprehensiveVerifier` 纯词面（数字边界匹配 + token 重叠 >0.5 即判幻觉
  `src/utils/verifier.py:196/244-291/293-300`），对 markdown 格式改写/同义表述脆。
- **性质**：同时是延迟（最坏调用翻倍）与正确性（可用答案被拒）问题。
- **修复方向**：幻觉检测引入证据集结构化对齐（如按句子/实体归一化后比对）；
  重跑前区分"格式不符"（可局部修复）与"事实缺失"（需重检索）。
- **验收**：`scripts/evaluate_answers.py` 离线集上误杀率下降且不放过真实幻觉；
  E2E 拒答率下降可测。

### ISSUE-14 (P1) LLM 客户端显式 max_tokens 与 max_retries

- **证据**：全仓未设置两参数 → langchain-openai 默认 max_retries=2，
  单 invoke 故障最坏 30s×3=90s（`llm_timeout_seconds=30`，`src/config.py:70`）。
- **修复方向**：understand/planner 设小 max_tokens 预算；max_retries=1；
  超预算输出走既有失败路径。
- **验收**：注入故障演练下单 invoke 最坏耗时 ≤60s；正常路径输出不受损。

### ISSUE-15 (P1) embedding 模型进程级单例

- **证据**：`embedder.get_embedding_model` 无任何缓存（`src/ingestion/embedder.py:77-101`），
  每跳/每工具调用重建 `HuggingFaceEmbeddings` → SentenceTransformer 重新加载权重
  0.11–0.44s × 2–6 次/请求；BM25 已有 `_bm25_cache` 先例（`src/retrieval/bm25_retriever.py:28`）。
- **修复方向**：模块级单例（对齐 reranker/rerank.py 的做法）。
- **验收**：单次多跳请求内模型加载次数 = 1（日志/审计可证）。

### ISSUE-16 (P1) 启动预热：冷启动不应落在首请求

- **证据**：无 lifespan/on_event；首请求同步建图 + torch import ~10s + BM25 全索引构建
  10–30s（jieba 分词 30,301 chunks）+ reranker 加载（`src/api/main.py:284-291`、
  `src/retrieval/bm25_retriever.py:120-135`）。
- **修复方向**：lifespan 启动时后台预热图/BM25/向量引擎/reranker。
- **验收**：重启后首个请求延迟与稳态请求同量级。

### ISSUE-17 (P1) 超时后真正取消图执行

- **证据**：非流式 `asyncio.wait_for(to_thread(...))` 超时不取消图线程（注释自认，
  `src/api/main.py:558-561/635-636`）→ 504 后孤儿任务继续烧 LLM 调用，客户端重试叠加。
- **修复方向**：利用既有 deadline 检查点（`STATE_REQUEST_DEADLINE`）在超时时置位
  cooperative cancel；或改结构化并发可取消执行。
- **验收**：超时后 audit 中该 request 无新增 LLM 轮次。

### ISSUE-18 (P1) SQLite WAL + 连接复用 + async 路由同步 IO

- **证据**：conversation/audit/portfolio/registry 均无 WAL，每操作新建连接 + 重放 DDL
  （每请求约 36 条 DDL；`src/utils/conversation.py:62-63` 等 11 处）；async 路由内
  同步 SQLite/health `count()` 阻塞事件循环（`src/api/main.py:114-117/319/337/357/375/490`）。
- **性质**：单请求影响小，50 QPS 目标下是尾延迟放大器。
- **修复方向**：各库开 WAL + busy_timeout；连接/DDL 收敛到单次初始化；
  会话 CRUD 与 health 探针移出事件循环。
- **验收**：`scripts/load_test.py` 达 50 QPS 且错误率 <1%（注意其 httpx timeout=15s
  硬编码需同步放宽到 SLO 预算）。

### ISSUE-19 (P2) 行情工具恒失败熔断

- **证据**：`financial.db` 无 market_history/market_snapshot 表且 baostock 未安装 →
  行情工具每次立即 RuntimeError → 熔断 60s（`src/tools/market_data.py:44-58/120-122`、
  `src/agents/nodes.py:1104-1111`），浪费 LLM 轮次。
- **修复方向**：按设计二选一——补齐数据源；或从工具清单摘除并返回"未配置"。
- **验收**：请求路径不再出现恒失败工具轮次。

### ISSUE-20 (P2) Ark 模型分层选型（reason 单轮 16–50s 本身压降）

- **证据**：reason 单轮 16.4–50.1s 是绝对大头；understand/planner 小任务也占秒级–十秒级。
  达到设计线 P95 ≤10s 必须压降单轮生成时长。
- **修复方向**：实测 Ark 各候选模型 TTFT/吞吐后分层——小任务（understand/planner）
  用更快小模型；reason 视质量/延迟权衡选型；配合 ISSUE-14 的 max_tokens 预算。
- **验收**：选型报告（TTFT/吞吐/质量对比）+ 分层配置落地；端到端 P95 复测对齐 ≤10s。

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
