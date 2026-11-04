# SecRAG 待办问题登记处

> 本文件为立项待办登记处。当前批次：**2026-09-29 全链路演练优化（ISSUE-21 ~ ISSUE-28）已全部修复**
> （一题一提交，TDD）：c3774a4 / aa1225d / 86e2339 / 4ecb13a / c76109a / 3e1094a /
> f68815b / 2a6dfe1，修复证据与未验证范围见下文「批次修复证据（2026-09-29 晚）」。
> 上一批次（LLM 响应延迟审计，ISSUE-9~20，12 项）已全部修复并清理，
> 修复提交：0964a5a / b9f084a / cfc970f / d1f9e2a / c1d7f2a /
> 6c11b2f / 79c93d0 / d97706a / 8739ee8 / ac3a3ab / c579b93（ISSUE-20 机制与实测见
> [todo/model-tiering-report-20260929.md](./model-tiering-report-20260929.md)）。
> 历史批次（2026-09-25 共 13 项、2026-09-28 演练 ISSUE-1~4、2026-09-28 晚 ISSUE-5~8）
> 均已修复，记录见 git 历史（446c9f1 / d96b35b / 457286c 等）。
> 最新批次：**2026-11-04 Knowledge QA MVP 实机测试（ISSUE-29~38）已登记，未修复**，
> 见下文「2026-11-04 Knowledge QA MVP 实机测试」。

---

## 2026-11-04 Knowledge QA MVP 实机测试（ISSUE-29 ~ ISSUE-38）

测试方式：浏览器黑盒实测（ZCode 内置浏览器，operations 身份，UI 全程真实操作）+ API 交叉验证
（curl：认证/threads/隔离/metrics/admin）+ 自动化矩阵复跑（双 profile pytest、56 契约、ruff、
frontend build，全部通过）。服务：8000 端口 uvicorn 直跑，Ark coding 端点。截图证据：
`artifacts/gui-test-20261104/`（截图通道中途故障，T6 之后以 DOM 快照 + audit.db 为证据）。

**结果总览：10 个 UI 测试点 → 通过 4（T1 初始页 / T2 answered+引用 / T4 追问消解 / T9 空输入），
失配 5，部分失配 1。API 安全边界全部通过（401×2、跨用户 thread 404、fail-closed 不伪装成功）。
8 类 MVP 契约场景实机只有 S1/S7 按预期终态呈现。**

| 场景（契约预期） | 实机终态 | 判定 |
|---|---|---|
| S1 产品风险等级（answered） | 已回答 + R2 + 引用 + medium（33s） | ✅ |
| S2 FAQ 密码重置（answered+faq_search） | **未找到资料（no_evidence）**（26s） | ❌ ISSUE-29 |
| S3 实时行情（no_evidence） | **已拦截（blocked）**（83s） | ❌ ISSUE-30 |
| S4 PE 歧义（needs_clarification） | 已回答（诚实拒答式回答）（36s） | ❌ ISSUE-29 根因 |
| S5 机密草稿（permission_denied） | 未找到资料（no_evidence）（~30s） | ❌ ISSUE-32 |
| S6 买入建议（blocked） | 已回答（合规自律回答）（79s） | ❌ ISSUE-31 |
| S7 同会话追问（answered×2） | 两轮均已回答，指代消解正确（27s×2） | ✅ |
| S8 服务不可用（transport_error） | 未实机触发（杀 provider 属破坏性操作，由契约测试覆盖）；超长输入 422 走了 alert 路径见 ISSUE-33 | ➖ |

### ISSUE-29（高，回归）：默认模型 glm-5.3-flash 烧穿 plan 预算，理解/规划 100% 退化兜底单源计划

> **已修复（fix: restore deepseek plan model and mark plan fallback in audit）**：
> ① `.env`/`.env.example` 默认模型改回 `deepseek-v4-flash`；② `query_understand` 与
> `_plan_only_llm_call` 退化兜底计划时写入 `STATE_PLAN_FALLBACK` 原因标记
> （budget_starved | invalid_json | invalid_plan），随审计 payload `query.plan_fallback`
> 落盘，兜底退化不再只在 uvicorn stdout 可见；③ 新增 6 个测试守护标记语义
> （test_agents.py：4 个 state 标记 + 2 个审计落盘）。验证：test_agents.py 161 passed
> （唯一失败为 ISSUE-38 既有环境依赖项）、ruff 全绿、basedpyright 无新增报错。
> **未验证**：真实 LLM 下不再出现 guard 告警需起服务实测（随 ISSUE-32 复测一并执行）。

- **现象**：服务端日志每次理解/规划调用都打出 guard 告警："理解/规划调用输出预算
  （plan_max_tokens=640）耗尽且正文为空（finish_reason=length），已退化为兜底检索计划"。
  audit.db 中 query 的 query_type/intent 恒为 "unknown"。
- **根因**：glm-5.3-flash 是强制 reasoning 模型且思考 token 计入 max_tokens（2026-11-03
  超时事故同一根因，当时已回退 deepseek-v4-flash 并同步 .env.example）。当前
  `.env` 与 `.env.example` 又配回 `OPENAI_MODEL=glm-5.3-flash`，**回归复发**。
- **后果链**（实测）：① S2 FAQ 问题 plan 只产出 product_search 单源 → faq 8 chunks 永不
  被检索 → no_evidence 误答；② S4 歧义澄清（needs_clarification 依赖理解节点输出
  ambiguity）在真实 LLM 下不可达；③ planner 回环 3 次每次退化同一兜底 plan，不换源，
  白烧 3×8~10s（S2 共 25.9s / S3 共 83.3s）。
- **修复方向**：.env/.env.example 默认模型改回 deepseek-v4-flash（或调大
  LLM_PLAN_MAX_TOKENS 并接受延迟）；guard 告警目前只在 uvicorn stdout，建议在审计
  payload 增加 plan_fallback 标记，让兜底退化在审计/指标层可见。
- **备注**：README 快速开始按 .env.example 配置的用户必然踩中；T2/S1 能答对纯属兜底
  plan（product_search）恰好命中产品语料。

### ISSUE-30（高）：验证 fail-closed 被映射为「已拦截」，语义误导

> **已修复（fix: map verification fallback to no_evidence and floor rerank relevance）**：
> ① `answer_status_from_state` 优先级改为 澄清 → 权限 → 无证据 → **blocked（仅合规
> 拦截）** → no_evidence（验证未通过兜底，含工具上限）→ answered——验证 fail-closed
> 不再冒充"已拦截"，blocked 只保留给合规层真实拦截的终态；② `grade_and_filter`
> 语义重排后按 `RERANK_MIN_RELEVANCE_SCORE=-6.0`（logit）丢弃明显无关候选并记日志。
> 阈值由本地权重实测校准：问题对真实相关证据 ≥ -1.8（含"适合买入吗"这类词面不匹配
> 问法），明显无关对 ≤ -9.3，-6.0 取间隙中点；**不能用 0.0**——会误杀 S6 类相关证据
> 使合规拦截场景退化成 no_evidence（实测踩中后校准）。③ 新增/更新测试：answer_status
> 3 例（验证失败→no_evidence、工具上限→no_evidence、验证+合规同时失败→blocked）、
> grade_and_filter 底线 2 例（丢弃无关保留相关、丢弃留痕）、S6 契约沿用 blocked 断言。
> 验证：受影响 4 文件 209 passed；全量 714 passed（7 个失败经 master 基线对照全部为
> 既有失败，与本修复无关）；ruff 全绿；basedpyright 无新增报错。
> **未验证**：S3 实机终态是否变为「未找到资料」需起服务复测（随 ISSUE-32 一并执行）。

- **现象**：S3 实时行情问题 → 兜底 plan 检回 1 个无关 product chunk（理财产品风险揭示）
  → reason 两轮各 35.5s/36.1s 强行作答 → 幻觉检测 50%/60% 两轮失败 → fail-closed 兜底
  文案，UI 显示「已拦截」。
- **问题**：① answer_status 映射把"验证未通过"归入 blocked（合规拦截语义）——审计里
  compliance_passed=1、flags 为空，实际并未触发合规规则，用户会误以为违规被拦；应为
  no_evidence 或独立的"验证未通过"语义。② grade_and_filter 对明显无关 chunk（上证行情
  vs 理财产品揭示书）无相关性底线拦截，任由无关证据进入 reason。
- **修复方向**：答案文本为验证兜底文案时终态映射改走 no_evidence；grade_and_filter 增加
  低分阈值丢弃（记录丢弃原因）。

### ISSUE-31（中）：合规拦截（blocked）在真实 LLM 下不可触发

- **现象**：S6 "适合买入吗？给我操作建议" → 模型自律地拒绝给时点建议并给出核实话流程
  → 答案合规 → answered（medium）。契约预期合规规则拦截 → blocked。
- **根因**：合规规则检查的是 LLM 生成的答案文本；真实模型规避了"建议买入"类表述，规则
  永不触发。mock 契约按剧本输出触发词，掩盖了该分歧。
- **修复方向**：合规层应在 query 理解/意图层增加确定性投资建议意图识别（不依赖生成文本），
  或接受"模型自律优先"并在设计文档中修正 S6 预期。需设计决策。

### ISSUE-32（高，安全演示失真）：permission_denied 终态在默认链路不可达

- **现象**：S5 机密重组草稿问题（allowed_roles 不含 operations）→ no_evidence 而非
  permission_denied。
- **根因**：ISSUE-29 兜底 plan 永远只查 product_search，research_report 受限源根本不进
  plan，"全部越权 → 权限拒绝"路径无输入。
- **影响**：产品核心卖点"权限不足以明确 answer_status 呈现"在默认配置下不可演示；
  权限过滤仍在（无泄露），但终态语义失真。ISSUE-29 修复后需复测本例。

### ISSUE-33（中）：前后端输入长度契约脱节 + 422 错误文案误导

> **已修复（fix: align frontend query limit and surface 422 copy）**：
> ① 前端 textarea `maxLength=500`（`MAX_QUERY_LENGTH` 常量对齐后端
> `AssistantQARequest.query max_length=500`）+ 输入框右下角字数计数
> （到上限变琥珀色）；② `api.ts` 新增 `toRequestError`：422 按 pydantic
> `detail[].type === "string_too_long"`（已实测确认）区分文案"输入超过 500 字
> 上限，请精简问题后重试"，其余校验错误给通用可操作文案，不再把可恢复的
> 输入错误渲染成"稍后重试"；③ `ChatPage.markFailed` 支持携带具体文案。
> 验证：`npm run build` 成功。**未验证**：实机超长输入的 UI 文案需随收尾
> 复测确认（前端项目无测试框架，未新增依赖）。

- **现象**：前端 textarea 无长度限制（maxLength=-1），5306 字符可输入并提交；后端
  QARequest `query max_length=500` 返回 422；前端把所有错误统一渲染为
  "请求未完成，请稍后重试。"（重试永远无效，误导用户）。
- **位置**：`src/schemas/request_response.py:25`、frontend `ChatPage.tsx:104` 错误分支。
- **修复方向**：前端 maxLength=500 + 字数提示；422 按 detail 区分文案（如"输入超过
  500 字上限"）。

### ISSUE-34（低）：引用展开不显示原文

> **已修复（fix: show citation quotes and mask source paths）**：
> ① 前端引用展开区渲染 `quote` 原文摘句（后端 CitationExtractor 一直产出该字段，
> 此前前端只读不存在的 `content` 字段），空时回退 `content` 兼容；② 服务端绝对路径
> 脱敏：`qa.py` 新增 `_sanitize_citation_sources`，REST 响应与 SSE answer 事件两个
> 出口的 `citation.source` 只保留文件名（内部引用/对账路径不动）；③ 新增 REST 出口
> 脱敏+透传断言测试。验证：test_api_routes.py 25 passed、`npm run build` 成功、
> basedpyright 0 错误。**未验证**：实机引用展开的 UI 呈现随收尾复测确认。

- **现象**：引用区展开后只有 `[1] product` + 源文件绝对路径，无 chunk 原文文本；README
  承诺"可展开查看引用原文"。运营人员无法就地核对引用内容。
- **备注**：源文件绝对路径直接暴露给前端（`/Users/ryan/...`），生产环境属信息泄露面。

### ISSUE-35（低）：进行中会话不出现在侧栏

> **已修复（fix: refresh thread list on auto-created threads）**：
> 会话列表加载提取为 `refreshThreads` 回调；answer 事件回写 `thread_id` 且本轮
> 此前无会话（自动创建）时立即刷新侧栏，条目带后端真实标题（query 前 100 字）。
> 已有会话的追问轮不重复刷新。验证：`npm run build` 成功。**未验证**：实机侧栏
> 呈现随收尾复测确认。

- **现象**：整轮 7 次问答期间侧栏一直"暂无会话"；点"+ 新建对话"后才出现条目。threads
  API 数据正常（两个 thread 都在）——纯前端列表刷新缺陷，历史会话在会话中不可找回。

### ISSUE-36（低，环境/运维）：Chroma 遗留空 collection + ingest 幂等跳过不校验向量库

- **现象**：`secrag_documents` 空壳 collection 残留（历史常量改名遗留，现行
  `CHROMA_COLLECTION_NAME="securities_docs"`）。ingest 对已登记文档按 path 哈希跳过，
  不校验向量库中 chunk 实际存在——若只清 chroma 目录而保留 registry，入库会全部跳过，
  检索静默为空（本次实测踩中前半段：0 chunks 状态下 ingest 全部跳过）。
- **修复方向**：清理空 collection；ingest 增加"registry 有记录但向量库无 chunk"的对账
  （至少 --full-scan 时重建）。

### ISSUE-37（低）：UI 推荐问题与 knowledge_qa 语料失配

- **现象**：欢迎页推荐问题为"货币基金风险等级 / 股票质押式回购新规 / 什么是量化交易"，
  当前演示语料为理财产品揭示书/FAQ/合规制度，点击必然 no_evidence，首触体验失真。

### ISSUE-38（低，测试封闭性）：双 profile 矩阵在本机 .env 下 1 例失败

- **现象**：`tests/test_agents.py::test_agent_ollama_client_ignores_environment_proxy`
  断言 `timeout: 30.0` 硬编码默认值，但 `_build_llm` 读 `config.llm.timeout`，本机
  `.env` 的 `LLM_TIMEOUT_SECONDS=90` 穿透测试（CI 无 .env 可过）。设
  `LLM_TIMEOUT_SECONDS=30` 复跑即通过，根因确认。
- **修复方向**：测试内 monkeypatch `config.llm.timeout`（或对 Settings 整体打桩），
  消除对宿主 .env 的隐式依赖。

### 实测通过项（留档）

- answered 主流程：R2 事实正确、结构化输出、引用标注 [来源N]、置信度呈现（T2/T4）。
- 追问指代消解：新对话内"它的适当性匹配要求"正确消解到理财产品并带引用（S7）。
- 输入防护：空输入/纯空白禁用发送。
- 安全边界（API）：无 token 401、伪造 token 401、跨用户读他人 thread 404（不泄露存在性）。
- fail-closed 全程未伪装成功：所有失败终态均无引用、无成功置信度。
- 流式反馈：思考中状态、输入禁用、光标动画正常。
- 自动化矩阵：双 profile pytest（除 ISSUE-38 外 194/195 通过）、API/契约 56 通过、
  ruff 全绿、frontend build 成功。
- 延迟： answered 26-36s / 拦截 83s，全部远超设计 SLO 10s；glm reason 单轮 35s+
  为主要大头（ISSUE-29 修复后需复测）。

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

> 状态：**8 项全部修复**（f68815b 为 ISSUE-27，2a6dfe1 为 ISSUE-28），
> 复测证据见文末「批次修复证据」。

### ISSUE-21 (P0) 入库按解析元素碎片化，设计分块尺寸从未生效 → 召回崩塌

> **已修复（c3774a4）**：`chunk_documents` 先按文档聚合元素（保留 category 顺序）、
> 过滤 Header/Footer/EmailAddress、Table 元素整块保留且切分时重复表头行、
> 文档内 ≥40 字符的重复 chunk 去重、按元素偏移回填 page_number；
> `CHUNKER_VERSION` 升到 v2 强制重入库。真实语料实测见「批次修复证据」。

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

> **已修复（aa1225d）**：新增 `CaliberVerifier` 做口径标签×数值成对校验
> （别名归一、单位换算容差、跨口径混用与凭空口径两类拦截）；
> `grade_and_filter` 在候选池同时含一手来源与转述时把公告/财报提到研报之前；
> reason prompt 要求口径全称并标注「一手来源/研报转述」。

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

> **已修复（86e2339）**：合并/重试 prompt 模板从约 446 tokens 压到约 289
> （预算常量 `PLAN_PROMPT_TOKEN_BUDGET=300` 由测试守护，JSON 契约字段不丢）；
> `llm_plan_max_tokens` 1024 → 384；计划调用按请求记录 prompt/completion tokens
> （进 Langfuse span 与审计 `node_timings`）；TTFT 已在设计中立项
> （README 性能行 + impl-08 §2 准入表，wiki 提交 24ba65f）。
> **未验证**：单轮 ≤3s 需起服务实测（当前会话未跑端到端）。

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

> **已修复（4ecb13a）**：`should_retry_retrieval` 区分 0 召回与非 0 低召回——
> 0 召回才重跑理解+规划，非 0 低召回走新 `widen` 路由直接重跑检索
> （top_k ×2，封顶 20，无 LLM 往返）；重试轮计划与上一轮实质相同
> （源/查询/过滤器指纹一致）时改用放宽版计划；路由表 `RETRIEVAL_RETRY_ROUTES`
> 由测试守护。**未验证**：回环平均成本 ≤3s 需起服务实测。

- **证据**：10 个请求中 **4 次**出现两轮 `query_understand`（量化交易 37.7s、示例稳健
  22.9s、货币基金 43.7s、S4 59.3s），第二轮 9.26–14.64s。触发条件为
  `should_retry_retrieval` 中 `len(usable) < RETRIEVAL_SUFFICIENT_RESULTS(2)`
  （`src/agents/graph.py:192-217`）。ISSUE-21 的碎片化导致 usable 数天然偏低，
  于是"低召回 → 重规划 → 仍低召回"空烧一轮 LLM。
- **修复方向**：①回环前判断新计划与首轮是否**实质不同**（源/查询/过滤器任一变化）；
  ②低召回优先扩 `top_k` 或换检索策略，而非重跑理解+规划；③回环仅在 0 召回时触发。
- **验收**：回环触发次数下降且 Recall 不退化；回环平均成本 ≤3s。

### ISSUE-25 (P1) reason 二次重推仍普遍，且中间验证结果无留痕

> **已修复（c76109a）**：`verify` 每轮留痕 `{round, passed, failure_kind, issues,
> confidence}`（新 state 键 `verification_attempts`），随 verification 落审计；
> 另附 `retry_diagnosis`（首轮失败轮次/类别/issues + `format_only_retries`），
> format-only 重推即 ISSUE-13 类误判信号，应为 0。修复批次中发现并修掉一类
> 真实误判：行首小数 `1.7%` 被列表序号正则剥成 `7%`（2a6dfe1 内）。
> **未验证**：误判类重推为 0 需真实请求样本统计。

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

> **已修复（3e1094a）**：选方向①补齐实现并启用。`CacheBinding` 绑定身份
> （user_id）、授权范围（permission_scope）、客户上下文（client_id）、规范化问题、
> 上下文摘要哈希、知识库版本（取 document_registry 文档数+最近入库时间，
> 跨进程可见），lookup 六维全匹配才参与相似度比较；命中路径补写会话回合并
> 标记 outbox；绑定在图执行前构造一次、lookup/store 复用，否则永不命中；
> `semantic_cache_enabled` 默认 True。
> **未验证**：命中率 >80% 需真实流量。

- **证据**：`semantic_cache_enabled=False`（`src/config.py:69`）、
  `DEFAULT_CACHE_ENABLED=False`（`src/utils/semantic_cache.py:44`），注释写明
  "重新启用前需满足 issues.md 一.1 的条件"——但该条目已随 484e7aa 批次清理删除，
  当前文件已无此条件。设计线为"缓存命中率 >80%"（README §性能）。
- **修复方向**：二选一——①按 484e7aa 记录的条件补齐实现（缓存绑定身份与授权范围、
  客户上下文、规范化问题、上下文摘要哈希、知识库版本；只缓存明确允许的成功终态；
  命中仍执行会话保存与审计）；②把条件重新登记回本文件再决定是否启用。
- **验收**：启用条件可追溯；启用后同角色多会话隔离与审计语义正确，命中率达标。

### ISSUE-27 (P2) BGE reranker 仍未配置，语义重排能力缺失

> **已修复（f68815b）**：选方向①落地。FlagEmbedding 纳入依赖（uv.lock 同步），
> bge-reranker-v2-m3 权重本地化至 HF cache（2.1G，经 hf-mirror；config.json 需
> 单独补抓）；模型名走 `config.rerank_model`（architecture.md §5.1 RERANK_MODEL）；
> 权重取不到归一为「未配置」而非节点故障；ISSUE-16 的启动预热 `_warm_reranker`
> 原本就在。实测 `grade_and_filter` 返回 `reranker_status="applied"`，
> 本地权重真实重排用例通过（tests/test_tools.py）。

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

> **已修复（2a6dfe1）**：新增 `scripts/build_evaluation_datasets.py`，从现网语料
> 与角色矩阵生成四份评估集到 `tests/evaluation/`（数据集绑定真实 chunk_id，
> 重新入库后重跑生成脚本即可）；新增一份机密级 demo 样本使 permission_level
> 拒绝类可测。顺带修了三个让准入门槛永不达标/直接崩溃的评估脚本缺陷：
> 检索权限判定按相关行（而非整批 any(denied)）判正负、答案数字/幻觉门槛只对
> 有据样本分层统计（负例归 expected_outcome_accuracy）、泄漏统计对空
> restricted_text 崩溃。四份产物全部过准入（见「批次修复证据」）。
> **局限**：数据集由语料机械生成（答案取自证据原文、查询取唯一片段），
> 验证的是流水线机制与验证器行为，不是人工标注的答案质量。

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

## 批次修复证据（2026-09-29 晚，本地验证）

**语料重入库（c3774a4，CHUNKER_VERSION v2）**：30,301 → **3,566 chunks**
（公告 1,996 / 财报 1,466 / 研报 94 / faq 8 / product 1 / regulation 1，
研报含新增机密样本），中位长度 **255 字符**、≤30 字占比 **0.1%**
（修复前全库 65.6% ≤10 字、87.9% ≤30 字、43.7% 重复行，现均为 0）、
最大 498 字符落在设计区间。S4 根因证据（国信研报 `EPS 为 20.83/25.96/30.45 元`）
现为 213 字符可检索单块。**S4 是否不再 fail-closed 需起服务实测，本批未跑。**

**评估产物（commit 2a6dfe1，`artifacts/evaluation/2a6dfe1…/`，四份均退出码 0）**：

| 数据集 | 样本 | 指标 |
|---|---|---|
| retrieval.json（§3.1） | 133（正例 101 + 负例 32，三类拒绝全覆盖） | recall@5 **0.891**（≥0.80）、recall@10 **0.960**（≥0.90）、permission **1.000** |
| answers.json（§3.2） | 135（有据 110 + 不可回答 20 + tool-only 正 10 / 负 5） | numeric **1.000**、citation **1.000**、hallucination **0.011**（≤0.05）、outcome **1.000** |
| compliance.json（§3.3） | 50 | block **1.000**、leakage **0.000** |
| conversations.json（§3.4） | 20 | case accuracy **1.000** |

**Reranker（f68815b）**：真实检索链路 `grade_and_filter` 返回
`reranker_status="applied"`（交叉编码器分数如 8.289/7.988 覆盖原始 cosine 序）。

**测试**：645 passed + ruff 全绿（本地需 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`，
否则装上 FlagEmbedding 后任何模型加载路径会去连 huggingface.co——系统代理死端口，
实测就是这种挂起）。

**本批仍未验证（留档）**：①ISSUE-23/24 的延迟验收（单轮 ≤3s、回环 ≤3s、
端到端 P95 ≤10s）需 8001 服务实测，当前会话未跑；②缓存命中率 >80% 需真实流量；
③「误判类重推为 0」需真实请求的 verification 快照统计；④CI 与完整安全审计未确认
（Mimosa 钩子多次提示 library_source/callgraph 扫描不完整，不宣称项目安全）。

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
