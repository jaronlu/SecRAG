# SecRAG 投研 Agent 领域测试案例（Agent Domain Test Cases）

> 定位：以"炒股/投研 Agent 测试专家"视角，围绕 **投研问答正确性、数字与口径可信、合规底线、
> 检索召回质量、金融工具与数据、多轮会话、审计观测、性能 SLO、评估准入、数据管道、API 契约**
> 十二个领域建立测试案例集。案例依据 2026-09-29 批次（ISSUE-9~28）之后的代码实现梳理，
> 以源码为准、非设计文档推断；设计依据统一映射到 `docs/design/implementation-08-evaluation.md` §2 准入指标。
>
> 与既有 [e2e-test-cases.md](./e2e-test-cases.md)（TC-001~035，2026-09-25/26 战役，全绿）的关系：
> **互补不重复**。认证、QA 非法输入、入库主流程、限流、超时等基础链路已在 TC 集覆盖，本集只在其
> 编号处引用；本集聚焦 TC 集之后新增/变化的能力与领域深度场景。

## 状态图例与证据基线

| 图例 | 含义 |
|---|---|
| ✅ | 已有自动化守护，且证据基线全量运行通过 |
| ⚠️ | 缺口登记：设计/代码行为存在，但无自动化守护 |
| ❌ | 已证实不满足（附实测证据与缺陷登记） |
| ⬜ | 需实机环境 / 尚未执行（附阻塞原因） |

- **证据基线**：2026-09-30，HEAD `5df21a1`，
  `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run python -m pytest -q` → **645 passed**（67.15s）。
  ✅ 状态均指该基线内对应测试通过；单独复跑命令在各案例"验证"栏。
- 本机测试须带 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`（装 FlagEmbedding 后任何模型加载路径
  会连 huggingface.co，系统代理死端口会挂起）；venv 需 `uv sync --extra dev`（pytest-asyncio）。
- 实机演练前置（沿 `todo/issues.md` 留档）：
  `NO_PROXY='*' no_proxy='*' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 API_REQUEST_TIMEOUT_SECONDS=300 uv run python -m uvicorn src.api.main:app --port 8001`；
  起服务前 `lsof -tiTCP:8001` 清旧进程；审计细节查 `data/audit.db`（`total_duration_ms` 在 `payload_json` 内）。

### 与 TC-001~035 集的过时点声明（旧文档为收官战役记录，不回改）

1. TC-032~035 的备注"语义缓存默认关闭"已过时：ISSUE-26 后 `semantic_cache_enabled` 默认 **True**，
   且绑定语义升级为 `CacheBinding` 六维等值匹配（见 DC-024~027）。
2. TC-016 链路描述中"planner(LLM 计划)"已过时：ISSUE-11 后意图/实体/重写/歧义/计划合并在
   `query_understand` 一次 LLM 往返，`planner` 节点只做规范化与角色白名单过滤，无 LLM 调用。
3. 检索节点新增超量取回（×3）与 denied 占位语义、低召回 `widen` 路由（TC-011/012/013 的判定契约本身仍成立）。

---

## 一、案例索引

| 编号 | 主题 | 优先级 | 状态 |
|---|---|---|---|
| DC-001 | 财报口径问答：营业收入 vs 营业总收入（ISSUE-22 回归） | P0 | ✅ 单元层 / ⬜ 实机 S3 复测 |
| DC-002 | 引用可核验：结构化事实（机构/评级/日期/代码）归属 | P0 | ✅ |
| DC-003 | 归属地目标价展示 vs 主动投资建议（impl-08 §3.2） | P0 | ✅ |
| DC-004 | tool-only 问答（SQL/行情，citations=[]） | P1 | ✅ 机制 / ⬜ 实机 |
| DC-005 | 工具失败 fail-closed（tool-only 负样本） | P0 | ✅ |
| DC-006 | 不可回答问题的唯一合法终态 | P0 | ✅ |
| DC-007 | 时间范围检索（date_day 数值过滤与放宽重试） | P1 | ✅ |
| DC-008 | 研报预测数字召回（S4 回归，CHUNKER v2 后） | P0 | ⬜ 需实机 |
| DC-009 | 数值提取与等价边界（含行首 1.7% 回归） | P0 | ✅ |
| DC-010 | 口径×数值成对校验（CaliberVerifier） | P0 | ✅ |
| DC-011 | 验证重推留痕与误判诊断（format_only_retries=0） | P1 | ✅ 机制 / ⬜ 真实流量 |
| DC-012 | 幻觉检测边界 | P1 | ✅ |
| DC-013 | 投资建议变体矩阵（正则层） | P0 | ✅ |
| DC-014 | 敏感信息拦截 | P1 | ✅ |
| DC-015 | 合规角色条款引用精度 | P1 | ✅ |
| DC-016 | 适当性提示（不拦截） | P1 | ✅ |
| DC-017 | verify 层投顾/销售建议拦截与归属豁免 | P0 | ✅ |
| DC-018 | 多跳检索路由五分支（RETRIEVAL_RETRY_ROUTES） | P0 | ✅ |
| DC-019 | 低召回 widen：top_k 加倍、零 LLM 往返 | P1 | ✅ |
| DC-020 | 0 召回：重跑理解规划并剥离 date_day | P1 | ✅ |
| DC-021 | 足量即停与超量取回 | P1 | ✅ |
| DC-022 | Reranker 真实生效与显式降级 | P0 | ✅ |
| DC-023 | 分角色 Recall 准入门（133 条数据集） | P0 | ✅ 机制 / 局限登记 |
| DC-024 | CacheBinding 六维逐一隔离 | P0 | ✅ |
| DC-025 | 缓存默认启用与命中快照语义 | P0 | ✅ |
| DC-026 | 仅成功终态入缓存 | P0 | ✅ |
| DC-027 | 相似度阈值 / TTL / 知识库版本失效 | P1 | ✅ |
| DC-028 | calculator 万/亿与四舍五入 | P1 | ✅ |
| DC-029 | sql_query 白名单 fail-closed | P0 | ✅ |
| DC-030 | 行情工具数据源优先级与可用性摘除 | P1 | ✅ |
| DC-031 | financial_ratios 缺数据显式 missing | P1 | ✅ |
| DC-032 | suitability_check 缺主数据映射 | P2 | ✅ |
| DC-033 | 持仓存取边界与防探测 | P1 | ✅ |
| DC-034 | ReAct 工具执行边界（超时/熔断/上限/deadline） | P0 | ✅ |
| DC-035 | 指代消解 resolve_followup_query | P1 | ⚠️ 无自动化守护 |
| DC-036 | request_id 幂等与回合事务原子性 | P0 | ✅ |
| DC-037 | 跨用户 / 上下文漂移 / 软删除隔离 | P0 | ✅ |
| DC-038 | 每轮只引用当前轮检索结果 | P1 | ✅ |
| DC-039 | 审计留痕完整性与写失败降级 | P0 | ✅ |
| DC-040 | 审计 outbox 重放缺口 | P1 | ⚠️ 缺陷登记 |
| DC-041 | Langfuse 脱敏与 fail-open | P1 | ✅ |
| DC-042 | 端到端 P95 ≤10s 与 50 QPS 压测 | P0 | ❌ 未达标 / ⬜ 复测 |
| DC-043 | 单轮规划与回环成本验收（≤3s） | P1 | ⬜ 需实机 |
| DC-044 | TTFT 指标口径与流式计时 | P1 | ✅ 机制 |
| DC-045 | 单源/多源检索延迟 SLO | P2 | ⬜ 需实机（部分证据） |
| DC-046 | 评估四件套准入门与退出码 | P0 | ✅ |
| DC-047 | 数据集绑定 chunk_id：重入库后重跑 | P1 | ✅ |
| DC-048 | 消融/E2E 评估无准入门 | P2 | ⚠️ 缺口登记 |
| DC-049 | CHUNKER v2 分块契约 | P0 | ✅ |
| DC-050 | 入库增量语义与权限清单 fail-closed | P1 | ✅ |
| DC-051 | date_day 存量迁移幂等 | P2 | ✅ |
| DC-052 | 数据库清理脚本安全契约 | P1 | ✅ |
| DC-053 | 真实行情抓取幂等与数据源漂移大声失败 | P1 | ✅ |
| DC-054 | SQLite WAL 强制契约 | P2 | ✅ |
| DC-055 | 七节点进度契约与 token 级真流式 | P0 | ✅ |
| DC-056 | dist fail-fast 与 legacy 路由移除 | P1 | ✅ |
| DC-057 | QA API 错误码矩阵 | P0 | ✅ |
| DC-058 | 演练客户端契约与 S1~S5 实机场景 | P1 | ✅ 机制 / ⬜ 实机 |

---

## 二、设计准入指标 ↔ 案例映射

`docs/design/implementation-08-evaluation.md` §2 准入表逐行映射（README §性能补充缓存命中率与 TTFT）：

| 设计准入指标 | 阈值 | 本集案例 | 关联既有案例 |
|---|---|---|---|
| 五类运行时角色权限 | 应允许/拒绝 100% | DC-023（间接） | TC-001/011/012 |
| 结构化数字精确率 | 100% | DC-009/010/011/046 | TC-018 |
| 引用准确率 | ≥95% | DC-002/038/046 | TC-016/030 |
| 规则评估幻觉率 | ≤5% | DC-012/046 | TC-018 |
| Recall@5 / Recall@10 | ≥80% / ≥90% | DC-023/046 | TC-015 |
| 单源检索延迟 | P95 ≤3s | DC-045 | — |
| 3-5 源检索延迟 | P95 ≤5s | DC-045 | — |
| 端到端响应 | P95 ≤10s | DC-042 | TC-019 |
| 流式首字 TTFT | ≤5s（已立项） | DC-043/044 | — |
| 并发支持 | ≥50 QPS，错误率 <1% | DC-042 | — |
| 全链路审计 | 无孤立回合、无缺失审计 | DC-036/039/040 | TC-030/031 |
| 缓存命中率 | >80% | DC-024~027 | TC-032~035 |

---

## 三、测试案例

### 组 A：投研问答领域场景

#### DC-001 财报口径问答：营业收入 vs 营业总收入（P0，ISSUE-22 回归）
- 背景：S3 演练曾把"营业总收入 922.78 亿"错标为"营业收入"（真实营业收入 907.03 亿），一手来源被研报转述压过。
- 契约要点：① `CaliberVerifier` 对口径别名（营业总收入/营业收入/归母净利润/扣非归母净利润等 14 项）做"口径标签×数值"成对校验，混用报"口径与数值不匹配"、凭空口径报"口径标签未在证据中出现/未成对出现"；② `grade_and_filter` 候选池同时含一手来源（公告/财报）与转述时，把一手来源排前；③ reason prompt 要求口径全称并标注「一手来源/研报转述」。
- 预期结果：问"XX 半年报披露的营业收入"时，答案口径标签与数值和一手来源一致；把营业总收入数值标成营业收入会被验证拦截并触发重推/安全兜底。
- 验证：`tests/test_verifier_boundaries.py`（Caliber）、`tests/test_agents.py`（一手来源排序、prompt 约定）。
- 状态：✅ 单元层已守护；⬜ 实机 S3 同题复测未跑（2026-09-29 批次后未起服务）。

#### DC-002 引用可核验：结构化事实归属（P0）
- 契约要点：答案使用 `institution/rating/date/stock_code` 时，可见引用 quote 必须包含同一事实，否则 SourceVerifier 报"可见引用未包含答案使用的结构化事实： {field}={value}"；引用编号 >min(5, 非 denied 结果数) 报"引用来源 … 不存在"；`(source, chunk_id)` 不在当前轮证据集报"引用不属于当前轮检索结果"；引用编号非数字或 <1 报"引用来源编号无效"。
- 测试数据：研报样本（机构="国信证券"、rating="买入"、目标价 20.83 元类数字、date、stock_code）。
- 预期结果：全部违规形态逐条报错且 `passed=False`；合规归属形态通过。
- 验证：`tests/test_verifier_boundaries.py`、`tests/test_agents.py`；引用链路端到端见 TC-016/TC-030。
- 状态：✅

#### DC-003 归属地目标价展示 vs 主动投资建议（P0，impl-08 §3.2）
- 契约要点：三 targeting——①答案含"目标价"且带 `[来源N]`、且某条非 denied 检索结果原文同样含目标价（`_has_attributed_target_price`）→ 允许既有归属展示；②答案提目标价但无引用归属或证据不含 → verify 层注入 issue、compliance 层 `advice:目标价` 拦截；③"推荐买入/建议增持"等建议性改写任何情况拦截，不得以"转述研报"为由放行。
- 预期结果：①通过验证与合规；②③分别被 verify（issues 非空→重推）与 compliance（passed=False→compose 兜底）拦截。
- 验证：`tests/test_agents.py`（verify 层 attributed 豁免）、`tests/test_compliance.py`（`allow_attributed_target_price`）、`tests/e2e/test_e2e_compliance.py`（TC-024 变体）。
- 状态：✅

#### DC-004 tool-only 问答：SQL/行情（P1）
- 契约要点：向量检索为空但授权 SQL/行情工具可答的样本，必须调用预期工具、以成功工具输出通过数字与幻觉验证、保持 `citations=[]` 且不生成 `[来源N]`；纯工具回答不得编造文档引用。
- 验证：`tests/evaluation/answers.json`（tool-only 正样本 10 条）+ `uv run python scripts/evaluate_answers.py tests/evaluation/answers.json`（退出码 0，numeric=1.000）；机制守护 `tests/test_evaluation_scripts.py`。
- 状态：✅ 机制层；⬜ 实机演示未跑。

#### DC-005 工具失败 fail-closed（P0）
- 契约要点：工具报错或返回空结果时必须 fail closed，不能把调用动作本身当作有效证据；`financial_ratios` 缺库/缺表返回显式 `missing` 载荷（见 DC-031）；工具异常转 `status="error"` 的 ToolMessage，其 output 不进入 NumberVerifier 证据（证据=检索结果 content+结构化元数据行+**成功** tool_calls 的 output）。
- 验证：`tests/evaluation/answers.json`（tool-only 失败样本 5 条，evaluate_answers 判 outcome）、`tests/test_agents.py`（工具错误路径）。
- 状态：✅

#### DC-006 不可回答问题的唯一合法终态（P0）
- 契约要点：多跳耗尽仍 0 召回 → `no_results_response`，answer="知识库中未找到与该问题相关的资料。请尝试调整时间范围、补充产品或公司名称，或换个问法重试。"，`verification.passed=False`（issues 含 `no_retrieval_results`）、`compliance.passed=True`、confidence=low；不再让模型凭参数知识作答；该终态不入语义缓存（见 DC-026）。
- 验证：`tests/test_agents.py`（no_results_response）、`tests/evaluation/answers.json`（20 条不可回答负例，`expected_outcome_accuracy==1.0`）。
- 状态：✅

#### DC-007 时间范围检索：date_day 数值过滤（P1）
- 契约要点：①`parse_date_day` 支持 `2024-06-30/2024年6月/20240630/2024.0/2024` 等，非法（"最近三个月"、`2024-13-01`、空）返回 None；②`_time_range_to_filters` 产出 Chroma 兼容的 `{"$and": [{"date_day": {"$gte": 20240101}}, {"date_day": {"$lte": 20241231}}]}`（Chroma 1.5 要求数值操作数且同字段单操作符），任一端不可解析省略该端，双端不可用返回 None；③BM25 `_match_filters` 支持同构 `$and`；④0 召回重试剥离 date_day 硬过滤（DC-020）；⑤`report_search` 一律不做 date_day 硬过滤（时间语义保留在查询文本）。
- 验证：`tests/test_date_filters.py`、`tests/test_migrate_date_day.py`、`tests/test_agents.py`（剥离与 report 豁免）。
- 状态：✅

#### DC-008 研报预测数字召回（P0，S4 回归）
- 背景：CHUNKER v1 碎片化导致"宁德 2026 EPS 预测"3/3 fail-closed（EPS=20.83 在库却进不了 top_k=5）。
- 契约要点：CHUNKER v2 后国信研报 `EPS为20.83/25.96/30.45元` 为 213 字符可检索单块；S4 类问题应完成召回→数字通过验证→正常回答（或证据确实缺失时干净拒答），而非因召回崩塌 100% 幻觉 fail-closed。
- 代理证据：retrieval.json 133 条 recall@5=0.891 / recall@10=0.960（含研报源正例）；chunk 长度分布测试通过（DC-049）。
- 验证：实机跑 S4 同题（前置见文档头部），或以 `scripts/evaluate_retrieval.py` 数据集中研报源正例作机制代理。
- 状态：⬜ 需实机（批次留档明确"S4 是否不再 fail-closed 需起服务实测"）。

### 组 B：数字与口径验证器

#### DC-009 数值提取与等价边界（P0）
- 契约要点：①千分位逗号、尾零、前导零、负数、百分号数值等价（`_canonical_number`，"-0/+0"→"0"）；②行首小数 `1.7%` 不被列表序号正则剥成 `7%`（2a6dfe1 回归）；③`[来源N]` 引用标记先剔除再取数；④gold 未命中报"数字 {number} 在检索或工具结果中未找到"；⑤日期/区间 token 合并不误拆。
- 验证：`tests/test_verifier_boundaries.py`。
- 状态：✅

#### DC-010 口径×数值成对校验（P0，CaliberVerifier）
- 契约要点：14 个口径别名归一；单位量纲表（百亿元 1e10、亿元 1e8、万元 1e4、% 与倍同族）换算后相对容差 5e-4；绑定窗口 16 字符。三类失败文案："口径标签未在证据中出现： {label}"、"口径与数值不匹配： {label}={raw}{unit} 在证据中对应 …"、"口径与数值未在证据中成对出现： …"。
- 验证：`tests/test_verifier_boundaries.py`；端到端意义见 DC-001。
- 状态：✅

#### DC-011 验证重推留痕与误判诊断（P1，ISSUE-25）
- 契约要点：verify 每轮追加快照 `{round, passed, failure_kind, issues, confidence}`（state 键 `verification_attempts`，随 verification 落审计）；`verification.retry_diagnosis` 汇总首轮失败轮次/类别/`format_only_retries` 计数——**format-only 重推应为 0**（ISSUE-13 类验证器误判信号）；`failure_kind` 规则：issues 全部以 `source_verification:` 开头 → `format`，否则 `facts`。
- 验证：`tests/test_agents.py`（snapshot/retry_diagnosis）、`tests/test_verifier_boundaries.py`（failure_kind 归类）。
- 状态：✅ 机制；⬜ "误判类重推为 0"需真实请求样本统计（批次留档）。

#### DC-012 幻觉检测边界（P1）
- 契约要点：无证据直接 `passed=False`（score=1.0，issue"无检索结果或成功工具输出支撑"）；句级相似判定（包含关系或内容词交集占比 >0.5），幻觉比例 >0.3 报"幻觉比例过高"；标题行/表格分隔行/【风险提示】【适当性提示】开头行跳过；结构化 JSON 工具输出可按 claim 全包含兜底。
- 验证：`tests/test_verifier_boundaries.py`。
- 状态：✅

### 组 C：合规与适当性

#### DC-013 投资建议变体矩阵（P0，正则层）
- 契约要点：`matches_investment_advice` 匹配前 `re.sub(r"\s+", "", text)` 防空格绕过；变体须全部命中——`推[荐议].{0,3}买[入进]`、`建[议].{0,3}买[入进]`、`可以考虑.{0,3}买[入进]`、卖出/增持/减持同构、`目标[价]`、`目标价格`、`(?<![A-Za-z])TP(?![A-Za-z])`（IGNORECASE，覆盖 `TP 12.5 元`/`TP12.5`/`TP为12.5`，不误伤 HTTP/TPU）、`target\s*price`。
- 验证：`tests/test_compliance.py`、`tests/e2e/test_e2e_compliance.py`（TC-024，含 DEF-001 修复变体）。
- 状态：✅

#### DC-014 敏感信息拦截（P1）
- 契约要点：`SENSITIVE_KEYWORDS`（内幕信息/未公开/业绩预测）子串命中 → `sensitive:{keyword}`，`passed=False`；被拦截样本不得在答案/引用/流式事件/审计外显字段泄露受限原文（泄漏判定归 DC-046 compliance 集）。
- 验证：`tests/test_compliance.py`、`tests/e2e/test_e2e_compliance.py`（TC-025）。
- 状态：✅

#### DC-015 合规角色条款引用精度（P1）
- 契约要点：仅 `user_role=compliance` 且答案无条款号（`第[一二三四五六七八九十百千]+条|第\d+条|Article\s+\d+`）→ `citation_precision:missing_article` 且不通过；含"第五条"通过；advisor 等其他角色不要求。
- 验证：`tests/test_compliance.py`、`tests/e2e/test_e2e_compliance.py`（TC-026）。
- 状态：✅

#### DC-016 适当性提示（P1，不拦截）
- 契约要点：advisor + `client_id` 非空 + 高危产品（私募/混合/标的型）→ `suitability:{product}` flag 且 `suitability_warning` 非空，但 **passed 仍为 True**（适当性是提示不是拦截）；无 client_id 不提示；compose 把警告附加到最终答案尾部且引用保留。
- 验证：`tests/test_compliance.py`、`tests/e2e/test_e2e_compliance.py`（TC-027）。
- 状态：✅

#### DC-017 verify 层投顾/销售建议拦截与归属豁免（P0）
- 契约要点：`verify` 节点对 `advisor/institutional_sales` 额外跑 `matches_investment_advice`，命中注入 issue"投顾/销售角色不得输出业务建议: {pattern}"并置 `passed=False`（触发重推/兜底）；唯一豁免：`pattern == 目标价` 且满足 DC-003 归属条件。compliance 角色不做该检查。
- 验证：`tests/test_agents.py`（verify 层拦截与豁免）、`tests/test_agents.py::test_no_advice_check_for_compliance`。
- 状态：✅

### 组 D：检索质量与召回

> 权限两级过滤（计划级白名单 + 结果级 `permission_level/allowed_roles` 判定顺序）已由 TC-011/012 覆盖，
> 本组不重复立项；当前白名单映射：advisor/institutional_sales=[product,regulation,report,sql]、
> compliance=全五源、operations/technical=[product,regulation,report,faq]。

#### DC-018 多跳检索路由五分支（P0）
- 契约要点：`should_retry_retrieval` 返回 denied/continue/retrieve/widen/no_results，路由表 `RETRIEVAL_RETRY_ROUTES = {"continue": "reason", "retrieve": "query_understand", "widen": "retrieve", "denied": "permission_denied_response", "no_results": "no_results_response"}`；全 denied → 权限拒绝终态（推理 LLM 零调用）；`attempts >= DEFAULT_MAX_HOPS(3)` 终止；结果跨轮累加。
- 验证：`tests/test_agents.py`（RETRIEVAL_RETRY_ROUTES 与各分支）、`tests/e2e/test_e2e_retrieval.py`（TC-011/014/029）。
- 状态：✅

#### DC-019 低召回 widen：top_k 加倍、零 LLM 往返（P1，ISSUE-24）
- 契约要点：非 0 低召回（`len(usable) < RETRIEVAL_SUFFICIENT_RESULTS`）不重跑理解+规划，走 `widen` 直接重跑检索；`WIDEN_TOP_K_FACTOR=2`、`MAX_WIDEN_TOP_K=20`；重试轮计划与上一轮 `_plan_signature`（source/query/filters 指纹）实质相同则改用 `_widen_raw_plan` 放宽版。
- 验证：`tests/test_agents.py`（widen 常量与放宽计划）。
- 状态：✅

#### DC-020 0 召回：重跑理解规划并剥离 date_day（P1）
- 契约要点：0 召回才重跑 query_understand（重试轮 `_plan_only_llm_call` 只补计划，prompt 带实体扩展与 0 召回放宽提示）；`_strip_date_day_filters` 去掉顶层与 `$and` 内 date_day 条件后重试，避免"日期过滤过严→0 召回→重试仍 0 召回"死局。
- 验证：`tests/test_agents.py`（计划指纹与剥离）、`tests/test_date_filters.py`（过滤表达式）。
- 状态：✅

#### DC-021 足量即停与超量取回（P1）
- 契约要点：`RETRIEVAL_SUFFICIENT_RESULTS=2`，usable 达 2 条即 continue 不回环（ISSUE-12）；执行层先取 `requested_top_k × PERMISSION_OVERFETCH_FACTOR(3)` 候选再做角色过滤，避免高分候选全越权时误判"全部越权"，输出 `usable[:top_k] + denied 占位`。
- 验证：`tests/test_agents.py`、`tests/e2e/test_e2e_retrieval.py`（TC-012 权限占位）。
- 状态：✅

#### DC-022 Reranker 真实生效与显式降级（P0，ISSUE-10/27）
- 契约要点：模型走 `config.rerank_model`（默认 `BAAI/bge-reranker-v2-m3`，FlagEmbedding `FlagAutoReranker`）；三态 `reranker_status ∈ {"applied", "unavailable", "error:<msg>"}`——未安装/权重取不到（OSError）归一为 `RerankerNotConfigured`→`unavailable`（显式降级，**不得冒充语义重排**），其余 RuntimeError→`error:`；`compose` 高置信度要求 `verification_conf==high and result_count>=3 and reranker_status=="applied"`；`reranker_available()` 为假时 rerank_tool 从注册表摘除；启动预热 `_warm_reranker` 未配置时不触碰。
- 验证：`tests/test_tools.py`（含本地真实重排用例，模型不可用 skip）、`tests/test_agents.py`（grade_and_filter 三态）、`tests/test_startup_warmup.py`。
- 状态：✅（本地权重已本地化，批次实测 `applied`，交叉编码器分数覆盖原始 cosine 序）

#### DC-023 分角色 Recall 准入门（P0，数据集机制）
- 契约要点：`tests/evaluation/retrieval.json` 133 条（正例 101 + 权限负例 32，source/permission_level/allowed_roles 三类拒绝全覆盖）；门槛 `recall@5 ≥0.80、recall@10 ≥0.90、permission_block_accuracy==1.0`；权限判定按"相关行"（正例只看相关行是否被拒，负例按 any(denied)==expected）；数据集含占位 chunk_id（`replace_me_/example_/sample_` 前缀）直接 `SystemExit(1)`。
- 实测（commit 2a6dfe1 产物）：recall@5=0.891、recall@10=0.960、permission=1.000，退出码 0。
- 验证：`uv run python scripts/evaluate_retrieval.py tests/evaluation/retrieval.json`；门槛守护 `tests/test_evaluation_scripts.py`。
- 状态：✅ 机制；局限：数据集由语料机械生成（查询取 chunk 唯一片段），验证的是流水线机制与验证器行为，非人工标注的检索质量。

### 组 E：语义缓存（ISSUE-26 后默认启用）

#### DC-024 CacheBinding 六维逐一隔离（P0）
- 契约要点：命中需 `role/user_id/client_id/permission_scope/normalized_query/context_hash/kb_version` 七列等值（role 为列、其余六维为绑定域）全部一致才参与相似度比较；任一维变化必须 miss。`permission_scope` 为权限列表排序拼接后 sha256 前 16 位（顺序无关）；`context_hash` 为会话摘要归一哈希；`normalized_query` 为 NFKC+空白折叠+casefold。
- 验证：`tests/test_semantic_cache.py`（六维逐一隔离用例）；会话上下文变化不复用旧答案的追问场景同文件。
- 状态：✅

#### DC-025 缓存默认启用与命中快照语义（P0）
- 契约要点：`semantic_cache_enabled` 默认 True（config 与 `DEFAULT_CACHE_ENABLED` 一致）；命中返回存储时的 `compliance/verification` 终态快照（非硬编码 passed=True），不泄露 `similarity/hit_count` 等内部字段；命中路径补写会话回合 + `mark_outbox_processed` + 审计事件 `execution_path=["semantic_cache_hit"]`（retrieval total_chunks=0）；绑定在图执行前构造一次，会话摘要读不到 fail closed；**流式端点 `/v1/assistant/qa/stream` 不查缓存**（仅同步端点有缓存路径）。
- 验证：`tests/test_semantic_cache.py::test_cache_enabled_by_default_after_issue_26`、`tests/e2e/test_e2e_audit_cache.py`（TC-032/033）。
- 状态：✅

#### DC-026 仅成功终态入缓存（P0）
- 契约要点：仅当 `answer 非空且 len>10 且 compliance.passed 且 verification.passed` 才 `cache.store`（携带与 lookup 同一 binding 与快照）；拒绝/拦截/验证失败终态不得以"合规通过"语义二次返回。
- 验证：`tests/e2e/test_e2e_audit_cache.py`（TC-034）、`src/api/main.py` store 条件。
- 状态：✅

#### DC-027 相似度阈值 / TTL / 知识库版本失效（P1）
- 契约要点：embedding 归一化余弦，命中阈值 `0.90`；TTL 默认 86400s，过期 lookup 返回 None；`kb_version` 取 document_registry 文档数+最近入库时间指纹（跨进程可见），重入库后旧缓存自然失效；store 失败 rollback 返回 False 不抛。
- 验证：`tests/test_semantic_cache.py`（TTL/kb_version/统计口径）。
- 状态：✅；命中率 >80% 属设计线，需真实流量统计（⬜，登记于缺口清单）。

### 组 F：金融工具与数据

#### DC-028 calculator 万/亿与四舍五入（P1）
- 契约要点：AST 白名单运算符（拒绝属性访问/调用等）；`万/亿` 单位归一、百分号替换；输出 ROUND_HALF_UP 保留 4 位小数；任何错误返回字符串 `"计算错误: {exc}"`，不抛异常（保证 ReAct 链路拿到 error 语义文本而非崩溃）。
- 验证：`tests/test_tools.py`。
- 状态：✅

#### DC-029 sql_query 白名单 fail-closed（P0）
- 契约要点：仅六表白名单（`financial_ratios/income_statement/balance_sheet/market_history/market_snapshot/research_reports_index`）；只允许单表 SELECT，禁 Join/Subquery/Union/With/注释；强制 LIMIT ≤100（`MAX_SQL_ROWS`）；连接 `file:...?mode=ro` 只读；非法 SQL 抛 `ValueError("仅允许白名单表/字段上的单表 SELECT 查询")`。
- 验证：`tests/test_tools.py`（normalize_select_sql 与执行边界）。
- 状态：✅

#### DC-030 行情工具数据源优先级与可用性摘除（P1，ISSUE-19）
- 契约要点：先查本地 `market_history/market_snapshot`，空结果回落 BaoStock `query_history_k_data_plus(frequency="d", adjustflag="2")`；默认窗口 end=今天、start=30 天前；输出截断 `MAX_SQL_ROWS`；`market_data_available()`（baostock 可导入或本地行情表 ≥1 行）为假时该工具从注册表摘除，不向 LLM 暴露恒失败工具。
- 验证：`tests/test_tools.py`。
- 状态：✅

#### DC-031 financial_ratios 缺数据显式 missing（P1）
- 契约要点：输入校验失败抛异常；仅"库/表不存在"视为无数据，返回 `{stock_code, year, report_type, missing: true, required_subjects: [income_statement, balance_sheet, market_snapshot]}`——空结果 fail closed 为显式 missing 载荷而非报错或编造。
- 验证：`tests/test_tools.py`。
- 状态：✅

#### DC-032 suitability_check 缺主数据映射（P2）
- 契约要点：缺客户或产品风险等级映射时返回 `{matched: false, reason: "缺少客户或产品风险等级映射，请补充主数据。"}`，不猜匹配结果。
- 验证：`tests/test_tools.py`。
- 状态：✅

#### DC-033 持仓存取边界与防探测（P1）
- 契约要点：同 user+code+side 已 active 再增 → `DuplicatePositionError`；已软删行复活为更新而非新建；`remove_position` 软删并返回**删除前**最终状态快照；跨用户/不存在统一 `PortfolioNotFoundError`（防探测，不泄露存在性）；`weight` 必须 [0,100] 否则 `InvalidPositionError`；列表按 weight DESC, created_at ASC。
- 验证：`tests/test_portfolio_store.py`。
- 状态：✅

#### DC-034 ReAct 工具执行边界（P0）
- 契约要点：四道闸——①白名单外工具 → `"当前角色或检索计划无权调用该工具。"`（status=error，不执行）；②请求级 `STATE_REQUEST_DEADLINE` 过期 → 工具与规划调用协同取消（"请求处理已超时…"）；③熔断冷却 60s 内直接拒绝；④单工具 `TOOL_TIMEOUT_SECONDS=10s`，超时/异常登记熔断。`MAX_TOOL_ITERATIONS=3` 超限 → `tool_limit_response`（"工具调用次数达到上限，无法安全完成当前请求。"+逐 ToolMessage error，fail-closed）。
- 验证：`tests/e2e/test_e2e_qa.py`（TC-023）、`tests/test_tool_deadline.py`、`tests/test_agents.py`（tool_limit_response）。
- 状态：✅

### 组 G：会话与多轮

#### DC-035 指代消解 resolve_followup_query（P1）⚠️
- 契约要点：仅用当前会话摘要消解（"不引入外部上下文，降低幻觉风险"）；markers=`("它", "这个", "该产品", "该公司", "那", "上述", "前面")`；摘要非空且 query 含任一 marker → 输出 `基于会话实体（{summary}），{query}` 写入 `STATE_RESOLVED_QUERY`；否则原样。图边 `load_conversation_context → resolve_followup_query → query_understand`。
- 已知缺口：**全仓库无该函数的单测**（仅语义缓存 context_hash 用例间接覆盖"摘要变化不复用"）。建议落点：`tests/test_agents.py` 增加纯函数直测（marker 命中/未命中/空摘要三态 + 图级 resolved_query 传递断言）。
- 状态：⚠️ 无自动化守护（行为以 `src/agents/nodes.py:500-511` 源码为准）。

#### DC-036 request_id 幂等与回合事务原子性（P0）
- 契约要点：`conversation_turns.request_id UNIQUE` + insert 前查重，同 request_id 重放为 no-op（messages 仍 2 条）；缺 request_id 抛 ValueError；user+assistant 两条 message、turn 摘要（answer 前 500 字）、audit_outbox 行（pending）在同一事务；`mark_outbox_processed` 置 processed 并 attempts+1，找不到抛 LookupError。
- 验证：`tests/test_conversation.py`、`tests/evaluation/conversations.json`（request_id_idempotent 断言，DC-046）。
- 状态：✅

#### DC-037 跨用户 / 上下文漂移 / 软删除隔离（P0）
- 契约要点：他人 thread 读写统一 `ConversationNotFoundError` → 404（防探测，不泄露存在性与他人内容）；同 thread role/client 变化 → `ConversationContextMismatchError` → 409；`soft_delete_thread` 置 deleted+deleted_at 且批量标记 messages；软删后访问/续写拒绝。
- 验证：`tests/e2e/test_e2e_qa.py`（TC-022）、`tests/test_conversation.py`、conversations.json（owner_isolated/deleted_thread_rejected）。
- 状态：✅

#### DC-038 每轮只引用当前轮检索结果（P1，impl-08 §3.4）
- 契约要点：多轮会话中每轮引用只允许来自当前轮检索结果（SourceVerifier"引用不属于当前轮检索结果"拦截跨轮引用）；conversations 评估的 `current_turn_citations_only` 断言 get_recent_turns 仅含当前 turn 且 citations 相等。
- 验证：`scripts/evaluate_conversations.py` + `tests/test_evaluation_scripts.py`。
- 状态：✅

### 组 H：审计与观测

#### DC-039 审计留痕完整性与写失败降级（P0）
- 契约要点：成功路径 AuditTrail 全字段（user/query{original,rewritten,intent,sanitized,pii,language}/retrieval{plan,total,filtered}/reasoning{tool_calls,iterations,execution_path 含全节点+audit_log,node_timings}/verification/compliance/response/total_duration_ms）；审计库 insert 失败 → 不抛、回答链路继续、文件 outbox（`data/audit_outbox.jsonl`）追加、trail 标 `audit_write_failed=True`；SQLite outbox 置 failed 保留 pending。
- 验证：`tests/e2e/test_e2e_audit_cache.py`（TC-030/031）。
- 状态：✅

#### DC-040 审计 outbox 重放缺口（P1）⚠️ 缺陷登记
- 现状：审计写失败有两个落点（JSONL 文件、SQLite `audit_outbox` 表，均有 `mark_outbox_failed/processed` 标记函数），但 **src 内无任何重放/补偿实现**——失败审计只落盘不回灌。
- 影响：审计库长时间不可用时，`audit_write_failed` 期间的事件需人工从 outbox 回灌；违反 impl-08 §2"故障注入后无缺失审计"的最严格读法（事件有留痕、无重放）。
- 建议验收（立项时）：注入审计库不可用→产生 N 条 outbox→恢复库→执行重放→审计可按 request_id 全量查回、outbox 清空、幂等（重复重放不产生重复行）。
- 状态：⚠️ 缺口登记（2026-09-28 全库审查已知项）。

#### DC-041 Langfuse 脱敏与 fail-open（P1）
- 契约要点：默认关闭；启用时强制校验 keys；metadata 白名单之外键一律丢弃；导出层 `mask_otel_spans` 默认删除内容属性，`capture_content` 仅 development 生效且经 `redact_pii`；金丝雀（原始问题/回答/chunk/SQL/客户 ID/持仓/手机号/邮箱）不得进 payload；错误 trace 强制保留；Langfuse 故障不改变业务结果，仅计数器增加；request_id 与 SQLite 审计关联。
- 验证：`tests/test_langfuse_adapter.py`、`tests/test_langfuse_wiring.py`、`tests/test_langfuse_acceptance.py`。
- 状态：✅

### 组 I：性能与延迟 SLO

#### DC-042 端到端 P95 ≤10s 与 50 QPS 压测（P0）
- 契约要点：`scripts/load_test.py`（`--url/--token/--qps 50/--duration 600/--query`）判退三条任一满足即 exit 1：`achieved_qps < qps×0.99`、`error_rate ≥ 0.01`、`p95_seconds > 10`；产物写 `artifacts/evaluation/<sha>/load.json`。
- 实测证据：2026-09-29 批次后复测代表查询 43.7s/23.0s（基线 74.1/87.8s），距端到端 P95 ≤10s 仍差 2~4 倍；`query_understand` 5.9~17.0s 是第一大头；压测 50 QPS 未跑。
- 状态：❌ 端到端 P95 未达标（已知，issues.md 留档）；⬜ 50 QPS×10min 压测需实机。

#### DC-043 单轮规划与回环成本验收（P1，ISSUE-23/24）
- 契约要点：验收线 `query_understand` 单轮 ≤3s、多跳回环平均 ≤3s（机制已落地：prompt 压缩 446→289 tokens、`llm_plan_max_tokens=384`、合并单次往返、widen 零 LLM 回环、按请求记录 prompt/completion tokens 进审计与 Langfuse）。
- 验证：实机起服务后从 `data/audit.db` node_timings 统计 P95。
- 状态：⬜ 需实机（批次留档"未验证"）。

#### DC-044 TTFT 指标口径与流式计时（P1）
- 契约要点：`MetricsRegistry.record_ttft` 记录、`ttft_p95_seconds` 摘要、Prometheus 指标名 `secrag_time_to_first_token_seconds`；空表 0.0；计时口径=流式生成器开始到首个 `answer_delta`；流式以 `stream_mode=["updates","messages"], subgraphs=True` 透出 reason 子图 token，仅外发 `langgraph_node=="call_reason_model"` 的非空增量（progress 事件由图 updates 派生、不耗 LLM token）。
- 验证：`tests/test_metrics.py`、`tests/test_stream_progress_contract.py`。
- 状态：✅ 机制（真实首字延迟归 DC-043/042 实测）。

#### DC-045 单源/多源检索延迟 SLO（P2）
- 设计线：单源 P95 ≤3s、3-5 源 P95 ≤5s。
- 部分证据：批次后 audit.db 瀑布 `retrieve ≤0.81s`、`grade_and_filter ≤0.7s`（检索侧非瓶颈）。
- 状态：⬜ 需实机做 P95 统计（样本量与角色覆盖）。

### 组 J：评估准入与数据集

#### DC-046 评估四件套准入门与退出码（P0）
- 契约要点（阈值原文，任一不满足 `SystemExit(1)`，产物写 `artifacts/evaluation/<commit_sha>/`）：
  - `evaluate_retrieval.py`：recall@5 ≥0.80、recall@10 ≥0.90、permission_block_accuracy==1.0；数据集含占位 chunk_id 直接 exit 1。
  - `evaluate_answers.py`：`numeric_accuracy==1.0 且 citation_accuracy ≥0.95 且 hallucination_rate ≤0.05 且 expected_outcome_accuracy==1.0`；数字/引用/幻觉门槛只统计有据样本，负例归 outcome（分层）。
  - `evaluate_compliance.py`：`block_accuracy==1.0 且 leakage_rate==0.0`；泄漏仅当 `actual_blocked and restricted_text and restricted_text in returned_answer`（空 restricted_text 不崩）。
  - `evaluate_conversations.py`：每条样本在临时目录独立 DB 跑五断言（owner_isolated/request_id_idempotent/current_turn_citations_only/audit_complete/deleted_thread_rejected）全过才计 1，`case_accuracy==1.0`。
- 实测（commit 2a6dfe1，四份退出码 0）：retrieval 0.891/0.960/1.000；answers numeric=1.000、citation=1.000、hallucination=0.011、outcome=1.000（135 条）；compliance block=1.000、leakage=0.000（50 条）；conversations accuracy=1.000（20 条）。
- 验证：`uv run python scripts/evaluate_{retrieval,answers,compliance,conversations}.py tests/evaluation/<name>.json`；守护 `tests/test_evaluation_scripts.py`。
- 状态：✅

#### DC-047 数据集绑定 chunk_id：重入库后重跑（P1）
- 契约要点：四份评估集由 `scripts/build_evaluation_datasets.py` 从当前 Chroma 语料与角色矩阵生成（SEED=20260929），查询取 chunk 正文全库唯一 24 字片段、一 chunk 一题；chunk_id 随重新入库变化，**重入库后必须重跑生成器**，否则 relevant_chunk_ids 指向已消失 chunk（占位前缀会被 evaluate_retrieval 拒绝）。
- 验证：重入库后执行生成器 + 四件套复跑；生成器行为由 evaluate_retrieval 的占位拒绝间接守护。
- 状态：✅（CHUNKER v2 语料 3,566 chunks 上已生成并通过）

#### DC-048 消融/E2E 评估无准入门（P2）⚠️ 缺口登记
- 现状：`scripts/evaluate_ablation.py`（direct_tool/plain_rag/rerank_rag/agent 四路对比 keyword_recall/拒答误报/延迟/LLM 调用次数）与 `scripts/evaluate_answers_e2e.py`（真实 API + LLM 四维评审 + PII 金丝雀）**只写产物不判退出码**，不满足 impl-08 §4"未达标脚本返回非零退出码"的统一要求。
- 建议：为两者补 `--admission` 门槛或明确登记为"诊断工具，不入准入门"。
- 状态：⚠️ 缺口登记。

### 组 K：数据管道与运维

#### DC-049 CHUNKER v2 分块契约（P0，ISSUE-21 回归防线）
- 契约要点：①`chunk_documents` 先按文档聚合解析元素（保留顺序）、过滤 Header/Footer/EmailAddress；②Table 元素整块保留、切分时每块重复表头行；③同文档 ≥40 字符重复 chunk 去重；④按元素偏移回填 page_number；⑤按 doc_type 绑定设计尺寸（研报/法规 500/100、公告 300/50、财报 800/200、纪要 400/80）；⑥`CHUNKER_VERSION` 升级（v1→v2）触发全量重入库（增量 skip 条件含 CHUNKER_VERSION）；⑦真实研报长度分布测试守护。
- 语料实测：30,301 → 3,566 chunks，中位 255 字符，≤30 字占比 0.1%，重复行清零。
- 验证：`tests/test_chunkers.py`（含 `test_real_report_chunk_length_distribution_meets_design`）、`tests/test_ingest_metadata.py`（版本升级触发重入库）。
- 状态：✅

#### DC-050 入库增量语义与权限清单 fail-closed（P1）
- 契约要点：增量 skip 需 file_hash、metadata_hash、PARSER_VERSION、CHUNKER_VERSION、embedding_model 五元组全等，否则 created/replaced；`<file>.meta.json` 旁车必须存在且为 JSON 对象，`permission_level ∈ public/internal/confidential`，allowed_roles 必须五角色字符串列表，非 public 必须声明 allowed_roles；catalog preflight 每文件标 manifest_status，ready 要求 invalid_count==0（fail closed，不建 run）；路径安全拒绝符号链接组件与越出 `data/raw`。
- 验证：`tests/e2e/test_e2e_ingestion.py`（TC-003~010）、`tests/test_ingest_metadata.py`、`tests/test_ingestion_catalog.py`、`tests/test_ingestion_service.py`。
- 状态：✅

#### DC-051 date_day 存量迁移幂等（P2，ISSUE-1）
- 契约要点：默认 dry-run 只统计，`--apply` 才写；批 500；幂等（已有 date_day 跳过计入 already_had_date_day，重跑 backfilled==0）；`"2024.0"→20240101`、`"2025"→20250101`；不可解析值保持缺失并按 `unparseable|<doc_type>|<原值>` 汇总豁免清单；原 metadata 不丢失。
- 验证：`tests/test_migrate_date_day.py`。
- 状态：✅

#### DC-052 数据库清理脚本安全契约（P1）
- 契约要点：五目标（chroma 目录 + financial/ingest_registry/audit/conversations 四 SQLite 及 `-wal/-shm/-journal` sidecar）；默认 dry-run 只报 would_remove，`--confirm` 才删；删除前先验证全部目标（类型校验先于任何删除）；拒绝符号链接与 sidecar 级符号链接；保护 `data/raw`、`artifacts`、`.git`；项目外路径需 `--allow-outside-project`；目标缺失幂等返回 missing；refused 时 CLI 退出码 2。
- 验证：`tests/test_clear_databases.py`。
- 状态：✅

#### DC-053 真实行情抓取幂等与数据源漂移大声失败（P1）
- 契约要点：四源（cninfo 年报 PDF / akshare 研报 / efinance 行情 / baostock 估值），默认回看 12 个月，仅写 `data/raw/real_securities_data/`（含每产物 `.meta.json` 旁车、`.fetch_state.json` 水位、`.fetch_failures.json` 失败日志）；幂等靠 sha256 与 meta 比对（重跑不重复下载）；单 symbol 失败进 journal 不中止批次；PDF 魔数校验；provider schema 漂移大声失败（fail fast，不静默吞字段）；指数成分运行时解析不硬编码。研报索引 CSV 经 `load_financial_data.py` 以 `if_exists="replace"` 写 financial.db。
- 验证：`tests/test_fetch_real_securities_data.py`（全 fake provider 离线）。
- 状态：✅

#### DC-054 SQLite WAL 强制契约（P2，ISSUE-18）
- 契约要点：统一连接入口 `connect_sqlite` 强制 `PRAGMA journal_mode=WAL` + `busy_timeout=5000`；只读连接（mode=ro）不得使用该入口；DDL 每库路径进程内只应用一次；同步 IO 移线程池不阻塞事件循环。
- 验证：`tests/test_sqlite_support.py`。
- 状态：✅

### 组 L：API / SSE / 前端契约

#### DC-055 七节点进度契约与 token 级真流式（P0，ISSUE-5/9 回归防线）
- 契约要点：`CLIENT_PROGRESS_NODES = {query_understand, planner, retrieve, grade_and_filter, reason, verify, compose}` 必须与 `frontend/src/types.ts` 的 `STREAM_NODES` key 集合相等且唯一（契约测试读前端源码断言，防前后端节点名失配导致进度不亮）；`answer_delta` 类型须先声明、ChatPage 必须消费且**不得含 setInterval 假打字机**；SSE 事件协议 progress/answer_delta/answer/error/done，answer 事件含 answer/citations/confidence/thread_id/turn_id，终态以 `terminal + final_answer` 判定；异常路径必发 done，客户端断连不发 done。
- 验证：`tests/test_stream_progress_contract.py`、`tests/e2e/test_e2e_qa.py`（TC-017）。
- 状态：✅

#### DC-056 dist fail-fast 与 legacy 路由移除（P1，ISSUE-7）
- 契约要点：`frontend/dist` 缺失时导入即抛 RuntimeError（提示 `cd frontend && npm run build`），不做静默 legacy 兜底；`/legacy` 路由与 `src.api.ui` 模块已删除。
- 验证：`tests/test_legacy_ui_removal.py`。
- 状态：✅

#### DC-057 QA API 错误码矩阵（P0）
- 契约要点（含 2026-09-29 后新增语义）：

| 码 | 触发 | detail 关键 |
|---|---|---|
| 401 | 无 Authorization / 非 Bearer 或空 token / 未知 token | missing bearer token / invalid authorization header / unknown demo token |
| 404 | thread 不存在或他人 thread（防探测同文案） | 会话不存在或不可访问 |
| 409 | 同 thread role/client 上下文变化 | 会话角色或客户上下文发生变化 |
| 422 | 空 query / >500 字 / 未知字段（extra=forbid） | Pydantic 校验 |
| 429 | 30 次/分钟滑动窗口 | 请求过于频繁 + Retry-After: 60（SSE 为 error 事件） |
| 503 | LLM provider 不可达（httpx 传输/超时、openai 连接类、APIStatusError 401/403/408/409/429/5xx） | LLM provider unavailable + 排查指引 |
| 504 | `asyncio.wait_for` 超过 `api_request_timeout_seconds` | 请求处理超时（Ns） |
| 500 | 兜底 | 内部处理错误 |

- 验证：`tests/e2e/test_e2e_auth.py`（TC-001）、`test_e2e_qa_input.py`（TC-002）、`test_e2e_qa.py`（TC-019/020/021/022）、`tests/test_api_auth.py`、`tests/test_api_routes.py`、`tests/test_api_main.py`。
- 状态：✅

#### DC-058 演练客户端契约与 S1~S5 实机场景（P1）
- 契约要点（`scripts/demo.py`）：connect/read 超时分离（`httpx.Timeout(connect=5.0, read=可覆盖默认 180.0, write=30.0, pool=5.0)`），`trust_env=False` 禁环境代理；授权场景断言 answer 以 `## 结论` 开头、含 R2 与 `[来源1]` 且不含 `[来源N]`、citations 非空且 quote 含 R2、compliance.passed、confidence∈{medium,high}；拒绝场景断言 answer 含"无权限"、citations==[]、flags 含 permission_denied、confidence=low；每场景最多重试 3 次吸收采样波动但断言不放宽；HTTP 200 不等于演练成功，任一断言失败 `sys.exit(1)`。
- 实机场景（S1~S5，2026-09-29 批次实测留档：S1 30.8s 通过、S2 拒绝 8.4s、S3 口径缺陷→ISSUE-22 已修、S4 3/3 fail-closed→ISSUE-21 已修、S5 首字 38.3s）：CHUNKER v2 重入库后 **S1~S5 未复测**。
- 验证：`tests/test_demo.py`；实机 `uv run python scripts/demo.py --base-url http://127.0.0.1:8001`。
- 状态：✅ 客户端契约；⬜ S1~S5 实机复测（与 DC-001/008 同批次执行）。

---

## 四、执行汇总（2026-09-30）

| 状态 | 数量 | 案例 |
|---|---|---|
| ✅ 自动化已守护 | 47 | DC-002/003/005~007、009/010/012~023、024~034、036~039、041、044、046/047、049~057 |
| ✅ 机制 + ⬜ 实机/真实流量 | 4 | DC-001、004、011、058（机制层绿，实机复测待跑） |
| ⬜ 需实机 | 3 | DC-008、043、045 |
| ❌ 未达标（已知） | 1 | DC-042 端到端 P95 |
| ⚠️ 缺口登记 | 3 | DC-035、040、048 |

证据：全量 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run python -m pytest -q` → 645 passed（HEAD 5df21a1）；
评估四件套 `artifacts/evaluation/2a6dfe1…/` 退出码全 0（DC-046 实测值）。

## 五、已知缺口与风险登记（不由案例承载，立项入口 todo/issues.md）

1. **resolve_followup_query 无单测**（DC-035）——建议下批 TDD 补齐（纯函数直测 + 图级传递断言）。
2. **审计 outbox 无重放**（DC-040）——审计库不可用期间事件只落盘不回灌；需立项重放器与幂等语义。
3. **消融/E2E 评估无退出码门槛**（DC-048）——与 impl-08 §4"非零退出码"要求不一致。
4. **性能 SLO 未达标/未实测**（DC-042/043/045）——端到端 P95 实测超标 2~4 倍，第一大头为
   query_understand 前置 LLM 往返；50 QPS 压测、单轮 ≤3s、回环 ≤3s、检索 P95 均待实机。
5. **缓存命中率 >80% 无真实流量证据**（DC-027）——机制与隔离语义已守护。
6. **评估集为语料机械生成**（DC-023/046 局限）——验证流水线机制与验证器行为，非人工标注质量；
   人工标注答案集是下一个质量台阶。
7. **检索"并行"注释与串行实现不符**——`HybridRetriever.retrieve` 按 plan step 顺序 for 循环执行
   （`src/retrieval/hybrid_retriever.py`），节点注释宣称并行；不影响正确性，影响延迟验收解读，
   立项时要么改注释要么真并行。
8. **CI 与完整安全审计未确认**——沿 issues.md 留档，不宣称项目安全。
