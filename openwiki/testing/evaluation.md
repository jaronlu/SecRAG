---
type: testing reference
title: 测试与评估：pytest 布局、检索评估与权限冒烟
description: SecRAG 的质量门禁地图：tests/ 单元测试与 tests/e2e/ TC 编号端到端用例的布局与 isolated_stores 隔离 fixture，scripts/evaluate_retrieval.py 的 recall@5/recall@10/MRR/precision@5/覆盖率/权限拦截准确率指标与准入阈值，check_permissions.py 的 RBAC 冒烟检查，LLM-as-Judge 回答质量评估（evaluate_answers_e2e.py + answer_judge.py 四维度），消融实验（evaluate_ablation.py），以及内置评估集规模很小的局限。
tags: [testing, evaluation, pytest, e2e, retrieval-eval]
verified:
  - by: openwiki/0.5.2
    at: 2026-09-27T15:51:42.447Z
sources:
  - id: openwiki-source-ed40286a906dbbaceea992aa
    resource: repo://data/raw/demo_knowledge_base/samples/faq/sample_project_technical_faq.html.meta.json
  - id: openwiki-source-fbb432e9dbedac4cebc851d3
    resource: repo://docs/test-plans/e2e-test-cases.md
  - id: openwiki-source-05ccef8d4cf1698187f20464
    resource: repo://pyproject.toml
  - id: openwiki-source-843a58e144ca2fc962ad5954
    resource: repo://scripts/check_permissions.py
  - id: openwiki-source-1e9ad33096f9f9618cd93182
    resource: repo://scripts/evaluate_ablation.py
  - id: openwiki-source-a34a4e473e317f16da1b21ec
    resource: repo://scripts/evaluate_answers_e2e.py
  - id: openwiki-source-4e3e29048da8b73ea062c6f6
    resource: repo://scripts/evaluate_answers.py
  - id: openwiki-source-da2220eb8be1dde92a0cace0
    resource: repo://scripts/evaluate_compliance.py
  - id: openwiki-source-7602df0db9576a8c84bded81
    resource: repo://scripts/evaluate_conversations.py
  - id: openwiki-source-04483c0b1d79bc8b0b0b799f
    resource: repo://scripts/evaluate_retrieval.py
  - id: openwiki-source-4b6f137802064d92dc98ccaa
    resource: repo://scripts/evaluate_retrieval.sample.json
  - id: openwiki-source-3c89b4f9bf3b01d82af78fa3
    resource: repo://scripts/evaluation_common.py
  - id: openwiki-source-cca77dd9640cdd9315de9a9f
    resource: repo://src/evaluation/answer_judge.py
  - id: openwiki-source-7b9ce1115cc4c844eaa6a7cb
    resource: repo://src/evaluation/retrieval_eval.py
  - id: openwiki-source-da1cc862f5a540f793502703
    resource: repo://tests/e2e/conftest.py
  - id: openwiki-source-8a05ce1bcff67c5b4e484d81
    resource: repo://tests/e2e/test_e2e_retrieval.py
  - id: openwiki-source-039ee515ff96c365fcc2e43d
    resource: repo://tests/test_evaluate_ablation.py
  - id: openwiki-source-649dddfe0f55d389c9197669
    resource: repo://tests/test_evaluate_retrieval.py
generated: { by: "openwiki/0.5.2", at: "2026-09-27T15:51:42.447Z" }
---

# 测试与评估：pytest 布局、检索评估与权限冒烟

SecRAG 的质量验证分成两层：**pytest 测试**（单元 + 端到端，全替身、无外部依赖）和 **评估脚本**（带可复现产物的指标门槛与冒烟检查）。一句话记忆：pytest 回答“实现是否符合契约”，评估脚本回答“检索/回答/权限/合规在当前数据上是否达到准入线”。

> ⚠️ 本页描述的是架构验证原型，不是生产系统的质量证明。内置评估集（`scripts/evaluate_retrieval.sample.json`、`scripts/evaluate_answers.dataset.json`）与 demo token 只验证流程与链路，不证明生产安全性、吞吐量或回答质量；事件分级阈值、检索指标阈值等是启发式起点，未经真实标注数据标定。

## 1. 开发验证命令与依赖

```bash
uv sync --all-extras   # 或 --extra dev：安装 pytest/pytest-asyncio/ruff
uv run ruff check .
uv run pytest
```

`pyproject.toml` 把 `pytest>=9,<10`、`pytest-asyncio`、`ruff` 声明在 `[project.optional-dependencies].dev`；**裸 `uv sync` 会卸载未写入 pyproject 的包**，所以开发依赖必须用 `uv sync --all-extras` 或 `--extra dev`（`repo://pyproject.toml#L41-L46`）。pytest 配置 `asyncio_mode = "auto"`、`testpaths = ["tests"]`（`repo://pyproject.toml#L62-L64`）。

## 2. 测试布局

### 2.1 单元测试（`tests/`）

- `tests/test_agents.py`：Agent 节点、Graph 路由、验证/合规/工具边界与重排状态等最大的一组；
- `tests/test_hybrid_retriever.py`：计划级/结果级权限过滤、超量取回先于截断、RRF 排序与 grade_and_filter 单量纲排序不变量（mock 领域检索器，不碰真实 ChromaDB）；
- `tests/test_retriever.py` / `tests/test_date_filters.py`：向量检索距离→分数转换、领域检索器强制 `retrieval_source` 过滤、date_day 时间范围过滤契约；
- `tests/test_api_auth.py` / `tests/test_api_main.py` / `tests/test_api_routes.py`：认证与 FastAPI 路由/请求模型契约；
- 其余按领域分文件：`test_compliance.py`、`test_conversation.py`、`test_ingestion_*`、`test_semantic_cache.py`、`test_langfuse_*`、`test_event_grading.py`、`test_portfolio_store.py`、`test_evaluate_retrieval.py`、`test_evaluate_ablation.py`、`test_evaluation_scripts.py` 等。

### 2.2 端到端用例（`tests/e2e/`，TC 编号）

`docs/test-plans/e2e-test-cases.md` 定义 TC-001~TC-035 案例，`tests/e2e/` 按环节分文件实现：

- 环节 A 认证（TC-001/002）→ `test_e2e_auth.py`；
- 环节 B 文档入库（TC-003~010）→ `test_e2e_ingestion.py`（注入式 `IngestionService`：registry/ChromaDB/分类目录落 `tmp_path`，确定性假 embedding）；
- 环节 C 检索与权限过滤（TC-011~015）→ `test_e2e_retrieval.py`（`HybridRetriever` 的真实权限/容错逻辑 + stub 源检索器与爆炸 BM25/向量库替身）；
- 环节 D QA 与 SSE（TC-016~023）→ `test_e2e_qa.py`；
- 环节 E 合规与安全（TC-024~029）→ `test_e2e_compliance.py`；
- 环节 F 审计与语义缓存（TC-030~035）→ `test_e2e_audit_cache.py`。

### 2.3 conftest 隔离 fixture

`tests/e2e/conftest.py` 是共享 fixture 的家：原则是 **LLM / 向量检索 / 外部网络一律替换为测试替身，SQLite 存储、ChromaDB、注册表落到 `tmp_path`**，保证用例可重复运行、相互独立。

- `fake_llm`：按 prompt 关键词路由的假 LLM（query_understand/planner 走 JSON 契约，其余视为 ReAct 返回无工具调用的 AIMessage），替换 `agent_nodes.llm` 并清理 `_get_bound_reason_model` 的 lru_cache 防止跨用例污染；
- `isolated_stores`：会话/审计/outbox 落到 `tmp_path`，并把检索结果 TTL 缓存与 BM25 索引缓存（`result_cache.invalidate_retrieval_caches()`）在用例前后清理，防止跨用例污染；
- `fake_retriever_factory`：替换 `nodes.HybridRetriever`，测试通过 `holder["results"]` 注入结果或异常；
- `run_agent_graph`：`fake_llm + isolated_stores + fake_retriever_factory` 组合，返回可调用的图执行器（fresh checkpointer 保证用例独立）。

## 3. 检索评估：`scripts/evaluate_retrieval.py`

### 3.1 命令与数据

```bash
uv run python scripts/evaluate_retrieval.py                 # 默认使用 evaluate_retrieval.sample.json
uv run python scripts/evaluate_retrieval.py path/to/dataset.json --output-root artifacts/evaluation
```

评估集是 JSON 数组，每个样本给出 `query`、`user_role`、检索计划（`plan`，或 `source`，或 `expected_query_type` 推导）与 `relevant_chunk_ids` / `relevant_doc_ids`；需要预期被拒的样本用 `expected_permission_denied: true` 且相关文档为空列表（`repo://scripts/evaluate_retrieval.py#L55-L112`）。`_normalize_plan` 优先取显式 `plan`，其次按 `source`，最后按 `expected_query_type → source` 映射（如 `technical_inquiry → faq_search`）推导（`repo://scripts/evaluate_retrieval.py#L38-L45`）。

### 3.2 指标口径

对每个样本用真实 `HybridRetriever`（角色与 `ROLE_DATA_PERMISSIONS` 注入）执行计划后计算（`repo://scripts/evaluate_retrieval.py#L142-L202`）：

- **recall@5 / recall@10**：前 5 / 前 10 条结果中命中的相关 chunk 数 ÷ 相关 chunk 总数；
- **MRR**：第一个相关结果的倒数排名均值；
- **precision@5**：前 5 条命中数 ÷ 5；
- **覆盖率（coverage）**：前 10 条是否命中相关文档；**无相关文档标注的纯权限样本**以“实际拒绝 == 预期拒绝”计 1.0，且不拉低召回/MRR 指标（`tests/test_evaluate_retrieval.py::test_permission_only_samples_do_not_lower_retrieval_metrics` 锁定此语义）；
- **permission_block_accuracy**：`actual_permission_denied == expected_permission_denied` 的比例（`_has_permission_denied` 检查结果列表中是否出现 `denied` 结果）。

### 3.3 准入阈值与退出码

```text
recall@5 ≥ 0.80
recall@10 ≥ 0.90
permission_block_accuracy = 1.0
```

`admission_passed` 要求三个门槛同时满足（`repo://scripts/evaluate_retrieval.py#L48-L52`、`#L205-L206`）；`main` 在占位 chunk id 未替换或未过门槛时 `SystemExit(1)`（`repo://scripts/evaluate_retrieval.py#L224-L242`）。样本量只有 5 条，**只验证评估链路本身，不代表生产检索效果**（README 与 quickstart 均明确此局限）。

### 3.4 产物可复现

`scripts/evaluation_common.write_artifact` 把 `{commit_sha, dataset（仓库相对路径）, summary}` 写入 `artifacts/evaluation/<commit_sha>/<name>.json`（`repo://scripts/evaluation_common.py#L36-L59`）；`current_commit_sha` 失败时回落 `"uncommitted"`。所有评估脚本共用这套产物格式。

## 4. 权限冒烟：`scripts/check_permissions.py`

`check_permissions.py` **不调用 LLM**，直接验证角色感知检索与 post-retrieval grading（`grade_and_filter`）在本地 Chroma 数据上是否执行预期的可见性（`repo://scripts/check_permissions.py#L1-L5`）。它对三个场景各做一次 `_visible_count`（`HybridRetriever.retrieve` → `grade_and_filter` 后统计非 denied 结果数，`repo://scripts/check_permissions.py#L26-L34`）：

- `technical_langgraph_faq`：technical 角色查 `faq_search`（"LangGraph"）→ 可见 > 0（demo 的 `sample_project_technical_faq.html` 含 LangGraph 内容且 `allowed_roles` 含 technical，`repo://data/raw/demo_knowledge_base/samples/faq/sample_project_technical_faq.html.meta.json`）；
- `operations_langgraph_faq_blocked`：operations 角色查同一 FAQ → 0——operations 在 `ROLE_ALLOWED_SOURCES` 中有 FAQ 源（计划级放行），但该 LangGraph FAQ 文档的 `allowed_roles` 只有 technical，结果级过滤把它转成 denied；
- `sales_langgraph_report_blocked`：institutional_sales 查 `report_search`（"LangGraph"）→ 可见结果为 0——报告样例内容不匹配该查询，无可见结果（检查名沿用"被拦"语义，实测值由检索与权限共同决定）。

三项检查任一失败则以非零码退出。它验证的是“检索执行层 + grade_and_filter 的可见性契约”，与 `tests/e2e/test_e2e_retrieval.py` 的 TC-011/012 相互印证：越权数据只产生空 content 的 denied 占位，原文不进上下文。

## 5. LLM-as-Judge 回答质量评估

### 5.1 端到端评估：`scripts/evaluate_answers_e2e.py`

```bash
uv run python scripts/evaluate_answers_e2e.py                       # 默认数据集 + http://127.0.0.1:8000
uv run python scripts/evaluate_answers_e2e.py --dataset scripts/evaluate_answers.dataset.json --limit 10
```

流程：加载标注数据集（问题 + 角色 + 预期关键词/安全预期）→ 逐条用 demo token 调 `POST /v1/assistant/qa` → **LLM-as-Judge 四维度评分** → 规则-based 辅助检查 → 生成 JSON 详情 + Markdown 报告（`repo://scripts/evaluate_answers_e2e.py#L1-L15`、`#L293-L427`）。`ROLE_TO_TOKEN` 把 advisor/sales/compliance/ops/technical 映射到五个 demo token。

规则-based 检查（不依赖 LLM）：预期关键词命中率、PII 泄露（固定测试模式串）、权限拒绝（denied 引用或"权限/无权"字样）、合规标记、澄清触发、安全拦截、引用数量与错误统计（`repo://scripts/evaluate_answers_e2e.py#L97-L156`）。

### 5.2 评判器：`src/evaluation/answer_judge.py`

`AnswerJudge` 用配置的 LLM（openai / ollama provider，temperature 默认 0.0 保证可复现）从**四个维度**各打 1-5 分（`repo://src/evaluation/answer_judge.py#L36-L96`）：

1. **准确性 accuracy**：回答是否正确、事实是否准确；
2. **引用相关性 citation_relevance**：引用是否支持回答中的论断（引用为空给 1 分）；
3. **合规性 compliance**：是否含投资建议、目标价等违规内容；
4. **完整性 completeness**：是否覆盖问题的所有方面。

总体分是**加权平均**：准确性与合规性权重 2.0，引用相关性与完整性权重 1.0（`repo://src/evaluation/answer_judge.py#L343-L355`）；`passed = overall >= 3.0`。LLM 返回的 JSON 用正则抽取并钳位到 1-5，解析失败时该维度记 0 分（显式失败而非猜测）。`summarize` 产出各维度平均分、1-5 分布、通过率、低分案例（overall < 3，最多 5 条），`generate_markdown_report` 转 Markdown。

> 局限：judge 的输入是 `answer[:3000]` + 前 5 条引用的 `content[:500]`，且评判质量本身依赖所选 LLM——它衡量的是相对质量信号，不是绝对答案正确性。

## 6. 消融实验：`scripts/evaluate_ablation.py`

同一批问题分别跑四条路径，比较**正确率（预期关键词命中率）、拒答误判、延迟与单次 LLM 调用数（成本代理）**，回答“哪些问题值得支付复杂路径的成本”（`repo://scripts/evaluate_ablation.py#L1-L19`）：

- `direct_tool`：单轮检索原始结果，不过滤不重排，直接进 prompt；
- `plain_rag`：单轮检索 + 阈值过滤去重（`_grade_results`，与 `grade_and_filter` 的确定性子集共用 `RETRIEVAL_MIN_SCORE` / `GRADE_TOP_K` / 去重键契约，不做语义重排）；
- `rerank_rag`：plain_rag 基础上加 BGE 语义重排（复用 `nodes._try_rerank_candidates`）；
- `agent`：完整 Agent 图（规划/多跳/验证/合规）。

权限说明：简化路径与 Agent 路径使用**同一个** `HybridRetriever` 执行层角色过滤，实验不旁路权限边界。`_CountingLLM` 包装真实 LLM 统计调用次数（含 `bind_tools` 后的 bound runnable）。拒答判定用 `REFUSAL_MARKERS`（"无法安全返回"、"已停止输出"、"无权限访问"、"无法回答"）；有预期关键词标注却被拒答计为 `refusal_false_positive_rate`。无单一源映射的类别（multi_hop/ambiguous 等）查全部四源。`llm` / `retriever_factory` / `agent_runner` 可注入替身供测试（`tests/test_evaluate_ablation.py` 全部用替身组件，不依赖真实 LLM 与检索库）。

## 7. 附属评估脚本

- **`scripts/evaluate_answers.py`**：离线验证回答的引用、数字与幻觉指标——`ComprehensiveVerifier.verify` 的 `numeric_accuracy`、`citation_accuracy`（≥0.95）、`hallucination_rate`（≤0.05）、`expected_outcome_accuracy`（=1.0）作为准入门槛；
- **`scripts/evaluate_compliance.py`**：合规拦截准确率与受限内容泄漏率，准入要求 `block_accuracy == 1.0` 且 `leakage_rate == 0.0`；
- **`scripts/evaluate_conversations.py`**：确定性执行会话隔离、删除、request_id 幂等、审计完成与“当前轮只含本轮引用”五组检查，准入要求全部通过；
- **`src/evaluation/retrieval_eval.py`**：更早的检索质量脚本（内置 5 条用例，Hit@K/MRR/空结果率/平均分，CI 用 Hit@K < 60% 时非零退出）——与 `evaluate_retrieval.py` 并存，后者是按 chunk_id 标注的正式检索评估。

## 8. 局限与定位建议

- **内置评估集很小**：`evaluate_retrieval.sample.json` 只有 5 条且部分 `relevant_chunk_ids` 可能仍为占位（脚本检测 `replace_me_`/`example_`/`sample_` 前缀并提示）；`evaluate_answers.dataset.json` 约 25 条、标注的是关键词命中而非人工打分。这些集子只证明**评估链路可运行**，不能据此宣称生产 recall/MRR/回答质量。
- **demo token 与样例数据**只证明流程，不证明生产安全性、吞吐量或回答质量；固定 token 不能替代生产 IdP/签名 token/授权策略。
- **事件分级阈值**（P0/P1/P2，见 [数据与离线作业](../operations/data-and-jobs.md)）与检索准入阈值均为启发式起点，未经真实标注数据标定。
- **LLM-as-Judge 是相对信号**：评判模型、温度（默认 0.0）与截断长度都会影响分数；合规/准确率等硬门槛仍应配合规则-based 检查（`rule_based_checks`）共同判定。
- **定位建议**：想验证某次改动不破坏权限/检索/合规契约，跑 `uv run pytest`（重点 `tests/e2e` 的 TC 用例）；想量化检索或回答质量，跑评估脚本并核对产物 JSON；两者都不能替代真实数据上的标注集与人工评审。
