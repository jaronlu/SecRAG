---
type: workflow tutorial
title: 知识入库链路：文件如何变成可检索证据
description: 本页用一条文件入库链路解释任务排队、快照校验、解析分块、Embedding、ChromaDB 持久化和增量更新，并说明这些 chunk 如何在问答时被权限感知检索。
tags: [ingestion, embeddings, chromadb, retrieval, workflow]
verified:
  - by: openwiki/0.5.2
    at: 2026-09-24T17:03:23.068Z
sources:
  - id: openwiki-source-189ee3bddfce796fdd51b25a
    resource: repo://scripts/ingest.py
  - id: openwiki-source-5574cff67cecb2bb8b5ee0de
    resource: repo://src/ingestion/chunkers.py
  - id: openwiki-source-526ddd73a007d1bcb9d714de
    resource: repo://src/ingestion/identity.py
  - id: openwiki-source-004bd813219e839a3cd50eec
    resource: repo://src/ingestion/pipeline.py
  - id: openwiki-source-a526102a3f2a67b2c4b65cc2
    resource: repo://src/ingestion/service.py
  - id: openwiki-source-e532544007c5ed049c805ecd
    resource: repo://src/retrieval/hybrid_retriever.py
  - id: openwiki-source-718743f2c00773741a22f6e9
    resource: repo://src/retrieval/vector_retriever.py
generated: { by: "codex", at: "2026-09-24T17:03:23.068Z" }
---

# 知识入库链路：文件如何变成可检索证据

“入库”就是把文件加工成带元数据的文本 chunk 和向量，保存到 ChromaDB；“检索”则是在问答时把问题也变成向量，找回最相关的 chunk。两者必须使用同一套身份、权限元数据和 embedding 模型，问答才有可靠证据。

## 1. 从 CLI 或 API 创建任务

最简单的 CLI 入口是：

```bash
uv run python scripts/ingest.py data/raw/demo_knowledge_base/announcements announcement
```

`scripts/ingest.py` 只负责解析参数，真正的工作由 `src/ingestion/pipeline.py` 的 `ingest_directory` 转给 `IngestionService`。

服务先做分类预检：确认目录位于项目的 `data/raw` 范围内、文件格式受支持、每个文件都有同名的 `.meta.json` 权限清单。API 管理页面走的是同一个 `IngestionService`，所以 CLI 和 HTTP 不会各自维护一套入库规则。

预检通过后，服务把待处理文件的相对路径、文件哈希、manifest 哈希、文档类型和顺序写入 SQLite `ingest_runs` / `ingest_run_files`，状态先是 `queued`。同一时间只允许一个活动任务；worker 领取任务后用租约和心跳防止两个进程同时处理。

## 2. 为什么要保存文件快照

排队和真正执行可能不是同一时刻。执行时 `_validate_snapshot` 会重新检查路径、文件哈希、manifest 哈希和文档类型。如果文件在排队后被替换或删除，这个文件会记录失败，不会把“排队时看到的旧内容”误当成当前内容。

`tests/test_ingestion_service.py` 专门验证了这一点：排队后新增的文件不会偷偷混入本轮快照；排队后的源文件变化会让任务失败且不调用真正的处理函数。

## 3. 文件先解析，再按文档类型分块

`src/ingestion/identity.py` 根据后缀选择 PDF、Word、HTML、CSV 或 Excel loader，把文件统一转换成 LangChain `Document`。随后 `chunk_documents` 按 `doc_type` 选择分块器：

- 研报和法规：默认约 500 字、重叠 100 字；
- 公告：约 300 字，适合短文本；
- 财务数据：约 800 字、重叠 200 字，尽量保留完整数据段；
- 会议纪要：约 400 字、重叠 80 字。

重叠区的作用是避免一句话刚好被切在两个 chunk 的边界，导致单独检索时上下文不完整。

## 4. 每个 chunk 都会补齐稳定身份和权限元数据

`derive_doc_id` 优先使用 manifest 中的 `doc_id`；没有时再按 URL、数据集内容或相对路径生成稳定 ID。每个 chunk 的 ID 由 `doc_id + chunk_index + 内容` 计算，内容变了，ID 通常也会变化。

`normalize_chunks` 会给 chunk 写入：

- `doc_id`、`chunk_id`、`chunk_index` 和文档版本；
- 文件、manifest、解析和 chunk 内容的哈希；
- `source`、标题、日期、股票代码等检索字段；
- `permission_level`、`allowed_roles` 和 `retrieval_source`；
- parser/chunker/embedding 模型版本。

manifest 不是可选备注：`load_sample_metadata` 会校验权限等级、角色列表和 `doc_type` 与检索源的对应关系。非公开文档没有 `allowed_roles` 会在入库前就失败。

## 5. Embedding 和 ChromaDB 写入

`get_embedding_model` 优先从本地 HuggingFace 缓存加载配置的模型，缓存没有才尝试下载。`upsert_chunks` 使用稳定的 chunk ID 分批写入 ChromaDB，并在 collection metadata 里记录 embedding 模型名称。

启动 `ChromaVectorRetriever` 或入库时都会检查模型名称。如果 ChromaDB 记录的模型和当前 `.env` 配置不一致，代码会抛出错误阻止继续，因为不同模型产生的向量不在同一个空间，检索分数没有意义。切换 embedding 模型需要重新入库。

## 6. 增量更新、跳过和旧 chunk 清理

入库 registry 会保存文件哈希、manifest 哈希、解析器版本、分块器版本、embedding 模型、chunk 数和文档版本。只有这些信息都没有变化，`_should_skip` 才把文件标记为 `skipped`，避免重复解析和向量化。

如果内容或版本发生变化，任务会走 `replaced`：

1. 重新解析、分块并生成新的 chunk；
2. 先 upsert 新 chunk；
3. 找出该 `doc_id` 旧有但新版本不再存在的 chunk ID；
4. 删除这些 stale chunk；
5. 在 registry 中写入新哈希、版本和 chunk 数。

`--full-scan` 还会把目录中已经消失的旧文档标记为 `archived`，并删除它们的向量；普通增量扫描不会因为目录里少了一个文件就自动归档。

## 7. 问答时如何把 chunk 找回来

问答链路中的 `HybridRetriever` 根据 Planner 的检索计划选择 product、regulation、report 或 FAQ 检索器。每个领域检索器共享一个 `ChromaVectorRetriever`：

1. 用同一个 embedding 模型把查询转成向量；
2. 在 ChromaDB 中按 query 和 filters 找候选 chunk；
3. 如果 BM25 索引可用，再把关键词结果与向量结果做 RRF 融合；
4. BM25 构建或查询失败时，明确退化为纯向量结果；
5. 最后按角色对结果 metadata 做权限过滤。

ChromaDB 的 distance 会在 `ChromaVectorRetriever._format` 中转换成统一的 `score = 1 - distance`，上游节点因此只需要处理项目自己的 `RetrievalResult` 格式。

## 8. 一张图记住整条链路

```text
文件 + .meta.json
      │
      ├─ 预检、路径安全、权限清单校验
      ├─ 建立任务快照（路径/哈希/顺序）
      ├─ 解析为 Document
      ├─ 按 doc_type 分块
      ├─ 补齐 doc_id/chunk_id/权限/版本元数据
      ├─ embedding → ChromaDB upsert
      └─ registry 记录 created / skipped / replaced / archived / failed

用户问题
      └─ HybridRetriever：向量 + BM25/RRF → 元数据权限过滤 → 返回证据 chunk
```

读这条链路时，最容易忽略的三个不变量是：入库与检索必须使用同一 embedding 模型；chunk 必须带可追踪的稳定 ID 和版本信息；权限元数据必须在检索结果进入 LLM 前完成过滤。
