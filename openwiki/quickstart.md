---
type: getting started
title: 快速开始：从启动服务到第一次问答
description: 面向第一次接触 SecRAG 的开发者，按安装依赖、配置模型、导入样例数据、启动 FastAPI 和发送第一条问答请求的顺序给出可复制步骤，并指向后续执行链路教程。
tags: [quickstart, setup, fastapi, demo]
verified:
  - by: openwiki/0.5.2
    at: 2026-09-24T17:03:23.068Z
sources:
  - id: openwiki-source-5f5b95b3d6a215fa02ceb945
    resource: repo://.env.example
  - id: openwiki-source-05ccef8d4cf1698187f20464
    resource: repo://pyproject.toml
  - id: openwiki-source-23775c3de52f3ab95a13cb8b
    resource: repo://README.md
  - id: openwiki-source-d7ed0bb14d48547c9f500bad
    resource: repo://scripts/demo.py
  - id: openwiki-source-189ee3bddfce796fdd51b25a
    resource: repo://scripts/ingest.py
  - id: openwiki-source-53bdf62a9d0ee4ca3a837299
    resource: repo://src/api/auth.py
  - id: openwiki-source-9abd0efc90fa978f061bb160
    resource: repo://src/api/main.py
  - id: openwiki-source-d502c275990c6476221bf080
    resource: repo://src/config.py
generated: { by: "codex", at: "2026-09-24T17:03:23.068Z" }
---

# 快速开始：从启动服务到第一次问答

SecRAG 是一个机构投研场景的 Agentic RAG 原型。你可以先把它理解成三件事：

1. 把资料切成 chunk，向量化后放入 ChromaDB；
2. 收到问题后按角色检索证据，让 Agent 推理和调用工具；
3. 对答案做引用、数字、合规检查，再保存会话和审计记录。

## 1. 安装依赖

项目要求 Python 3.11+，使用 `uv` 管理环境：

```bash
uv sync --all-extras
```

这条命令会根据 `pyproject.toml` 和 `uv.lock` 创建或同步虚拟环境，并安装运行与测试依赖。

## 2. 创建配置文件

```bash
cp .env.example .env
```

### 选择 OpenAI-compatible 服务

默认配置是 OpenAI 兼容接口。把 `.env` 中的值换成你实际使用的服务：

```dotenv
LLM_PROVIDER=openai
OPENAI_API_BASE=https://your-provider.example/v1
OPENAI_MODEL=your-model
OPENAI_API_KEY=your-key
```

`LLM_PROVIDER=openai` 时没有 `OPENAI_API_KEY`，应用启动会直接报错。密钥不要提交到 Git。

### 选择本地 Ollama

如果本机已经运行 Ollama，可以改成：

```dotenv
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=llama3.1:8b
```

两种模式都使用 `EMBEDDING_MODEL` 做文档和问题的向量化，默认是 `BAAI/bge-small-zh-v1.5`。第一次运行可能需要下载模型；入库和检索必须保持同一个 embedding 模型。

## 3. 导入最小样例知识库

先把示例资料写入向量库：

```bash
uv run python scripts/ingest.py data/raw/demo_knowledge_base/samples/product product
uv run python scripts/ingest.py data/raw/demo_knowledge_base/samples/regulation regulation
uv run python scripts/ingest.py data/raw/demo_knowledge_base/samples/faq faq
uv run python scripts/ingest.py data/raw/demo_knowledge_base/samples/report research_report
```

每个源文件旁边都应有 `.meta.json` 权限清单。入库过程会解析文件、分块、生成 embedding，并将 chunk 的来源、版本和权限元数据写入 ChromaDB。重复执行时，未变化文件会被跳过；需要完整扫描并归档已删除文档时再加 `--full-scan`。

更详细的入库说明见：[知识入库链路](tutorials/knowledge-ingestion.md)。

## 4. 启动服务

```bash
uv run uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

启动后可以打开：

| 地址或接口 | 用途 |
| --- | --- |
| `http://127.0.0.1:8000/` | 问答和文档入库 UI |
| `http://127.0.0.1:8000/docs` | FastAPI 自动生成的 Swagger UI |
| `POST /v1/assistant/qa` | 问答接口 |
| `/v1/assistant/threads*` | 会话创建、查询和删除 |
| `/v1/admin/ingestion/*` | technical 角色的入库管理 |

## 5. 发送第一条问答请求

仓库内置 demo token，服务端会根据 token 绑定角色：

| Token | 角色 |
| --- | --- |
| `demo-advisor` | advisor |
| `demo-sales` | institutional_sales |
| `demo-compliance` | compliance |
| `demo-ops` | operations |
| `demo-tech` | technical |

示例：

```bash
curl -X POST http://127.0.0.1:8000/v1/assistant/qa \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer demo-tech' \
  -d '{"query":"系统操作流程怎么查？"}'
```

响应中最值得先看的字段是：

- `answer`：最终回答；
- `citations`：答案引用的证据；
- `confidence`：综合置信度；
- `compliance`：合规检查结果；
- `thread_id` / `turn_id`：会话和轮次标识。

完整审计链路不会通过这个接口返回，而是由服务端写入 SQLite。

## 6. 运行两个内置演示场景

保持服务运行，再开一个终端：

```bash
uv run python scripts/demo.py
```

脚本会演示一个允许查询和一个因角色权限受限的查询，并检查回答、引用、置信度和合规字段。

固定 demo token 只适合本地演示，不能替代生产环境的 IdP、签名 token 和授权策略。

## 7. 下一步按什么顺序读代码

建议按下面顺序：

1. 先读 [问答请求执行链路](tutorials/request-execution.md)，跟着 `POST /v1/assistant/qa` 走一遍；
2. 再读 [知识入库链路](tutorials/knowledge-ingestion.md)，理解证据如何进入 ChromaDB；
3. 最后读 [状态、权限与安全边界](architecture/state-and-safety.md)，理解每一层为什么要重复检查。

开发验证命令：

```bash
uv run ruff check .
uv run pytest
```

这个项目是架构验证原型，不是生产系统；样例数据和固定 token 仅用于本地学习与测试。
