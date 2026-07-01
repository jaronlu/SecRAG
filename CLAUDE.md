# SecRAG 项目指令

## 项目概述

券商内部投研知识平台 — Agentic RAG 系统。完整设计文档见 `docs/design/`。

## 设计文档索引

| 文档 | 路径 |
|------|------|
| PRD | `docs/design/PRD.md` |
| 架构设计 | `docs/design/architecture.md` |
| 01-数据管道 | `docs/design/implementation-01-data-pipeline.md` |
| 02-基础 RAG | `docs/design/implementation-02-basic-rag.md` |
| 03-Agent 工作流 | `docs/design/implementation-03-agent-graph.md` |
| 04-多跳检索 | `docs/design/implementation-04-multi-hop-retrieval.md` |
| 05-金融工具 | `docs/design/implementation-05-financial-tools.md` |
| 06-引用与验证 | `docs/design/implementation-06-citation-verification.md` |

## 核心规则

0. **只答不操作**：LLM 只回答问题、给出建议，不直接执行任何文件操作。所有写/删/改/执行命令必须等用户明确确认后再动手。
0.1 **禁止弃用 API**：代码和设计文档中**禁止使用任何被标记为 Deprecated / Sunset 的 API**。第三方库优先使用当前稳定版的官方最新入口；涉及 LangChain 生态时，不再使用 `langchain-community`、`langchain` 老 Agent/TextSplitter 入口等已知弃用路径，改走对应独立包或社区明确迁移方案。
1. **文档先行**：当设计文档（Wiki）与代码不一致时，先更新 Wiki 设计文档，再改代码。设计驱动实现，不允许代码偏离后补文档。
2. **源码优先查本地**：涉及第三方库（LangChain, LangGraph, ChromaDB, FastAPI 等）的 API 签名、实现细节、版本差异时，先查 `~/llm-wiki/summaries/local-source-repos.md` 找到对应本地源码路径，再从本地源码中确认，不依赖记忆或网络搜索。
3. **代码自解释**：API 签名、docstring、测试即文档，不维护两份冗余内容。
4. **偏差记录**：实现中如果发现设计有问题，先在 Wiki 文档中修正设计，再回头改代码。同时在 `DESIGN.md` 中记录变更原因。

## 编码约定

- Python 3.11+，使用 uv 管理依赖
- 代码风格：ruff，行宽 100
- 测试：pytest + pytest-asyncio

## 项目结构

```
src/
├── api/              # FastAPI 路由
├── agents/           # LangGraph 工作流节点
├── retrieval/        # 混合检索 + Reranker
├── ingestion/        # 文档加载、分块、Embedding
├── data/             # 行情、财务、资讯接口
└── utils/            # 审计、引用、合规
tests/
docs/
scripts/
```
