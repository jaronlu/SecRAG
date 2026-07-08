# SecRAG 项目指令

## 项目概述

机构内部投研知识平台 — Agentic RAG 系统。完整设计文档见 `docs/design/`。
- `docs/design` 只是软连接
- 文档真实路径: `llm-wiki/raw/10-AI/projects/agentic-rag-securities/`
## 设计文档索引

| 文档 | 路径 |
|------|------|
| PRD | `docs/design/PRD.md` |
| 架构设计 | `docs/design/architecture.md` |
| 01-数据管道 | `docs/design/implementation-01-data-pipeline.md` |
| 02-基础 RAG | `docs/design/implementation-02-basic-rag.md` |
| 03-Agent 工作流 | `docs/design/implementation-03-agent-graph.md` |
| 04-多跳检索 | `docs/design/implementation-04-multi-hop-retrieval.md` |
| 05-业务工具 | `docs/design/implementation-05-financial-tools.md` |
| 06-引用与验证 | `docs/design/implementation-06-citation-verification.md` |
| 07-多轮会话与聊天持久化 | `docs/design/implementation-07-conversation-management.md` |
| 08-RAG 文档合并策略 | `docs/design/implementation-08-rag-document-combine.md` |

## 核心规则

0. **默认只答不操作**：默认只回答问题、给出建议；当用户明确要求修改、执行或 coding 时，可按设计驱动流程执行。所有高风险写/删/改/执行命令必须等用户明确确认后再动手。
0.1 **禁止弃用 API**：代码和设计文档中**禁止使用任何被标记为 Deprecated / Sunset 的 API**。第三方库优先使用当前稳定版的官方最新入口；涉及 LangChain 生态时，不再使用 `langchain-community`、`langchain` 老 Agent/TextSplitter 入口等已知弃用路径，改走对应独立包或社区明确迁移方案。
1. **文档先行**：当设计文档（Wiki）与代码不一致时，先更新 Wiki 设计文档，再改代码。设计驱动实现，不允许代码偏离后补文档。
2. **源码优先查本地**：涉及第三方库（LangChain, LangGraph, ChromaDB, FastAPI 等）的 API 签名、实现细节、版本差异时，先查 `~/llm-wiki/summaries/local-source-repos.md` 找到对应本地源码路径，再从本地源码中确认，不依赖记忆或网络搜索。
3. **代码自解释**：API 签名、docstring、测试即文档，不维护两份冗余内容。
4. **偏差记录**：实现中如果发现设计有问题，先在 Wiki 文档中修正设计，再回头改代码。同时在 `DESIGN.md` 中记录变更原因。

## 关键强约束

- **禁止使用任何 Deprecation / Sunset 标记 API**；必须使用当前稳定版官方最新 API。
- 遇到 `langchain-community` 被弃用提示时，不能只“忽略警告”，必须迁移到独立集成包或社区最新替代方案。
- 设计文档中的代码示例同样受此规则约束，发现一处即视为待修复问题。
- 强制令：设计驱动coding
- 先读相关设计文档，再读将修改的代码、引用、测试和配置。
- Coding 必须按设计文档实现；禁止把“当前已有代码”当成最终设计。
- 如果设计与代码不一致，默认改代码贴合设计；只有确认设计错误或不可落地时，先改设计文档并说明原因，再改代码。
- 发现设计中写的是 TODO、占位、Phase 2 或未完成能力时，不能把占位实现注册成已完成能力；必须二选一：
  - 补齐真实实现和测试；
  - 或保留未实现状态，并让工具/接口明确返回“未配置/未实现”，不得 silent fallback 冒充能力。
- 禁止用降级实现掩盖设计能力缺口。例如：文档要求 BGE reranker，就不能用原始 `score` 排序冒充语义重排。
- 高风险操作、引入新依赖、改动架构边界、删除数据或重写历史前必须先确认。

## 输出要求

- 先给结论，再说明原因、修改内容、验证结果和风险。
- 无法完全按设计实现时，必须明确说“不满足设计”，并列出缺口，不得包装成完成。

## 编码约定

- Python 3.11+，使用 uv 管理依赖
- 代码风格：ruff，行宽 100
- 测试：pytest + pytest-asyncio
- 结构固定的 `dict` 优先抽成 `TypedDict`，尤其是 Agent state、检索计划/结果、审计日志、API 序列化 payload、跨模块传递的数据结构；临时局部字典、第三方原始 payload、动态 metadata 可保持普通 `dict`

## 项目结构

```
src/
├── api/              # FastAPI 路由
├── agents/           # LangGraph 工作流节点
├── retrieval/        # 混合检索 + Reranker
├── ingestion/        # 文档加载、分块、Embedding
├── data/             # 行情、财务、资讯接口
└── utils/            # 追踪、引用、规则
tests/
docs/
scripts/
```
