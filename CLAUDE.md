# SecRAG 项目指令

## 项目概述

机构内部投研知识平台 — Agentic RAG 系统。完整设计文档见 `docs/design/`。

- `docs/design` 只是软连接
- 文档真实路径: `~/llm-wiki/workshop/SecRAG/raw/`
- 项目级设计同步规则: `~/llm-wiki/workshop/SecRAG/AGENTS.md`

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
| 08-系统评估与准入 | `docs/design/implementation-08-evaluation.md` |
| 参考-RAG 文档合并策略 | `docs/design/reference-rag-document-combine.md` |

## 规则优先级

1. 用户当前明确指令。
2. 本文件与 `~/llm-wiki/workshop/SecRAG/AGENTS.md`；两者冲突时，以更具体、风险更高的约束为准，并明确说明冲突。
3. `docs/design/` 中与任务相关的设计文档。
4. 当前实现仅用于理解现状，不得反向覆盖设计意图。

## 操作边界

- 默认只回答和执行只读检查；用户明确要求修改、执行或编码后，才进行对应的写操作。
- 普通代码编辑、测试和静态检查属于已授权任务的正常步骤，无需重复确认。
- 删除数据、破坏性 Git 操作、重写历史、修改生产环境或外部系统、处理密钥，以及其他不可逆或影响范围不明确的操作，必须先确认。
- 引入新依赖、改变架构边界或公开接口前，先说明必要性、影响和替代方案；需求未明确授权时先确认。

## 设计驱动

1. 编码前依次读取相关设计、待改代码、定义与引用、测试和配置。
2. 设计与实现不一致时：
   - 设计正确且可落地：修改实现、测试和配置以对齐设计，不反向修改设计迁就现状。
   - 设计错误、相互冲突或不可落地：先修订 Wiki 并记录原因，再同步实现。
3. `docs/design/` 的语义变更必须同步审计实现、依赖、数据模型、API、测试、评估脚本和运行配置。
4. 每条设计要求必须有实现责任和验证证据；TODO、占位、Phase 2 或 silent fallback 不得算作已实现。
5. 未实现能力只能二选一：补齐真实实现和测试；或明确返回“未配置/未实现”。不得用降级行为冒充设计能力，例如用原始 `score` 排序冒充 BGE 语义重排。
6. `docs/design/` 指向仓库外的 Wiki；修改后同时检查 Wiki 工作区状态，不能仅凭本仓库的 `git status` 判断变更是否已记录。

## 实现约束

- 遵循 Search → Plan → Execute → Verify → Report；不确定方案先做有时间边界的最小实验，再决定是否完整实现。
- 优先复用现有模式、依赖和约定；只做与任务直接相关的最小改动，不顺手重构或格式化无关代码。
- 修复根因；外部输入默认不可信，权限、合规、引用、数字验证和会话审计等安全边界必须 fail closed。
- 代码和设计示例不得使用当前依赖版本已标记为 Deprecated / Sunset 的 API。先查锁文件和本地源码确认受支持入口，不为追逐最新版擅自升级依赖。
- 涉及 LangChain 生态时，不使用已弃用的 `langchain-community` 或旧 Agent/TextSplitter 入口；采用项目现有独立集成包或官方迁移路径。
- 涉及第三方库 API 签名、实现细节或版本差异时，先查 `~/llm-wiki/domains/agent/summaries/local-source-repos.md` 定位本地源码，再以项目锁定版本为准；本地无法确认时查官方文档并说明依据。
- API 签名、必要的 docstring 和测试作为代码级说明；不维护与实现重复且容易失真的文档。

## 验证与报告

- 验证以行为为中心：优先运行能覆盖改动的测试，再按影响范围运行 `ruff`、完整测试和设计差异审计。
- 无法执行验证时，明确说明原因、未验证范围和剩余风险。
- 输出顺序：结论 → 原因 → 修改内容 → 验证结果 → 风险（如有）。
- 未完全满足设计时，必须明确标记“不满足设计”并列出缺口，不得包装成完成。

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
