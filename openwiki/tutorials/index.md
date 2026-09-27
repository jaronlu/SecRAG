# 文件

- [知识入库链路：文件如何变成可检索证据](knowledge-ingestion.md) - 本页用一条文件入库链路解释分类预检、任务排队、快照校验、worker 租约、解析分块、稳定身份与权限元数据、Embedding、ChromaDB 写入、增量替换/归档与缓存失效，并说明这些 chunk 在问答时如何被权限感知检索。
- [问答请求执行链路：从 HTTP 到最终答案](request-execution.md) - 本页跟踪 POST /v1/assistant/qa 与 /v1/assistant/qa/stream 的一次完整执行，解释认证、限流、会话、LangGraph 外层图与 ReAct 子图、检索权限过滤、验证重推、合规阻断、语义缓存、持久化和审计，以及各个提前结束或重试分支。
