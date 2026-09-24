# 文件

- [知识入库链路：文件如何变成可检索证据](knowledge-ingestion.md) - 本页用一条文件入库链路解释任务排队、快照校验、解析分块、Embedding、ChromaDB 持久化和增量更新，并说明这些 chunk 如何在问答时被权限感知检索。
- [问答请求执行链路：从 HTTP 到最终答案](request-execution.md) - 本页跟踪 POST /v1/assistant/qa 的一次完整执行，解释认证、会话、LangGraph 状态图、检索、ReAct、验证、合规、持久化和审计，以及各个提前结束或重试分支。
