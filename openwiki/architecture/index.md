# 文件

- [API 表面与前端契约](api-surface.md) - 完整列出 SecRAG 的 HTTP 公共表面：Bearer demo token 认证、问答与 SSE 流式事件协议（progress/answer_delta/answer/error/done）、会话线程接口、technical 角色入库管理、admin 知识库与语义缓存管理、健康检查与 Prometheus 指标端点，以及 React 前端（唯一 UI）的消费契约、markdown 渲染与错误映射。
- [检索系统：混合检索与权限过滤](retrieval-system.md) - 本页解释从 Planner 检索计划到 HybridRetriever 执行、向量 + BM25/RRF 融合、双层权限过滤、grade_and_filter 排序与 BGE 语义重排的完整检索子系统，包括低召回放宽轮与多跳早停、分数语义、主来源优先、结果 TTL 缓存与入库失效联动。
- [状态、权限与安全边界](state-and-safety.md) - 本页用一条问答请求说明 AssistantState 如何贯穿 SecRAG 的各个节点，并区分认证、检索权限、工具授权、答案验证（含每轮验证快照与失败分类）、合规检查、会话隔离、缓存绑定和审计各自负责什么。
- [Agent 工具边界：可见性、授权与执行保护](tool-boundaries.md) - 完整覆盖 ReAct 工具子系统：工具注册表与 get_tools_for_role 的可见性规则（检索工具按 ROLE_ALLOWED_SOURCES 过滤并排除已满足检索源、非检索工具必须显式列入白名单）、authorize_reason_tool_call 的执行边界四重检查（角色授权、请求级截止时间、60 秒熔断器、10 秒单工具超时），以及 record_tool_results 的审计语义与失败输出不参与验证。
