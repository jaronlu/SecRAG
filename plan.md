# SecRAG MVP 产品收敛 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. Each task ends with a focused test cycle. Do not reset, overwrite, or stage unrelated existing worktree changes.

**Goal:** 将 SecRAG 从同时覆盖问答、证券数据工具、文档运营、持仓扫描和评估平台的原型，收敛成一个面向 `operations` 角色的、带引用的内部知识问答产品；保留权限、引用、合规和审计安全边界，把其他能力切换为可回退的 `full` 配置或移出默认产品路径。

**Architecture:** 默认使用 `knowledge_qa` 产品 profile。该 profile 只向产品流程暴露产品文档和 FAQ 检索，用户通过认证上下文获得角色，问答界面只展示问题、回答、引用和明确的业务状态。现有完整 Agent、工具和管理接口先保留为 `full` profile 或技术入口，避免一次性重写造成安全回归；确认没有运行时引用后，再做物理清理。

**Tech Stack:** Python 3.11+, FastAPI, LangGraph, ChromaDB, Pydantic, SQLite, React 18, TypeScript, Vite, uv, pytest, ruff。

**Spec:** `README.md`, `docs/design/PRD.md`, `docs/design/architecture.md`, `docs/design/implementation-07-conversation-management.md`, `docs/design/implementation-09-ingestion-ui.md`。`docs/design/` 是指向 `~/llm-wiki/workshop/SecRAG/raw/` 的软链接，设计语义变更必须同步真实 Wiki 源文件。

## Global Constraints

- 使用 Python 3.11+，依赖继续由 `uv` 管理；不新增依赖。
- Python 代码遵循 `ruff`，行宽 100；固定结构的跨模块字典使用现有 `TypedDict`/Pydantic 模式。
- 认证、角色权限、`allowed_roles`、引用验证、合规检查和审计必须 fail closed；产品收敛不得把受限 chunk 送入 LLM。
- 默认 `PRODUCT_PROFILE=knowledge_qa`；设置 `PRODUCT_PROFILE=full` 可以恢复当前完整工具路径，便于回滚和对比。
- 默认试点角色为 `operations`，默认可用知识源为 `product_search` 和 `faq_search`；服务端仍保留完整角色矩阵，除非某任务明确修改它。
- API 不信任请求体中的身份或角色字段；前端不得提供角色选择器或可编辑 API Token。
- 不覆盖工作区已有改动：当前已有 `.env.example`、`src/agents/graph.py`、`src/agents/nodes.py`、`tests/test_agents.py` 的用户修改，以及 `gui-test-screenshots/` 未跟踪文件。每次修改前先检查 `git diff`，提交时只 stage 当前任务文件。
- 不执行全量 `uv run python -m pytest -q`，除非用户另行要求；默认运行受影响测试、`ruff` 和前端构建。
- 不删除数据库、重写 Git 历史、强制推送或修改生产环境；物理删除模块前必须先完成引用扫描和回退验证。
- 代码注释、docstring 和 commit message 使用英文；产品文案和用户可见状态使用中文。

## Review Focus

- **配置错误不能扩大权限：** `PRODUCT_PROFILE` 取非法值时启动/读取必须失败，不能静默切换到 `full`；由 Task 1 的 `test_unknown_profile_fails_closed` 覆盖。
- **请求体不能伪造身份：** `/v1/assistant/me` 和问答接口只使用 Bearer token 派生的角色；由 Task 2 的认证路由测试和现有 `test_assistant_request_rejects_removed_identity_fields` 覆盖。
- **默认 Agent 不能绕过产品范围：** 模型请求 `sql_query` 或生成市场/研报计划时必须被白名单和计划过滤同时拦截；由 Task 4 的工具授权与计划过滤测试覆盖。
- **失败终态不能伪装成成功答案：** 无证据、无权限、澄清和合规拦截必须输出明确 `answer_status`，且不能带成功置信度或未经验证引用；由 Task 5 和 Task 9 的契约测试覆盖。
- **前端错误不能泄露内部实现：** HTTP/SSE 错误、工具名、节点名、SQL 和异常正文不能出现在助手答案气泡中；由 Task 3 的前端行为约束、Task 5 的 SSE 契约测试和构建检查覆盖。

## Product Contract

第一版只承诺一个任务：运营/支持人员输入产品、FAQ 或操作流程问题，得到带 1–3 条原文引用的答案。

对外结果状态固定为：

| `answer_status` | 用户可见含义 | 终态条件 |
|---|---|---|
| `answered` | 已找到资料并返回答案 | 引用验证和合规检查通过 |
| `no_evidence` | 知识库没有可靠资料 | 检索耗尽且没有可用结果 |
| `permission_denied` | 当前身份没有访问完成请求所需资料的权限 | 所有结果被权限过滤 |
| `needs_clarification` | 问题缺少必要信息 | `clarify` 节点直接终止 |
| `blocked` | 请求或答案未通过安全/合规检查 | 合规拦截或验证失败兜底 |

用户界面不展示 `planner`、`grade_and_filter`、`reason`、`verify` 等内部节点，不展示完整 `audit_trail`、工具参数、模型名称或 SQL。服务端继续记录这些信息供审计和调试。

## File Map Before Implementation

- Create: `src/product_scope.py` — 产品 profile、默认角色、默认知识源和工具白名单。
- Modify: `src/config.py`, `.env.example` — profile 配置和回退开关。
- Create: `src/api/profile.py` — 返回当前认证上下文的最小接口。
- Modify: `src/api/auth.py`, `src/api/main.py`, `src/schemas/request_response.py` — 当前用户响应模型和路由注册。
- Modify: `src/agents/tools.py`, `src/agents/nodes.py` — profile 级工具与检索计划限制。
- Modify: `src/schemas/typed_dicts.py`, `src/api/qa.py` — 统一 `answer_status` 结果契约。
- Modify: `frontend/src/api.ts`, `frontend/src/types.ts`, `frontend/src/pages/ChatPage.tsx`, `frontend/src/components/ChatMessage.tsx`, `frontend/src/App.tsx` — 简化问答入口和身份显示。
- Remove after reference scan: `frontend/src/components/StreamingProgress.tsx` — 不再向用户展示内部节点进度。
- Test: `tests/test_product_scope.py`, `tests/test_api_auth.py`, `tests/test_api_routes.py`, `tests/test_agents.py`, `tests/test_tools.py`, `tests/test_stream_progress_contract.py`。
- Docs: `README.md`, `docs/design/PRD.md`, `docs/design/architecture.md` 的真实 Wiki 源文件；实现阶段不得只修改生成软链接内容。

---

### Task 1: 建立产品 profile 和默认范围

**Files:**

- Create: `src/product_scope.py`
- Modify: `src/config.py`
- Modify: `.env.example`
- Create: `tests/test_product_scope.py`

**Interfaces:**

- Produces `ProductProfile = Literal["knowledge_qa", "full"]`。
- Produces `MVP_PRIMARY_ROLE: Final[str] = ROLE_OPERATIONS`。
- Produces `PROFILE_ALLOWED_SOURCES: dict[ProductProfile, frozenset[str] | None]`，其中 `knowledge_qa` 为 `{SOURCE_PRODUCT, SOURCE_FAQ}`，`full` 为 `None`。
- Produces `PROFILE_ALLOWED_TOOLS: dict[ProductProfile, frozenset[str] | None]`，其中 `knowledge_qa` 为 `{"product_search", "faq_search"}`，`full` 为 `None`。
- Produces `get_product_profile(profile: str | None = None) -> ProductProfile`，无参数时读取 `config.product_profile`；非法值抛出 `ValueError`，不得静默回退到 `full`。
- Produces `allowed_sources_for_profile(profile: ProductProfile) -> frozenset[str] | None`。
- Produces `allowed_tools_for_profile(profile: ProductProfile) -> frozenset[str] | None`。

- [ ] **Step 1: 先写 profile 行为测试**

```python
def test_knowledge_qa_profile_allows_only_curated_sources():
    from src.product_scope import allowed_sources_for_profile
    from src.schemas.constants import SOURCE_FAQ, SOURCE_PRODUCT

    assert allowed_sources_for_profile("knowledge_qa") == {SOURCE_FAQ, SOURCE_PRODUCT}


def test_full_profile_has_no_product_source_restriction():
    from src.product_scope import allowed_sources_for_profile

    assert allowed_sources_for_profile("full") is None


def test_unknown_profile_fails_closed():
    import pytest
    from src.product_scope import get_product_profile

    with pytest.raises(ValueError):
        get_product_profile("unknown")
```

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run pytest tests/test_product_scope.py -q`

Expected: FAIL because `src.product_scope` and its profile helpers do not exist yet。

- [ ] **Step 3: 实现最小 profile 模块**

在 `src/product_scope.py` 中只声明上述类型、常量和纯函数；不要在此文件导入 Agent、Retriever 或前端模块，避免形成循环依赖。`full` profile 返回 `None` 表示不增加产品层限制，角色层权限仍由 `ROLE_ALLOWED_SOURCES` 执行。

- [ ] **Step 4: 加入配置**

在 `Settings` 中加入：

```python
product_profile: Literal["knowledge_qa", "full"] = "knowledge_qa"
```

在 `.env.example` 加入：

```dotenv
PRODUCT_PROFILE=knowledge_qa
```

配置解析失败时让 Pydantic 启动失败，不把错误值转换成 `full`。

- [ ] **Step 5: 运行测试和静态检查**

Run: `uv run pytest tests/test_product_scope.py tests/test_config.py -q`

Run: `uv run ruff check src/product_scope.py src/config.py tests/test_product_scope.py`

Expected: all targeted tests pass and ruff reports no error。

- [ ] **Step 6: Commit only this task**

```bash
git diff -- src/product_scope.py src/config.py .env.example tests/test_product_scope.py
git add src/product_scope.py src/config.py .env.example tests/test_product_scope.py
git commit -m "feat: add knowledge qa product profile"
```

如果 `git diff` 显示任务之外的用户改动，取消本次 stage，不要使用 `git add -A`。

### Task 2: 用认证上下文替代前端角色选择

**Files:**

- Create: `src/api/profile.py`
- Modify: `src/schemas/request_response.py`
- Modify: `src/api/main.py`
- Modify: `frontend/src/api.ts`
- Modify: `frontend/src/types.ts`
- Modify: `frontend/src/pages/ChatPage.tsx`
- Test: `tests/test_api_auth.py`, `tests/test_api_main.py`

**Interfaces:**

- Create `CurrentUserResponse` with fields `role: str`, `department: str`, `product_profile: str`。
- Add `GET /v1/assistant/me` requiring `authenticate_user`。
- Add `getCurrentUser(): Promise<CurrentUser>` in `frontend/src/api.ts`。

- [ ] **Step 1: 写认证接口测试**

在 `tests/test_api_auth.py` 保留纯函数认证测试；HTTP 路由测试放在已有 `tests/test_api_routes.py`，复用该文件的 `client` fixture，不要在 `tests/test_api_auth.py` 新建第二套 `TestClient` fixture。添加以下行为：

```python
def test_current_user_returns_server_derived_role(client):
    response = client.get(
        "/v1/assistant/me",
        headers={"Authorization": "Bearer demo-ops"},
    )

    assert response.status_code == 200
    assert response.json()["role"] == "operations"
    assert response.json()["department"] == "ops"


def test_current_user_requires_bearer_token(client):
    response = client.get("/v1/assistant/me")

    assert response.status_code == 401
```

如果执行时 fixture 被移动，先读取现有 API 测试并复用唯一的 `TestClient(app)` 来源；不要把认证测试改成直接调用路由函数，因为那会绕过 FastAPI 依赖注入和路由注册。

- [ ] **Step 2: 实现 response model 和 profile router**

`src/api/profile.py` 使用：

```python
@router.get("/v1/assistant/me", response_model=CurrentUserResponse)
async def current_user(user: AuthenticatedUser = Depends(authenticate_user)):
    return CurrentUserResponse(
        role=user.role,
        department=user.department,
        product_profile=config.product_profile,
    )
```

不要返回 Bearer token、原始请求头、完整 `user_id` 或权限列表。

- [ ] **Step 3: 注册 router 并实现前端类型/API**

在 `src/api/main.py` 中把 profile router 放在 frontend catch-all 之前。前端新增：

```ts
export interface CurrentUser {
  role: string
  department: string
  product_profile: 'knowledge_qa' | 'full'
}

export async function getCurrentUser(): Promise<CurrentUser> {
  return request<CurrentUser>('/v1/assistant/me')
}
```

- [ ] **Step 4: 删除身份选择交互**

在 `ChatPage.tsx`：

1. 删除 `ROLES` 数组、`handleTokenChange`、角色 `<select>`、设置弹窗中的 API Token `<input>`。
2. demo 默认 token 从 `demo-advisor` 改为 `demo-ops`，但只保留在开发客户端的请求兜底逻辑中，不在页面显示或允许编辑。
3. 页面加载时调用 `getCurrentUser()`，用固定中文映射显示“运营支持”；未知角色显示“已认证用户”，不把原始枚举当作营销文案。
4. 保留服务端身份校验，不在请求体增加角色字段。

- [ ] **Step 5: 验证**

Run: `uv run pytest tests/test_api_auth.py tests/test_api_main.py -q`

Run: `cd frontend && npm run build`

Expected: API 测试通过；前端 TypeScript 编译成功；页面源码中不再出现角色 `<select>`、API Token 编辑输入框或 `ROLES` 常量。

- [ ] **Step 6: Commit**

```bash
git add src/api/profile.py src/schemas/request_response.py src/api/main.py src/api/auth.py frontend/src/api.ts frontend/src/types.ts frontend/src/pages/ChatPage.tsx tests/test_api_auth.py tests/test_api_main.py
git commit -m "feat: derive chat identity from authenticated context"
```

### Task 3: 把问答界面收敛成一条主流程

**Files:**

- Modify: `frontend/src/pages/ChatPage.tsx`
- Modify: `frontend/src/components/ChatMessage.tsx`
- Modify: `frontend/src/api.ts`
- Modify: `frontend/src/types.ts`
- Remove after reference scan: `frontend/src/components/StreamingProgress.tsx`
- Modify: `frontend/src/App.tsx` only if the final route decision in this task requires it

**Interfaces:**

- UI 默认只使用 `streamQuestion`；不再在页面提供同步/流式切换。
- SSE 对外只消费 `answer_delta`、`answer`、`error`、`done`；`progress` 事件可以由后端暂时发送，但前端必须忽略。
- 错误使用独立的可操作错误状态，不把 `HTTP 500`、`Failed to fetch` 或异常堆栈拼进助手答案正文。

- [ ] **Step 1: 先扫描引用并记录删除边界**

Run: `rg -n "StreamingProgress|streamEnabled|currentNode|progressCompleted|askQuestion|STREAM_NODES" frontend/src tests`

Expected: 输出所有引用；只有确认 `StreamingProgress` 没有其他消费者后，才删除该文件。不得因为计划要求而删除仍被测试或其他页面使用的符号。

- [ ] **Step 2: 写前端行为约束**

在 `frontend/src/types.ts` 为 `ChatMessage` 增加：

```ts
error?: boolean
answer_status?: AnswerStatus
```

并定义：

```ts
export type AnswerStatus =
  | 'answered'
  | 'no_evidence'
  | 'permission_denied'
  | 'needs_clarification'
  | 'blocked'
```

在 `ChatMessage.tsx` 中，`error === true` 时使用 `role="alert"` 的错误块；不要使用 `dangerouslySetInnerHTML` 渲染错误文本。

- [ ] **Step 3: 删除进度和传输开关状态**

从 `ChatPage.tsx` 删除 `streamEnabled`、`currentNode`、`progressCompleted`、`StreamingProgress` 及设置弹窗中的流式切换。发送问题时始终走 `streamQuestion`，完成后只更新最终 `answer`、`citations`、`confidence`、`compliance` 和 `answer_status`。

- [ ] **Step 4: 处理错误和空状态**

网络错误、非 2xx、SSE 解析失败和服务端 `error` 事件都生成独立错误消息：

```ts
{
  id: assistantId,
  role: 'assistant',
  content: '请求未完成，请稍后重试。',
  error: true,
}
```

不把底层异常内容直接展示给用户。引用区、置信度和合规状态只在收到 `answer` 事件后渲染。

- [ ] **Step 5: 删除未使用组件并构建**

完成 `rg` 复查后删除 `frontend/src/components/StreamingProgress.tsx`。如果还有引用，先移除引用再删除，不要强制删除。

Run: `cd frontend && npm run build`

Expected: 构建通过，页面不再显示内部节点名称，不再有“流式输出”开关。

- [ ] **Step 6: Commit**

```bash
git add frontend/src/pages/ChatPage.tsx frontend/src/components/ChatMessage.tsx frontend/src/api.ts frontend/src/types.ts frontend/src/App.tsx
git add -u frontend/src/components/StreamingProgress.tsx
git commit -m "refactor: simplify chat to one user flow"
```

### Task 4: 在服务端执行 profile 级工具和检索限制

**Files:**

- Modify: `src/agents/tools.py`
- Modify: `src/agents/nodes.py`
- Modify: `src/agents/graph.py` only where authorization or tool binding requires it
- Modify: `tests/test_agents.py`
- Modify: `tests/test_tools.py`
- Modify: `tests/test_hybrid_retriever.py` if plan filtering changes its input contract

**Interfaces:**

- Extend `get_tools_for_role` to:

```python
def get_tools_for_role(
    user_role: str,
    excluded_retrieval_sources: set[str] | None = None,
    *,
    profile: ProductProfile | None = None,
) -> list[BaseTool]:
```

- Add a pure helper:

```python
def restrict_plan_to_profile(
    plan: list[RetrievalPlanStep],
    *,
    profile: ProductProfile,
) -> list[RetrievalPlanStep]:
```

It must remove sources outside the profile after role filtering. If all planned steps are removed, return one FAQ step using the resolved query so the graph reaches the existing no-result or permission terminal instead of silently expanding scope。

- [ ] **Step 1: 写工具白名单测试**

在 `tests/test_tools.py` 增加：

```python
def test_knowledge_qa_profile_exposes_only_knowledge_search_tools():
    from src.agents.tools import get_tools_for_role

    names = {tool.name for tool in get_tools_for_role('operations', profile='knowledge_qa')}

    assert names == {'product_search', 'faq_search'}


def test_full_profile_preserves_existing_role_filtering():
    from src.agents.tools import get_tools_for_role

    names = {tool.name for tool in get_tools_for_role('technical', profile='full')}

    assert 'product_search' in names
    assert 'faq_search' in names
```

- [ ] **Step 2: 写检索计划限制测试**

在 `tests/test_agents.py` 或新的 `tests/test_product_scope.py` 增加：

```python
def test_profile_filter_removes_market_and_sql_sources():
    from src.agents.nodes import restrict_plan_to_profile

    plan = [
        {'source': 'sql_query', 'query': '毛利率', 'top_k': 5, 'filters': None},
        {'source': 'faq_search', 'query': '操作流程', 'top_k': 5, 'filters': None},
    ]

    filtered = restrict_plan_to_profile(plan, profile='knowledge_qa')

    assert [step['source'] for step in filtered] == ['faq_search']
```

- [ ] **Step 3: 实现 profile 过滤**

在 `get_tools_for_role` 中先执行现有 `ROLE_ALLOWED_SOURCES` 角色过滤，再与 `allowed_sources_for_profile(profile)` 求交集；对 `knowledge_qa` 只返回 `product_search` 和 `faq_search`，不要把 `calculator`、`market_data_tool`、`sql_query_tool`、`financial_ratios_tool`、`suitability_check` 或 `rerank_tool` 作为默认工具暴露给模型。

在 `planner` 节点产出计划后调用 `restrict_plan_to_profile`。如果 profile 为 `full`，保持原有计划行为。不要用原始检索 score 代替 rerank 语义能力；未配置的 reranker 仍按现有显式 `unavailable` 语义处理。

- [ ] **Step 4: 守护授权层**

确认 `authorize_reason_tool_call` 与动态工具列表使用同一 profile 过滤结果。对一个模型直接请求 `sql_query` 的测试必须得到拒绝结果，且 SQL 不得执行；对 `knowledge_qa` 允许的两个检索工具仍能正常调用。

- [ ] **Step 5: 验证**

Run: `uv run pytest tests/test_tools.py tests/test_agents.py tests/test_hybrid_retriever.py -q`

Run: `uv run ruff check src/agents/tools.py src/agents/nodes.py src/agents/graph.py tests/test_tools.py tests/test_agents.py`

Expected: profile 过滤和原有权限测试都通过；没有新的 warning 或异常回退。

- [ ] **Step 6: Commit**

```bash
git add src/agents/tools.py src/agents/nodes.py src/agents/graph.py tests/test_tools.py tests/test_agents.py tests/test_hybrid_retriever.py
git commit -m "feat: constrain default agent to knowledge qa tools"
```

### Task 5: 统一 `answer_status` 对外契约

**Files:**

- Modify: `src/schemas/request_response.py`
- Modify: `src/schemas/typed_dicts.py`
- Modify: `src/api/qa.py`
- Modify: `frontend/src/types.ts`
- Modify: `frontend/src/pages/ChatPage.tsx`
- Modify: `frontend/src/components/ChatMessage.tsx`
- Test: `tests/test_api_routes.py`, `tests/test_agents.py`, `tests/test_stream_progress_contract.py`

**Interfaces:**

- Add `AnswerStatus = Literal[...]` using the five values in Product Contract。
- Extend `AnswerOutcome` with required `answer_status: AnswerStatus`。
- Extend `AssistantQAResponse` and SSE `answer` payload with `answer_status`。
- Add `answer_status_from_state(state: Mapping[str, Any]) -> AnswerStatus` in `src/api/qa.py` or a small `src/utils/answer_status.py` module. Prefer the utility module if the logic is independently testable。

- [ ] **Step 1: 写状态映射测试**

覆盖四个现有终态和合规拦截：

```python
def test_no_results_maps_to_no_evidence():
    assert answer_status_from_state({
        'terminal': True,
        'verification': {'issues': ['no_retrieval_results']},
    }) == 'no_evidence'


def test_permission_denied_maps_to_permission_denied():
    assert answer_status_from_state({
        'terminal': True,
        'verification': {'issues': ['permission_denied']},
    }) == 'permission_denied'


def test_clarification_maps_to_needs_clarification():
    assert answer_status_from_state({
        'terminal': True,
        'clarification_needed': True,
    }) == 'needs_clarification'
```

测试中的 key 必须引用 `src.schemas.constants`，不要在测试中重复创建新的业务常量。

- [ ] **Step 2: 实现状态映射并填充所有构造点**

状态优先级固定为：`needs_clarification` → `permission_denied` → `no_evidence` → `blocked` → `answered`。缓存命中只缓存已通过验证和合规的成功答案，因此命中路径固定为 `answered`，并在 `AnswerOutcome` 中显式填充，不依赖 Pydantic 默认值。

- [ ] **Step 3: 更新同步和 SSE 响应**

`_qa_response_from_outcome` 返回 `answer_status`。SSE 的 `answer` 事件增加同名字段；`error` 事件不伪装成业务终态。现有 `progress` 事件可以在本任务暂时保留，Task 3 的前端已经不再消费它。

- [ ] **Step 4: 更新前端状态文案**

在 `ChatMessage.tsx` 使用中文映射：

```ts
const ANSWER_STATUS_LABELS: Record<AnswerStatus, string> = {
  answered: '已回答',
  no_evidence: '未找到资料',
  permission_denied: '无权访问',
  needs_clarification: '需要补充信息',
  blocked: '已拦截',
}
```

状态标签必须有文字；`no_evidence`、`permission_denied`、`blocked` 不显示“置信度：高/中”。引用只在 `answered` 且引用数组非空时展示。

- [ ] **Step 5: 验证**

Run: `uv run pytest tests/test_api_routes.py tests/test_agents.py tests/test_stream_progress_contract.py -q`

Run: `cd frontend && npm run build`

Expected: 同步响应和 SSE 终态都有 `answer_status`；缓存、澄清、无结果、无权限和合规拦截路径不缺字段。

- [ ] **Step 6: Commit**

```bash
git add src/schemas/request_response.py src/schemas/typed_dicts.py src/api/qa.py frontend/src/types.ts frontend/src/pages/ChatPage.tsx frontend/src/components/ChatMessage.tsx tests/test_api_routes.py tests/test_agents.py tests/test_stream_progress_contract.py
git commit -m "feat: expose explicit answer status"
```

### Task 6: 将管理和入库能力移出普通用户主流程

**Files:**

- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/pages/ChatPage.tsx`
- Modify: `frontend/src/pages/AdminPage.tsx` only for technical-role guard and clear empty/error states
- Modify: `src/api/admin.py` only if route authorization is inconsistent with UI guard
- Modify: `frontend/src/api.ts` only if unused admin calls are removed from the default bundle
- Test: `tests/test_api_routes.py`, `tests/test_ingestion_api.py`

**Interfaces:**

- `/` is the only default product entry point。
- `/admin` remains a direct technical/admin route for the time being, but is not linked from the end-user chat page。
- `POST /v1/admin/ingestion/*` and document inspection endpoints remain independently authorized on the server。

- [ ] **Step 1: 写路由边界测试**

确认非 technical 用户访问文档管理和入库接口仍返回 403；technical 用户可以访问。不要把“前端隐藏入口”当作权限控制。

- [ ] **Step 2: 删除主流程入口**

从 `ChatPage.tsx` 删除“管理后台”导航链接；保留 `/admin` 路由，避免技术用户无法进入已有运维页面。不要把文档上传、任意路径浏览或 `full_scan` 添加到普通聊天页。

- [ ] **Step 3: 增加管理页身份守卫**

`AdminPage` 加载 `getCurrentUser()`。当角色不是 `technical` 时显示“当前账号没有知识库管理权限”，不调用文档、缓存或 chunk 接口；服务端仍执行独立认证和角色校验。

- [ ] **Step 4: 验证**

Run: `uv run pytest tests/test_api_routes.py tests/test_ingestion_api.py -q`

Run: `cd frontend && npm run build`

Expected: 普通用户的主导航只有问答；管理 API 权限行为不变且测试通过。

- [ ] **Step 5: Commit**

```bash
git add frontend/src/App.tsx frontend/src/pages/ChatPage.tsx frontend/src/pages/AdminPage.tsx frontend/src/api.ts src/api/admin.py tests/test_api_routes.py tests/test_ingestion_api.py
git commit -m "refactor: isolate admin workflows from chat"
```

### Task 7: 清理默认产品路径中的实验性能力

**Files:**

- Modify: `src/agents/tools.py`
- Modify: `src/api/main.py` only if a route is removed after consumer scan
- Modify: `frontend/src/api.ts`, `frontend/src/types.ts`
- Keep but do not import by default: `src/tools/market_data.py`, `src/tools/sql_query.py`, `src/tools/financial_ratios.py`, `src/tools/suitability.py`, `src/tools/rerank.py`, `src/portfolio/`, `src/jobs/`
- Test: all tests that reference the default tool registry or removed frontend symbols

- [ ] **Step 1: 做运行时引用扫描**

Run:

```bash
rg -n "market_data_tool|sql_query_tool|financial_ratios_tool|suitability_check|rerank_tool|portfolio|daily_scan|event_grading|askQuestion|STREAM_NODES" src frontend scripts tests
```

把结果分成三类：默认 `knowledge_qa` 路径、`full` profile 路径、仅评估/脚本/测试路径。不要根据文件名猜测“未使用”；必须按 import 和调用关系判断。

- [ ] **Step 2: 收敛默认 registry，不删除实验模块**

默认 profile 只注册两个知识检索工具；完整工具列表保留在 `full` profile。实验工具文件暂不删除，避免破坏现有评估脚本和设计验证。

- [ ] **Step 3: 删除前端未使用的 API 类型和方法**

只有当 `rg` 确认 `askQuestion`、`STREAM_NODES` 或管理 API 方法没有默认产品消费者时，才从前端源码删除；后端兼容路由暂不删除。

- [ ] **Step 4: 验证两个 profile**

Run: `PRODUCT_PROFILE=knowledge_qa uv run pytest tests/test_tools.py tests/test_agents.py -q`

Run: `PRODUCT_PROFILE=full uv run pytest tests/test_tools.py tests/test_agents.py -q`

Expected: 默认路径只暴露知识问答工具；full profile 的既有角色工具测试继续通过。

- [ ] **Step 5: Commit**

```bash
git add src/agents/tools.py frontend/src/api.ts frontend/src/types.ts
git commit -m "refactor: keep experimental tools out of default product"
```

### Task 8: 更新产品文档和 Wiki 设计源

**Files:**

- Modify: `README.md`
- Modify real source: `~/llm-wiki/workshop/SecRAG/raw/PRD.md`
- Modify real source: `~/llm-wiki/workshop/SecRAG/raw/architecture.md`
- Review only: `docs/design/PRD.md`, `docs/design/architecture.md`

- [ ] **Step 1: 更新 README 的产品定位**

把首段和核心能力改成“带引用的内部知识问答”，明确默认角色为 `operations`、默认知识源为 FAQ/产品文档、`full` profile 为开发回退。把行情、SQL、持仓扫描、每日事件、金融工具和文档运维标记为 deferred capabilities，而不是默认产品能力。

- [ ] **Step 2: 更新真实 PRD 源文件**

将用户故事收敛为：产品/流程问题问答、带引用回答、无证据拒答、权限拒答、歧义澄清、服务端审计。把研究员、基金经理、实时行情、结构化财务排名、持仓扫描和事件卡片移到“后续候选”，保留原始设计原因作为背景，不把它们写成当前验收标准。

- [ ] **Step 3: 更新真实 architecture 源文件**

增加 `knowledge_qa`/`full` profile 边界，说明 profile 过滤发生在角色权限过滤之后；明确前端只显示业务状态和引用，内部节点与工具调用留在审计/观测系统；说明 CLI 入库是默认运营路径，管理 UI 是技术入口。

- [ ] **Step 4: 检查 Wiki 工作区状态**

Run: `git -C ~/llm-wiki/workshop/SecRAG status --short`

确认修改落在真实 Wiki 仓库，并没有只改到生成软链接。不要手工编辑 `openwiki/` 生成页面。

- [ ] **Step 5: Commit separately**

仓库代码与 Wiki 是两个 Git 工作区。分别检查 diff 后提交：

```bash
git add README.md
git commit -m "docs: define knowledge qa product scope"

git -C ~/llm-wiki/workshop/SecRAG add raw/PRD.md raw/architecture.md
git -C ~/llm-wiki/workshop/SecRAG commit -m "docs: narrow SecRAG MVP product scope"
```

如果 Wiki 工作区还有无关用户改动，只 stage 两个设计源文件。

### Task 9: 建立 MVP 验收集并完成最小验证

**Files:**

- Create: `tests/evaluation/mvp_knowledge_qa.json`
- Create: `tests/test_mvp_contract.py`
- Modify: `README.md` with exact targeted verification commands
- Review: `docs/test-plans/agent-domain-test-cases.md`, `docs/test-plans/e2e-test-cases.md`

- [ ] **Step 1: 创建 8 条最小验收问题**

`tests/evaluation/mvp_knowledge_qa.json` 固定包含以下场景，每条指定 `query`、`expected_status`、`required_source_types`：

1. 产品说明中的风险等级 → `answered`、`product_search`。
2. FAQ 中的操作流程 → `answered`、`faq_search`。
3. 知识库不存在的实时行情问题 → `no_evidence`。
4. 明显歧义的产品简称问题 → `needs_clarification`。
5. 受限文档请求 → `permission_denied`。
6. 触发投资建议规则的问题 → `blocked`。
7. 带追问的同一会话 → 两轮都带 `answer_status` 且不跨用户串线。
8. 服务不可用/超时 → HTTP 错误或 SSE `error`，不渲染成 `answered`。

使用仓库现有 demo 文档中的稳定事实，避免把实时外部数据写入验收集。

- [ ] **Step 2: 写契约测试**

测试每条记录的 `answer_status`、引用存在性、来源类型和“无证据不得返回引用”。测试使用 mock LLM/检索 fixture，不调用外部 provider，不修改 `data/*.db`。

- [ ] **Step 3: 运行最小验证矩阵**

Run:

```bash
uv run pytest tests/test_product_scope.py tests/test_api_auth.py tests/test_api_routes.py tests/test_tools.py tests/test_agents.py tests/test_mvp_contract.py -q
uv run ruff check src tests
cd frontend && npm run build
```

Expected: targeted tests、ruff 和前端构建全部通过。若已有用户改动导致无关测试失败，保留完整失败输出，先判断是否为本次任务引入。

- [ ] **Step 4: 按项目规则做设计差异审计**

逐项检查：权限执行顺序未改变；引用和数字验证仍 fail closed；审计仍服务端持久化；入库仍禁止任意路径；`full` profile 可回退；前端不展示内部工具或 Token。

- [ ] **Step 5: Commit**

```bash
git add tests/evaluation/mvp_knowledge_qa.json tests/test_mvp_contract.py README.md
git commit -m "test: add knowledge qa mvp acceptance contract"
```

## Final Verification Checklist

执行者在宣布完成前必须逐项确认：

- [ ] `GET /v1/assistant/me` 从 Bearer token 服务端派生角色；前端没有角色选择器和 Token 编辑入口。
- [ ] 默认 `PRODUCT_PROFILE=knowledge_qa`，只暴露产品文档和 FAQ 检索；`PRODUCT_PROFILE=full` 可回退。
- [ ] `product_search`、`faq_search` 之外的工具不会进入默认 Agent 工具列表，也不会绕过授权层执行。
- [ ] 检索计划经过 profile 过滤；空计划进入已有 `no_results`/权限终态，不扩大到其他数据源。
- [ ] 同步响应和 SSE `answer` 事件都包含 `answer_status`。
- [ ] `no_evidence`、`permission_denied`、`needs_clarification` 和 `blocked` 不伪装成成功答案。
- [ ] 用户界面只展示回答、引用、业务状态和可操作错误；不展示 Agent 节点、工具名、SQL 或完整审计轨迹。
- [ ] `/admin` 不是普通聊天主流程的一部分，后端权限校验仍独立存在。
- [ ] 文档入库默认仍由受控 CLI/技术入口执行，没有新增任意路径、上传或 `full_scan` 能力。
- [ ] 目标测试、ruff 和前端构建通过；没有把未验证的全量测试结果称为通过。
- [ ] `git status --short` 和 `git -C ~/llm-wiki/workshop/SecRAG status --short` 中只剩下明确记录的计划/代码变更，没有覆盖用户已有修改。

## Recommended Execution Order

按 `Task 1 → Task 2 → Task 3 → Task 4 → Task 5 → Task 6 → Task 7 → Task 8 → Task 9` 顺序执行。每个任务完成其测试和检查后再进入下一个任务；如果 Task 4 的 profile 限制与现有用户对 `src/agents/graph.py`、`src/agents/nodes.py` 的未提交改动冲突，暂停该任务，先以 diff 为依据拆分冲突范围，不要覆盖已有改动。
