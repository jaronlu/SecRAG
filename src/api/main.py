import logging
import uuid
from html import escape

import httpx
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from langchain_core.runnables.config import RunnableConfig
from openai import APIConnectionError, APIStatusError, APITimeoutError

from src.api.auth import (
    TOKEN_USER_BINDINGS,
    AuthenticatedUser,
    authenticate_user,
    build_assistant_initial_state,
)
from src.config import config
from src.rag.chain import build_rag_chain, format_docs
from src.rag.formatter import estimate_confidence, format_citations
from src.retrieval.vector_retriever import ChromaVectorRetriever
from src.schemas.constants import (
    META_SOURCE,
    ROLE_ADVISOR,
    ROLE_COMPLIANCE,
    ROLE_INSTITUTIONAL_SALES,
    ROLE_OPERATIONS,
    ROLE_TECHNICAL,
    RR_METADATA,
    STATE_AUDIT_TRAIL,
    STATE_CITATIONS,
    STATE_COMPLIANCE,
    STATE_CONFIDENCE,
    STATE_FINAL_ANSWER,
)
from src.schemas.request_response import (
    AssistantQARequest,
    AssistantQAResponse,
    QARequest,
    QAResponse,
)

app = FastAPI(title="机构内部投研知识平台", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


ROLE_UI_OPTIONS = {
    ROLE_TECHNICAL: "技术支持",
    ROLE_ADVISOR: "投顾",
    ROLE_INSTITUTIONAL_SALES: "机构销售",
    ROLE_COMPLIANCE: "合规",
    ROLE_OPERATIONS: "运营",
}


def _render_identity_options() -> str:
    options = []
    token_by_role = {}
    duplicate_roles = set()
    for token, user in TOKEN_USER_BINDINGS.items():
        if user.role in token_by_role:
            duplicate_roles.add(user.role)
        token_by_role[user.role] = token

    missing_roles = [role for role in ROLE_UI_OPTIONS if role not in token_by_role]
    extra_roles = [role for role in token_by_role if role not in ROLE_UI_OPTIONS]
    if duplicate_roles or missing_roles or extra_roles:
        raise RuntimeError(
            "UI role options and demo token roles are inconsistent: "
            f"duplicates={sorted(duplicate_roles)}, missing={missing_roles}, extra={extra_roles}"
        )

    for role, label in ROLE_UI_OPTIONS.items():
        token = token_by_role[role]
        options.append(
            '<option value="{token}" data-role="{role}">{label} ({role})</option>'.format(
                token=escape(token),
                role=escape(role),
                label=escape(label),
            )
        )
    return "\n".join(options)


@app.get("/", response_class=HTMLResponse)
async def ui():
    return """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>SecRAG</title>
  <style>
    :root {
      color-scheme: light;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f6f7f9;
      color: #1f2937;
    }
    body {
      margin: 0;
      padding: 32px;
    }
    main {
      max-width: 960px;
      margin: 0 auto;
    }
    h1 {
      margin: 0 0 20px;
      font-size: 28px;
      letter-spacing: 0;
    }
    .panel {
      background: #fff;
      border: 1px solid #d7dce2;
      border-radius: 8px;
      padding: 20px;
    }
    label {
      display: block;
      margin-bottom: 8px;
      font-weight: 600;
      font-size: 14px;
    }
    textarea,
    select,
    input {
      width: 100%;
      box-sizing: border-box;
      border: 1px solid #b9c2cc;
      border-radius: 6px;
      padding: 10px 12px;
      font: inherit;
      background: #fff;
    }
    textarea {
      min-height: 132px;
      resize: vertical;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 14px;
      margin: 16px 0;
    }
    button {
      border: 0;
      border-radius: 6px;
      padding: 11px 16px;
      font-weight: 700;
      cursor: pointer;
      background: #2563eb;
      color: #fff;
    }
    button:disabled {
      cursor: not-allowed;
      opacity: 0.6;
    }
    pre {
      white-space: pre-wrap;
      word-break: break-word;
      background: #111827;
      color: #e5e7eb;
      border-radius: 8px;
      padding: 16px;
      min-height: 180px;
      overflow: auto;
    }
    .result-sections {
      display: grid;
      gap: 14px;
    }
    .output-section {
      background: #fff;
      border: 1px solid #d7dce2;
      border-radius: 8px;
      padding: 16px;
    }
    .output-section h3 {
      margin: 0 0 10px;
      font-size: 16px;
    }
    .answer-text {
      white-space: pre-wrap;
      line-height: 1.65;
    }
    .meta-row {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 12px;
    }
    .pill {
      border: 1px solid #c8d0da;
      border-radius: 999px;
      padding: 4px 10px;
      font-size: 13px;
      color: #374151;
      background: #f8fafc;
    }
    .citation-list {
      margin: 0;
      padding-left: 20px;
    }
    .citation-list li {
      margin-bottom: 10px;
    }
    details summary {
      cursor: pointer;
      font-weight: 700;
    }
    .toolbar {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      margin-top: 16px;
    }
    .result-heading {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      margin-top: 24px;
    }
    .result-heading h2 {
      margin: 0;
    }
    .copy-tools {
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .icon-button {
      width: 36px;
      height: 36px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      padding: 0;
    }
    .icon-button svg {
      width: 18px;
      height: 18px;
      stroke: currentColor;
    }
    .status {
      font-size: 14px;
      color: #4b5563;
    }
    @media (max-width: 760px) {
      body { padding: 18px; }
      .grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <main>
    <h1>SecRAG</h1>
    <section class="panel">
      <label for="query">查询问题</label>
      <textarea id="query" placeholder="输入你的问题，例如：系统操作流程怎么查？"></textarea>
      <div class="grid">
        <div>
          <label for="token">身份</label>
          <select id="token">
            __IDENTITY_OPTIONS__
          </select>
          <div id="roleBinding" class="status"></div>
        </div>
        <div>
          <label for="clientId">客户 ID（可选）</label>
          <input id="clientId" placeholder="client id" />
        </div>
        <div>
          <label for="threadId">会话 ID（可选）</label>
          <input id="threadId" placeholder="thread id" />
        </div>
      </div>
      <div class="toolbar">
        <button id="submit" type="button">提交查询</button>
        <span id="status" class="status">Ready</span>
      </div>
    </section>
    <div class="result-heading">
      <h2>结果</h2>
      <div class="copy-tools">
        <span id="copyStatus" class="status"></span>
        <button id="copyResult" class="icon-button" type="button" aria-label="复制结果" title="复制结果">
          <svg viewBox="0 0 24 24" fill="none" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
            <rect x="8" y="8" width="12" height="12" rx="2"></rect>
            <path d="M16 8V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h2"></path>
          </svg>
        </button>
      </div>
    </div>
    <section id="emptyResult" class="output-section">等待查询...</section>
    <div id="resultSections" class="result-sections" hidden>
      <section class="output-section">
        <h3>Answer</h3>
        <div id="answerText" class="answer-text"></div>
        <div class="meta-row">
          <span id="confidenceText" class="pill"></span>
          <span id="complianceText" class="pill"></span>
        </div>
      </section>
      <section class="output-section">
        <h3>Citations</h3>
        <ol id="citationList" class="citation-list"></ol>
      </section>
      <details class="output-section">
        <summary>Audit Trail</summary>
        <pre id="auditText"></pre>
      </details>
      <details class="output-section">
        <summary>Raw JSON</summary>
        <pre id="rawJson"></pre>
      </details>
    </div>
  </main>
  <script>
    const submit = document.getElementById("submit");
    const copyResult = document.getElementById("copyResult");
    const tokenSelect = document.getElementById("token");
    const roleBinding = document.getElementById("roleBinding");
    const statusEl = document.getElementById("status");
    const copyStatusEl = document.getElementById("copyStatus");
    const emptyResult = document.getElementById("emptyResult");
    const resultSections = document.getElementById("resultSections");
    const answerText = document.getElementById("answerText");
    const confidenceText = document.getElementById("confidenceText");
    const complianceText = document.getElementById("complianceText");
    const citationList = document.getElementById("citationList");
    const auditText = document.getElementById("auditText");
    const rawJson = document.getElementById("rawJson");
    let lastResultText = "";

    function updateRoleBinding() {
      const selected = tokenSelect.options[tokenSelect.selectedIndex];
      roleBinding.textContent = `绑定 user_role: ${selected.dataset.role}`;
    }

    tokenSelect.addEventListener("change", updateRoleBinding);
    updateRoleBinding();

    function renderResult(data) {
      emptyResult.hidden = true;
      resultSections.hidden = false;

      const answer = data.answer || data.detail || data.raw || "";
      const citations = Array.isArray(data.citations) ? data.citations : [];
      const audit = data.audit_trail || {};
      const compliance = data.compliance || {};

      answerText.textContent = answer || "无回答内容";
      confidenceText.textContent = `confidence: ${data.confidence || "n/a"}`;
      complianceText.textContent = `compliance: ${compliance.passed === false ? "blocked" : "passed/unknown"}`;

      citationList.replaceChildren();
      if (citations.length === 0) {
        const item = document.createElement("li");
        item.textContent = "无引用";
        citationList.appendChild(item);
      } else {
        for (const citation of citations) {
          const item = document.createElement("li");
          const title = citation.doc_title || "未知文档";
          const source = citation.source || "";
          const quote = citation.quote || "";
          item.textContent = `${title} — ${source}\n${quote}`;
          citationList.appendChild(item);
        }
      }

      auditText.textContent = JSON.stringify(audit, null, 2);
      rawJson.textContent = JSON.stringify(data, null, 2);
      lastResultText = [
        `Answer:\n${answer}`,
        `Citations:\n${citations.map((c, i) => `${i + 1}. ${c.doc_title || "未知文档"} — ${c.source || ""}`).join("\n") || "无引用"}`,
        `Audit Trail:\n${JSON.stringify(audit, null, 2)}`
      ].join("\n\n");
    }

    submit.addEventListener("click", async () => {
      const query = document.getElementById("query").value.trim();
      const token = tokenSelect.value;
      const clientId = document.getElementById("clientId").value.trim();
      const threadId = document.getElementById("threadId").value.trim();

      if (!query) {
        statusEl.textContent = "请输入查询问题";
        return;
      }

      const body = { query };
      if (clientId) body.client_id = clientId;
      if (threadId) body.thread_id = threadId;

      submit.disabled = true;
      statusEl.textContent = "查询中...";
      emptyResult.hidden = false;
      emptyResult.textContent = "查询中...";
      resultSections.hidden = true;
      lastResultText = "";

      try {
        const response = await fetch("/v1/assistant/qa", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "Authorization": `Bearer ${token}`
          },
          body: JSON.stringify(body)
        });
        const text = await response.text();
        let data;
        try {
          data = JSON.parse(text);
        } catch {
          data = { raw: text };
        }
        renderResult(data);
        statusEl.textContent = response.ok ? "完成" : `请求失败：${response.status}`;
      } catch (error) {
        renderResult({ raw: String(error) });
        statusEl.textContent = "请求失败";
      } finally {
        submit.disabled = false;
      }
    });

    copyResult.addEventListener("click", async () => {
      const text = lastResultText.trim();
      if (!text) {
        copyStatusEl.textContent = "暂无可复制内容";
        return;
      }

      try {
        await navigator.clipboard.writeText(text);
        copyStatusEl.textContent = "已复制";
      } catch {
        const textarea = document.createElement("textarea");
        textarea.value = text;
        textarea.style.position = "fixed";
        textarea.style.opacity = "0";
        document.body.appendChild(textarea);
        textarea.focus();
        textarea.select();
        const copied = document.execCommand("copy");
        textarea.remove();
        copyStatusEl.textContent = copied ? "已复制" : "复制失败";
      }
    });
  </script>
</body>
</html>""".replace("__IDENTITY_OPTIONS__", _render_identity_options())


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)


@app.get("/.well-known/appspecific/com.chrome.devtools.json", include_in_schema=False)
async def chrome_devtools_probe():
    return {}


# 构建 RAG 链（langchain LCEL），全局复用，避免每次请求重新编译
rag_chain = build_rag_chain()

# 检索器全局复用，避免反复创建 ChromaDB 客户端
retriever = ChromaVectorRetriever(persist_directory=config.chroma.persist_directory)

# 追踪日志记录器（结构化 JSON，可对接 ELK / Loki）
audit_logger = logging.getLogger("secrag.audit")


@app.post("/v1/qa", response_model=QAResponse)
async def qa(request: QARequest):
    request_id = str(uuid.uuid4())

    try:
        # 1. 检索（仅一次，结果同时用于生成和引用格式化）
        retrieval_results = retriever.retrieve(
            query=request.query,
            top_k=request.top_k,
        )

        # 2. 将检索结果格式化为 context 传入生成链，避免链内重复检索
        context = format_docs(retrieval_results)
        answer = rag_chain.invoke({
            "question": request.query,
            "context": context,
        })

        # 3. 格式化引用与置信度
        citations = format_citations(retrieval_results)
        confidence = estimate_confidence(retrieval_results)

        # 4. 追踪日志
        audit_logger.info({
            "request_id": request_id,
            "query": request.query,
            "top_k": request.top_k,
            "retrieval_count": len(retrieval_results),
            "confidence": confidence,
        })
        return QAResponse(
            answer=answer,
            citations=citations,
            confidence=confidence,
            retrieval_path=[r.get(RR_METADATA, {}).get(META_SOURCE, "") for r in retrieval_results],
        )
    except Exception:
        audit_logger.exception("QA 请求处理失败: request_id=%s", request_id)
        raise HTTPException(status_code=500, detail="内部处理错误")


# ══════════════════════════════════════════════════════════════════════
# Agent 接口（impl-03 §7）
# ══════════════════════════════════════════════════════════════════════

agent_app = None  # 懒加载，首次请求时构建


def _get_agent_app():
    """懒加载 Agent Graph（避免启动时 import 链触发 ChromaDB 连接）"""
    global agent_app
    if agent_app is None:
        from src.agents.graph import build_agent_with_checkpoint

        agent_app = build_agent_with_checkpoint()
    return agent_app


def _is_provider_unavailable(exc: Exception) -> bool:
    if isinstance(exc, (httpx.TimeoutException, httpx.TransportError)):
        return True

    if isinstance(exc, (APIConnectionError, APITimeoutError)):
        return True

    if isinstance(exc, APIStatusError):
        return exc.status_code in {401, 403, 408, 409, 429} or exc.status_code >= 500

    return False


@app.post("/v1/assistant/qa", response_model=AssistantQAResponse)
async def assistant_qa(
    request: AssistantQARequest,
    user: AuthenticatedUser = Depends(authenticate_user),
):
    thread_id = request.thread_id or str(uuid.uuid4())
    initial_state = build_assistant_initial_state(request, user)

    app = _get_agent_app()
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}, "recursion_limit": 20}
    try:
        result = app.invoke(initial_state, config=config)
    except Exception as exc:
        if _is_provider_unavailable(exc):
            audit_logger.warning(
                "Assistant provider unavailable: thread_id=%s error=%s",
                thread_id,
                exc.__class__.__name__,
            )
            raise HTTPException(
                status_code=503,
                detail=(
                    "LLM provider unavailable. Check OPENAI_API_BASE, OPENAI_API_KEY, "
                    "network/proxy settings, or start a local Ollama service and set "
                    "LLM_PROVIDER=ollama."
                ),
            ) from exc

        audit_logger.exception("Assistant QA failed: thread_id=%s", thread_id)
        raise HTTPException(status_code=500, detail="内部处理错误") from exc

    return AssistantQAResponse(
        answer=result[STATE_FINAL_ANSWER],
        citations=result[STATE_CITATIONS],
        confidence=result[STATE_CONFIDENCE],
        compliance=result[STATE_COMPLIANCE],
        audit_trail=result[STATE_AUDIT_TRAIL],
    )
