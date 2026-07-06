import logging
import uuid
from fastapi import Depends, FastAPI, HTTPException
from langchain_core.runnables.config import RunnableConfig
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response

from src.api.auth import AuthenticatedUser, authenticate_user, build_assistant_initial_state
from src.config import config
from src.rag.chain import build_rag_chain, format_docs
from src.rag.formatter import estimate_confidence, format_citations
from src.retrieval.vector_retriever import ChromaVectorRetriever
from src.schemas.constants import (
    META_SOURCE,
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
            <option value="demo-tech">技术支持</option>
            <option value="demo-advisor">投顾</option>
            <option value="demo-sales">机构销售</option>
            <option value="demo-compliance">规则</option>
            <option value="demo-ops">运营</option>
          </select>
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
    <pre id="result">等待查询...</pre>
  </main>
  <script>
    const submit = document.getElementById("submit");
    const copyResult = document.getElementById("copyResult");
    const statusEl = document.getElementById("status");
    const copyStatusEl = document.getElementById("copyStatus");
    const resultEl = document.getElementById("result");

    submit.addEventListener("click", async () => {
      const query = document.getElementById("query").value.trim();
      const token = document.getElementById("token").value;
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
      resultEl.textContent = "";

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
        resultEl.textContent = JSON.stringify(data, null, 2);
        statusEl.textContent = response.ok ? "完成" : `请求失败：${response.status}`;
      } catch (error) {
        resultEl.textContent = String(error);
        statusEl.textContent = "请求失败";
      } finally {
        submit.disabled = false;
      }
    });

    copyResult.addEventListener("click", async () => {
      const text = resultEl.textContent.trim();
      if (!text || text === "等待查询...") {
        copyStatusEl.textContent = "暂无可复制内容";
        return;
      }

      try {
        await navigator.clipboard.writeText(text);
        copyStatusEl.textContent = "已复制";
      } catch {
        const range = document.createRange();
        range.selectNodeContents(resultEl);
        const selection = window.getSelection();
        selection.removeAllRanges();
        selection.addRange(range);
        copyStatusEl.textContent = "已选中，请手动复制";
      }
    });
  </script>
</body>
</html>"""


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
        )

    return AssistantQAResponse(
        answer=result[STATE_FINAL_ANSWER],
        citations=result[STATE_CITATIONS],
        confidence=result[STATE_CONFIDENCE],
        compliance=result[STATE_COMPLIANCE],
        audit_trail=result[STATE_AUDIT_TRAIL],
    )
