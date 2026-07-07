import logging
import uuid

import httpx
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from langchain_core.runnables.config import RunnableConfig
from openai import APIConnectionError, APIStatusError, APITimeoutError

from src.api.auth import (
    AuthenticatedUser,
    authenticate_user,
    build_assistant_initial_state,
)
from src.api.ui import render_ui_html
from src.config import config
from src.rag.chain import build_rag_chain, format_docs
from src.rag.formatter import estimate_confidence, format_citations
from src.retrieval.vector_retriever import ChromaVectorRetriever
from src.schemas.constants import (
    API_ROUTE_ASSISTANT_QA,
    API_ROUTE_QA,
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
    return render_ui_html()


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


@app.post(API_ROUTE_QA, response_model=QAResponse)
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


@app.post(API_ROUTE_ASSISTANT_QA, response_model=AssistantQAResponse)
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
