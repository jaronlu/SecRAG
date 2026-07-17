import logging
import uuid
from typing import cast

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
from src.api.ingestion import router as ingestion_router
from src.api.ui import render_ui_html
from src.config import config
from src.schemas.constants import (
    AGENT_RECURSION_LIMIT,
    API_ROUTE_ASSISTANT_QA,
    API_ROUTE_ASSISTANT_THREAD,
    API_ROUTE_ASSISTANT_THREAD_MESSAGES,
    API_ROUTE_ASSISTANT_THREADS,
    STATE_CITATIONS,
    STATE_COMPLIANCE,
    STATE_CONFIDENCE,
    STATE_FINAL_ANSWER,
    STATE_THREAD_ID,
    STATE_TURN_ID,
)
from src.schemas.request_response import (
    AssistantQARequest,
    AssistantQAResponse,
    ConversationMessageResponse,
    ConversationMessagesResponse,
    ConversationThreadCreate,
    ConversationThreadResponse,
)

app = FastAPI(title="机构内部投研知识平台", version="0.1.0")
app.include_router(ingestion_router)

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


# 追踪日志记录器（结构化 JSON，可对接 ELK / Loki）
audit_logger = logging.getLogger("secrag.audit")


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


def _get_conversation_store():
    from src.utils.conversation import SQLiteConversationStore

    return SQLiteConversationStore(config.conversation_db_path)


def _conversation_http_error(exc: Exception) -> HTTPException:
    from src.utils.conversation import (
        ConversationContextMismatchError,
        ConversationNotFoundError,
    )

    if isinstance(exc, ConversationNotFoundError):
        return HTTPException(status_code=404, detail="会话不存在或不可访问")
    if isinstance(exc, ConversationContextMismatchError):
        return HTTPException(status_code=409, detail=str(exc))
    raise exc


@app.post(API_ROUTE_ASSISTANT_THREADS, response_model=ConversationThreadResponse)
async def create_assistant_thread(
    request: ConversationThreadCreate,
    user: AuthenticatedUser = Depends(authenticate_user),
):
    thread = _get_conversation_store().create_thread(
        user_id=user.user_id,
        user_role=user.role,
        client_id=request.client_id,
        title=request.title,
    )
    # create_thread 返回全量字段，收窄类型以消除 TypedDict(total=False) 的访问警告
    assert "thread_id" in thread and "title" in thread and "created_at" in thread
    return ConversationThreadResponse(
        thread_id=thread["thread_id"],
        title=thread["title"],
        created_at=thread["created_at"],
    )


@app.get(API_ROUTE_ASSISTANT_THREAD_MESSAGES, response_model=ConversationMessagesResponse)
async def get_assistant_thread_messages(
    thread_id: str,
    user: AuthenticatedUser = Depends(authenticate_user),
):
    try:
        messages = _get_conversation_store().list_messages(
            thread_id=thread_id,
            user_id=user.user_id,
        )
    except Exception as exc:
        raise _conversation_http_error(exc) from exc
    return ConversationMessagesResponse(
        thread_id=thread_id,
        messages=cast(list[ConversationMessageResponse], messages),
    )


@app.delete(API_ROUTE_ASSISTANT_THREAD, status_code=204)
async def delete_assistant_thread(
    thread_id: str,
    user: AuthenticatedUser = Depends(authenticate_user),
):
    try:
        _get_conversation_store().soft_delete_thread(
            thread_id=thread_id,
            user_id=user.user_id,
        )
    except Exception as exc:
        raise _conversation_http_error(exc) from exc
    return Response(status_code=204)


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
    try:
        thread = _get_conversation_store().ensure_thread_for_qa(
            thread_id=request.thread_id,
            user_id=user.user_id,
            user_role=user.role,
            client_id=request.client_id,
            title=request.query[:100],
        )
    except Exception as exc:
        raise _conversation_http_error(exc) from exc
    # ensure_thread_for_qa 返回的 thread 保证含 thread_id（查找键或新建时赋值）
    thread_id = thread.get("thread_id", request.thread_id)
    turn_id = str(uuid.uuid4())
    initial_state = build_assistant_initial_state(
        request,
        user,
        thread_id=thread_id,
        turn_id=turn_id,
        turn_index=thread.get("turn_count", 0),
    )

    app = _get_agent_app()
    config: RunnableConfig = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": AGENT_RECURSION_LIMIT,
    }
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
        thread_id=result.get(STATE_THREAD_ID, thread_id),
        turn_id=result.get(STATE_TURN_ID, turn_id),
        answer=result[STATE_FINAL_ANSWER],
        citations=result[STATE_CITATIONS],
        confidence=result[STATE_CONFIDENCE],
        compliance=result[STATE_COMPLIANCE],
    )
