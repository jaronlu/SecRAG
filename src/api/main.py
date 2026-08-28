import asyncio
import json
import logging
import uuid
from typing import cast

import httpx
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response, StreamingResponse
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
    API_ROUTE_ASSISTANT_QA_STREAM,
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
from src.utils.rate_limit import check_rate_limit, get_rate_limit_key

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


@app.get("/health")
async def health_check():
    """P2-4: 健康检查端点——检查 ChromaDB 连通性和文档计数。

    返回 200 表示服务存活；chroma 字段为 "error" 时不影响整体 200，
    避免 ChromaDB 短暂不可用导致负载均衡器摘除节点。
    """
    import time

    status = {"status": "ok", "timestamp": time.time()}
    try:
        from src.retrieval.vector_retriever import ChromaVectorRetriever

        engine = ChromaVectorRetriever()
        count = engine.collection.count()
        status["chroma"] = {"status": "ok", "doc_count": count}
    except Exception as exc:
        status["chroma"] = {"status": "error", "error": str(exc)[:200]}
    return status


@app.get("/v1/admin/stats/queries")
async def query_stats(
    days: int = 7,
    user: AuthenticatedUser = Depends(authenticate_user),
):
    """P2-9: 查询统计端点——返回查询量、命中率、无引用查询示例。

    仅 admin 角色可访问。
    """
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="仅管理员可访问统计数据")
    return _get_conversation_store().query_stats(days=days)


@app.delete("/v1/admin/documents")
async def delete_document(
    source: str,
    user: AuthenticatedUser = Depends(authenticate_user),
):
    """P2-7: 文档删除端点——按 source 路径删除该文档的所有 chunk。

    仅 admin 角色可访问。用于文档版本管理：删除旧版本后重新入库。
    """
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="仅管理员可删除文档")
    from src.retrieval.vector_retriever import ChromaVectorRetriever

    engine = ChromaVectorRetriever()
    deleted = engine.delete_by_source(source)
    return {"source": source, "deleted_chunks": deleted}


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
        status_code = getattr(exc, "status_code", None)
        return status_code in {401, 403, 408, 409, 429} or (
            status_code is not None and status_code >= 500
        )

    return False


@app.post(API_ROUTE_ASSISTANT_QA, response_model=AssistantQAResponse)
async def assistant_qa(
    request: AssistantQARequest,
    user: AuthenticatedUser = Depends(authenticate_user),
):
    # P2-2: 限流——按 user_id 滑动窗口，每分钟 30 次
    rate_key = get_rate_limit_key(user_id=user.user_id)
    allowed, _ = check_rate_limit(rate_key)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="请求过于频繁，请稍后再试（每分钟最多 30 次）。",
            headers={"Retry-After": "60"},
        )
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

    agent = _get_agent_app()
    runnable_config: RunnableConfig = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": AGENT_RECURSION_LIMIT,
    }
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(agent.invoke, initial_state, runnable_config),
            timeout=config.api_request_timeout_seconds,
        )
    except asyncio.TimeoutError:
        audit_logger.warning(
            "Assistant QA timed out after %.1fs: thread_id=%s",
            config.api_request_timeout_seconds,
            thread_id,
        )
        raise HTTPException(
            status_code=504,
            detail=f"请求处理超时（{config.api_request_timeout_seconds:.0f}s），请简化问题或稍后重试。",
        )
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


@app.post(API_ROUTE_ASSISTANT_QA_STREAM)
async def assistant_qa_stream(
    request: AssistantQARequest,
    user: AuthenticatedUser = Depends(authenticate_user),
):
    """P2-1: 流式输出端点——SSE 逐事件返回 Agent 执行进度和最终回答。

    事件格式：
    - event: progress, data: {"node": "...", "status": "done"}
    - event: answer, data: {"answer": "...", "citations": [...], "confidence": "..."}
    - event: error, data: {"detail": "..."}
    - event: done
    """
    # P2-2: 限流
    rate_key = get_rate_limit_key(user_id=user.user_id)
    allowed, _ = check_rate_limit(rate_key)
    if not allowed:
        return StreamingResponse(
            iter([f"event: error\ndata: {json.dumps({'detail': '请求过于频繁'})}\n\n"]),
            media_type="text/event-stream",
            status_code=429,
        )

    async def event_generator():
        try:
            thread = _get_conversation_store().ensure_thread_for_qa(
                thread_id=request.thread_id,
                user_id=user.user_id,
                user_role=user.role,
                client_id=request.client_id,
                title=request.query[:100],
            )
        except Exception as exc:
            yield f"event: error\ndata: {json.dumps({'detail': str(exc)})}\n\n"
            return

        thread_id = thread.get("thread_id", request.thread_id)
        turn_id = str(uuid.uuid4())
        initial_state = build_assistant_initial_state(
            request,
            user,
            thread_id=thread_id,
            turn_id=turn_id,
            turn_index=thread.get("turn_count", 0),
        )

        agent = _get_agent_app()
        runnable_config: RunnableConfig = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": AGENT_RECURSION_LIMIT,
        }

        try:
            # 流式获取每个节点的状态更新
            async for state_update in agent.astream(
                initial_state, runnable_config, stream_mode="updates"
            ):
                for node_name, node_output in state_update.items():
                    # 只发送关键节点的进度，避免事件过多
                    if node_name in ("query_understand", "planner", "retrieve", "grade_and_filter", "reason", "verify", "compose"):
                        yield f"event: progress\ndata: {json.dumps({'node': node_name, 'status': 'done'})}\n\n"
                    # compose 节点输出包含最终回答
                    if node_name == "compose" and STATE_FINAL_ANSWER in node_output:
                        answer_data = json.dumps({
                            "answer": node_output[STATE_FINAL_ANSWER],
                            "citations": node_output.get(STATE_CITATIONS, []),
                            "confidence": node_output.get(STATE_CONFIDENCE, "unknown"),
                            "thread_id": thread_id,
                            "turn_id": turn_id,
                        }, ensure_ascii=False)
                        yield f"event: answer\ndata: {answer_data}\n\n"
        except asyncio.TimeoutError:
            yield f"event: error\ndata: {json.dumps({'detail': '请求处理超时'})}\n\n"
        except Exception as exc:
            yield f"event: error\ndata: {json.dumps({'detail': str(exc)[:200]})}\n\n"

        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
