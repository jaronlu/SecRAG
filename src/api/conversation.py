"""会话管理端点（impl-07）：线程列表、创建、消息查看与软删除。"""

from __future__ import annotations

import asyncio
from typing import cast

from fastapi import APIRouter, Depends, Response

from src.api.auth import AuthenticatedUser, authenticate_user
from src.api.deps import _conversation_http_error, _get_conversation_store
from src.schemas.constants import (
    API_ROUTE_ASSISTANT_THREAD,
    API_ROUTE_ASSISTANT_THREAD_MESSAGES,
    API_ROUTE_ASSISTANT_THREADS,
)
from src.schemas.request_response import (
    ConversationMessageResponse,
    ConversationMessagesResponse,
    ConversationThreadCreate,
    ConversationThreadResponse,
)

router = APIRouter()


@router.get(API_ROUTE_ASSISTANT_THREADS, response_model=dict[str, list[ConversationThreadResponse]])
async def list_assistant_threads(
    limit: int = 50,
    user: AuthenticatedUser = Depends(authenticate_user),
):
    """列出当前用户的活跃会话（issues.md 一.3：前端会话列表契约）。"""
    # ISSUE-18：同步 SQLite 调用移入线程池，避免阻塞事件循环
    threads = await asyncio.to_thread(
        _get_conversation_store().list_threads,
        user_id=user.user_id,
        limit=limit,
    )
    return {
        "threads": [
            ConversationThreadResponse(
                thread_id=thread["thread_id"],
                title=thread["title"],
                created_at=thread["created_at"],
            )
            for thread in threads
        ]
    }


@router.post(API_ROUTE_ASSISTANT_THREADS, response_model=ConversationThreadResponse)
async def create_assistant_thread(
    request: ConversationThreadCreate,
    user: AuthenticatedUser = Depends(authenticate_user),
):
    thread = await asyncio.to_thread(
        _get_conversation_store().create_thread,
        user_id=user.user_id,
        user_role=user.role,
        client_id=request.client_id,
        title=request.title,
    )
    # create_thread 返回的 ConversationThreadDict 全键必填（SQLiteConversationStore 全量构造）
    return ConversationThreadResponse(
        thread_id=thread["thread_id"],
        title=thread["title"],
        created_at=thread["created_at"],
    )


@router.get(API_ROUTE_ASSISTANT_THREAD_MESSAGES, response_model=ConversationMessagesResponse)
async def get_assistant_thread_messages(
    thread_id: str,
    user: AuthenticatedUser = Depends(authenticate_user),
):
    try:
        messages = await asyncio.to_thread(
            _get_conversation_store().list_messages,
            thread_id=thread_id,
            user_id=user.user_id,
        )
    except Exception as exc:
        raise _conversation_http_error(exc) from exc
    return ConversationMessagesResponse(
        thread_id=thread_id,
        messages=cast(list[ConversationMessageResponse], messages),
    )


@router.delete(API_ROUTE_ASSISTANT_THREAD, status_code=204)
async def delete_assistant_thread(
    thread_id: str,
    user: AuthenticatedUser = Depends(authenticate_user),
):
    try:
        await asyncio.to_thread(
            _get_conversation_store().soft_delete_thread,
            thread_id=thread_id,
            user_id=user.user_id,
        )
    except Exception as exc:
        raise _conversation_http_error(exc) from exc
    return Response(status_code=204)
