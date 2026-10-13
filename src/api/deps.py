"""API 层共享的懒加载单例与错误映射（供各 router 模块复用）。"""

from __future__ import annotations

import threading

from fastapi import HTTPException

from src.config import config

agent_app = None  # 懒加载，首次请求时构建
_agent_app_lock = threading.Lock()


def _get_agent_app():
    """懒加载 Agent Graph（避免启动时 import 链触发 ChromaDB 连接）。

    双检锁：lifespan 预热线程与首个请求可能并发进入。
    """
    global agent_app
    if agent_app is None:
        with _agent_app_lock:
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
