"""SQLite conversation persistence with a transactional audit outbox."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

from src.agents.state import AssistantState
from src.schemas.constants import (
    AUDIT_REQUEST_ID,
    STATE_AUDIT_TRAIL,
    STATE_CITATIONS,
    STATE_CLIENT_ID,
    STATE_ENTITIES,
    STATE_FINAL_ANSWER,
    STATE_ORIGINAL_QUERY,
    STATE_RESOLVED_QUERY,
    STATE_REWRITTEN_QUERY,
    STATE_THREAD_ID,
    STATE_TURN_ID,
    STATE_USER_ID,
    STATE_USER_ROLE,
)
from src.schemas.typed_dicts import (
    ConversationMessageDict,
    ConversationThreadDict,
    ConversationTurnDict,
)


class ConversationNotFoundError(LookupError):
    """Raised for missing, deleted, or inaccessible threads."""


class ConversationContextMismatchError(ValueError):
    """Raised when role or client context changes within a thread."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SQLiteConversationStore:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)

    def create_thread(
        self,
        *,
        user_id: str,
        user_role: str,
        client_id: str | None,
        title: str = "新会话",
    ) -> ConversationThreadDict:
        thread_id = str(uuid.uuid4())
        now = utc_now()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(self.db_path), timeout=5) as conn:
            self._ensure_schema(conn)
            conn.execute(
                """
                INSERT INTO conversation_threads (
                    thread_id, user_id, user_role, client_id, title, status,
                    turn_count, created_at, updated_at, deleted_at
                ) VALUES (?, ?, ?, ?, ?, 'active', 0, ?, ?, NULL)
                """,
                (thread_id, user_id, user_role, client_id, title.strip(), now, now),
            )
        return ConversationThreadDict(
            thread_id=thread_id,
            user_id=user_id,
            user_role=user_role,
            client_id=client_id,
            title=title.strip(),
            status="active",
            turn_count=0,
            created_at=now,
            updated_at=now,
            deleted_at=None,
        )

    def ensure_thread_for_qa(
        self,
        *,
        thread_id: str | None,
        user_id: str,
        user_role: str,
        client_id: str | None,
        title: str,
    ) -> ConversationThreadDict:
        if not thread_id:
            return self.create_thread(
                user_id=user_id,
                user_role=user_role,
                client_id=client_id,
                title=title[:100] or "新会话",
            )
        thread = self.get_thread_for_user(thread_id=thread_id, user_id=user_id)
        if thread.get("user_role") != user_role or thread.get("client_id") != client_id:
            raise ConversationContextMismatchError("会话角色或客户上下文发生变化")
        return thread

    def get_thread_for_user(self, *, thread_id: str, user_id: str) -> ConversationThreadDict:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(self.db_path), timeout=5) as conn:
            conn.row_factory = sqlite3.Row
            self._ensure_schema(conn)
            row = conn.execute(
                """
                SELECT * FROM conversation_threads
                WHERE thread_id = ? AND user_id = ? AND status = 'active'
                """,
                (thread_id, user_id),
            ).fetchone()
        if row is None:
            raise ConversationNotFoundError("会话不存在或不可访问")
        return cast(ConversationThreadDict, dict(row))

    def soft_delete_thread(self, *, thread_id: str, user_id: str) -> None:
        deleted_at = utc_now()
        with sqlite3.connect(str(self.db_path), timeout=5) as conn:
            self._ensure_schema(conn)
            row = conn.execute(
                """
                SELECT thread_id FROM conversation_threads
                WHERE thread_id = ? AND user_id = ? AND status = 'active'
                """,
                (thread_id, user_id),
            ).fetchone()
            if row is None:
                raise ConversationNotFoundError("会话不存在或不可访问")
            conn.execute(
                """
                UPDATE conversation_threads
                SET status = 'deleted', deleted_at = ?, updated_at = ?
                WHERE thread_id = ?
                """,
                (deleted_at, deleted_at, thread_id),
            )
            conn.execute(
                """
                UPDATE conversation_messages SET deleted_at = ?
                WHERE thread_id = ? AND deleted_at IS NULL
                """,
                (deleted_at, thread_id),
            )

    def list_messages(
        self, *, thread_id: str, user_id: str, limit: int = 100
    ) -> list[ConversationMessageDict]:
        self.get_thread_for_user(thread_id=thread_id, user_id=user_id)
        with sqlite3.connect(str(self.db_path), timeout=5) as conn:
            conn.row_factory = sqlite3.Row
            self._ensure_schema(conn)
            rows = conn.execute(
                """
                SELECT message_id, thread_id, turn_id, role, content, sequence,
                       created_at, request_id, deleted_at
                FROM conversation_messages
                WHERE thread_id = ? AND deleted_at IS NULL
                ORDER BY created_at DESC, sequence DESC
                LIMIT ?
                """,
                (thread_id, limit),
            ).fetchall()
        return [cast(ConversationMessageDict, dict(row)) for row in rows]

    def load_context(
        self, *, thread_id: str, user_id: str, limit: int = 12
    ) -> tuple[list[ConversationMessageDict], str]:
        messages = self.list_messages(thread_id=thread_id, user_id=user_id, limit=limit)
        chronological = list(reversed(messages))
        turns = self.get_recent_turns(thread_id=thread_id, user_id=user_id, limit=5)
        entities = [turn.get("entities", {}) for turn in turns]
        parts = []
        for item in entities:
            for key, value in item.items():
                if value:
                    parts.append(f"{key}={value}")
        return chronological, "；".join(dict.fromkeys(parts))

    def get_recent_turns(
        self, *, thread_id: str, user_id: str, limit: int = 5
    ) -> list[ConversationTurnDict]:
        self.get_thread_for_user(thread_id=thread_id, user_id=user_id)
        with sqlite3.connect(str(self.db_path), timeout=5) as conn:
            conn.row_factory = sqlite3.Row
            self._ensure_schema(conn)
            rows = conn.execute(
                """
                SELECT turn_id, thread_id, user_query, resolved_query, answer_summary,
                       entities_json, citations_json, request_id, created_at
                FROM conversation_turns
                WHERE thread_id = ? ORDER BY created_at DESC LIMIT ?
                """,
                (thread_id, limit),
            ).fetchall()
        return [
            ConversationTurnDict(
                turn_id=row["turn_id"],
                thread_id=row["thread_id"],
                user_query=row["user_query"],
                resolved_query=row["resolved_query"],
                answer_summary=row["answer_summary"],
                entities=json.loads(row["entities_json"]),
                citations=json.loads(row["citations_json"]),
                request_id=row["request_id"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def insert_turn(self, state: AssistantState) -> None:
        audit_trail = state.get(STATE_AUDIT_TRAIL, {})
        request_id = str(audit_trail.get(AUDIT_REQUEST_ID, ""))
        if not request_id:
            raise ValueError("persist_conversation_turn 缺少 request_id")
        thread_id = state[STATE_THREAD_ID]
        turn_id = state[STATE_TURN_ID]
        created_at = utc_now()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(str(self.db_path), timeout=5) as conn:
            conn.row_factory = sqlite3.Row
            self._ensure_schema(conn)
            thread = conn.execute(
                """
                SELECT * FROM conversation_threads
                WHERE thread_id = ? AND user_id = ? AND status = 'active'
                """,
                (thread_id, state[STATE_USER_ID]),
            ).fetchone()
            if thread is None:
                raise ConversationNotFoundError("会话不存在或不可访问")
            if thread["user_role"] != state[STATE_USER_ROLE] or thread["client_id"] != state.get(
                STATE_CLIENT_ID
            ):
                raise ConversationContextMismatchError("会话角色或客户上下文发生变化")
            if conn.execute(
                "SELECT 1 FROM conversation_turns WHERE request_id = ?", (request_id,)
            ).fetchone():
                return

            next_sequence = conn.execute(
                "SELECT COALESCE(MAX(sequence), 0) + 1 FROM conversation_messages WHERE thread_id = ?",
                (thread_id,),
            ).fetchone()[0]
            user_query = state.get(STATE_ORIGINAL_QUERY, "")
            answer = state.get(STATE_FINAL_ANSWER, "")
            resolved_query = (
                state.get(STATE_RESOLVED_QUERY) or state.get(STATE_REWRITTEN_QUERY) or user_query
            )
            conn.executemany(
                """
                INSERT INTO conversation_messages (
                    message_id, thread_id, turn_id, role, content, sequence,
                    created_at, request_id, deleted_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL)
                """,
                (
                    (
                        str(uuid.uuid4()),
                        thread_id,
                        turn_id,
                        "user",
                        user_query,
                        next_sequence,
                        created_at,
                        request_id,
                    ),
                    (
                        str(uuid.uuid4()),
                        thread_id,
                        turn_id,
                        "assistant",
                        answer,
                        next_sequence + 1,
                        created_at,
                        request_id,
                    ),
                ),
            )
            conn.execute(
                """
                INSERT INTO conversation_turns (
                    turn_id, thread_id, user_query, resolved_query, answer_summary,
                    entities_json, citations_json, request_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    turn_id,
                    thread_id,
                    user_query,
                    resolved_query,
                    answer[:500],
                    json.dumps(state.get(STATE_ENTITIES, {}), ensure_ascii=False, sort_keys=True),
                    json.dumps(state.get(STATE_CITATIONS, []), ensure_ascii=False, sort_keys=True),
                    request_id,
                    created_at,
                ),
            )
            outbox_payload = {
                "request_id": request_id,
                "thread_id": thread_id,
                "turn_id": turn_id,
                "entities": state.get(STATE_ENTITIES, {}),
                "citations": state.get(STATE_CITATIONS, []),
            }
            conn.execute(
                """
                INSERT INTO audit_outbox (
                    request_id, event_type, payload_json, status, attempts,
                    last_error, created_at, processed_at
                ) VALUES (?, 'conversation_turn_persisted', ?, 'pending', 0, '', ?, NULL)
                """,
                (request_id, json.dumps(outbox_payload, ensure_ascii=False), created_at),
            )
            title = thread["title"]
            if thread["turn_count"] == 0 and title == "新会话":
                title = user_query[:100] or title
            conn.execute(
                """
                UPDATE conversation_threads
                SET title = ?, turn_count = turn_count + 1, updated_at = ?
                WHERE thread_id = ?
                """,
                (title, created_at, thread_id),
            )

    def mark_outbox_processed(self, request_id: str) -> None:
        with sqlite3.connect(str(self.db_path), timeout=5) as conn:
            self._ensure_schema(conn)
            cursor = conn.execute(
                """
                UPDATE audit_outbox
                SET status = 'processed', processed_at = ?, attempts = attempts + 1, last_error = ''
                WHERE request_id = ?
                """,
                (utc_now(), request_id),
            )
            if cursor.rowcount == 0:
                raise LookupError(f"audit outbox event not found: {request_id}")

    def mark_outbox_failed(self, request_id: str, error: str) -> None:
        with sqlite3.connect(str(self.db_path), timeout=5) as conn:
            self._ensure_schema(conn)
            conn.execute(
                """
                UPDATE audit_outbox
                SET status = 'pending', attempts = attempts + 1, last_error = ?
                WHERE request_id = ?
                """,
                (error[:500], request_id),
            )

    def get_outbox_status(self, request_id: str) -> str | None:
        with sqlite3.connect(str(self.db_path), timeout=5) as conn:
            self._ensure_schema(conn)
            row = conn.execute(
                "SELECT status FROM audit_outbox WHERE request_id = ?", (request_id,)
            ).fetchone()
        return str(row[0]) if row else None

    def _ensure_schema(self, conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS conversation_threads (
                thread_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                user_role TEXT NOT NULL,
                client_id TEXT,
                title TEXT NOT NULL,
                status TEXT NOT NULL,
                turn_count INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                deleted_at TEXT
            );
            CREATE TABLE IF NOT EXISTS conversation_messages (
                message_id TEXT PRIMARY KEY,
                thread_id TEXT NOT NULL,
                turn_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                sequence INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                request_id TEXT,
                deleted_at TEXT,
                UNIQUE(thread_id, sequence),
                UNIQUE(request_id, role)
            );
            CREATE TABLE IF NOT EXISTS conversation_turns (
                turn_id TEXT PRIMARY KEY,
                thread_id TEXT NOT NULL,
                user_query TEXT NOT NULL,
                resolved_query TEXT NOT NULL,
                answer_summary TEXT NOT NULL,
                entities_json TEXT NOT NULL,
                citations_json TEXT NOT NULL,
                request_id TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS audit_outbox (
                request_id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                status TEXT NOT NULL,
                attempts INTEGER NOT NULL,
                last_error TEXT NOT NULL,
                created_at TEXT NOT NULL,
                processed_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_threads_user_updated
                ON conversation_threads(user_id, updated_at DESC);
            CREATE INDEX IF NOT EXISTS idx_messages_thread_timeline
                ON conversation_messages(thread_id, created_at DESC, sequence DESC);
            CREATE INDEX IF NOT EXISTS idx_turns_thread_created
                ON conversation_turns(thread_id, created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_outbox_status_created
                ON audit_outbox(status, created_at);
            """
        )
