from __future__ import annotations

import pytest

from src.agents.nodes import audit_log
from src.api.auth import AuthenticatedUser, build_assistant_initial_state
from src.schemas.constants import (
    AUDIT_REQUEST_ID,
    ROLE_TECHNICAL,
    STATE_AUDIT_TRAIL,
    STATE_CITATIONS,
    STATE_COMPLIANCE,
    STATE_CONFIDENCE,
    STATE_ENTITIES,
    STATE_FINAL_ANSWER,
    STATE_RESOLVED_QUERY,
    STATE_RISK_DISCLOSURE,
    STATE_VERIFICATION,
)
from src.schemas.request_response import AssistantQARequest
from src.utils.audit import SQLiteAuditStore
from src.utils.conversation import (
    ConversationContextMismatchError,
    ConversationNotFoundError,
    SQLiteConversationStore,
)


def _state(store: SQLiteConversationStore, *, request_id: str = "request-1"):
    user = AuthenticatedUser("user-tech", ROLE_TECHNICAL, "tech")
    thread = store.create_thread(
        user_id=user.user_id,
        user_role=user.role,
        client_id=None,
        title="新会话",
    )
    state = build_assistant_initial_state(
        AssistantQARequest(query="600519 的净利润是多少"),
        user,
        thread_id=thread["thread_id"],
        turn_id="turn-1",
        request_id=request_id,
    )
    state[STATE_FINAL_ANSWER] = "600519 的净利润为 747 亿元"
    state[STATE_RESOLVED_QUERY] = "600519 的净利润是多少"
    state[STATE_ENTITIES] = {"stock_code": "600519"}
    state[STATE_CITATIONS] = [{"source": "report.pdf", "chunk_id": "chunk-1"}]
    state[STATE_VERIFICATION] = {"passed": True, "confidence": "high", "issues": []}
    state[STATE_COMPLIANCE] = {"passed": True, "flags": [], "risk_disclosure": ""}
    state[STATE_CONFIDENCE] = "high"
    state[STATE_RISK_DISCLOSURE] = ""
    return state


def test_conversation_store_persists_turn_and_is_idempotent(tmp_path):
    store = SQLiteConversationStore(tmp_path / "conversations.db")
    state = _state(store)

    store.insert_turn(state)
    store.insert_turn(state)

    messages = store.list_messages(
        thread_id=state["thread_id"],
        user_id=state["user_id"],
    )
    assert [message["role"] for message in messages] == ["assistant", "user"]
    assert store.get_outbox_status("request-1") == "pending"


def test_conversation_store_enforces_owner_role_and_client_context(tmp_path):
    store = SQLiteConversationStore(tmp_path / "conversations.db")
    thread = store.create_thread(
        user_id="owner",
        user_role=ROLE_TECHNICAL,
        client_id="client-1",
    )

    with pytest.raises(ConversationNotFoundError):
        store.get_thread_for_user(thread_id=thread["thread_id"], user_id="other")
    with pytest.raises(ConversationContextMismatchError):
        store.ensure_thread_for_qa(
            thread_id=thread["thread_id"],
            user_id="owner",
            user_role=ROLE_TECHNICAL,
            client_id="client-2",
            title="query",
        )


def test_soft_delete_marks_messages_and_prevents_reuse(tmp_path):
    store = SQLiteConversationStore(tmp_path / "conversations.db")
    state = _state(store)
    store.insert_turn(state)

    store.soft_delete_thread(thread_id=state["thread_id"], user_id=state["user_id"])

    with pytest.raises(ConversationNotFoundError):
        store.get_thread_for_user(thread_id=state["thread_id"], user_id=state["user_id"])
    with pytest.raises(ConversationNotFoundError):
        store.list_messages(thread_id=state["thread_id"], user_id=state["user_id"])


def test_audit_log_consumes_outbox_idempotently(monkeypatch, tmp_path):
    conversation_store = SQLiteConversationStore(tmp_path / "conversations.db")
    audit_store = SQLiteAuditStore(tmp_path / "audit.db")
    state = _state(conversation_store, request_id="request-audit")
    conversation_store.insert_turn(state)
    monkeypatch.setattr("src.agents.nodes._get_conversation_store", lambda: conversation_store)
    monkeypatch.setattr("src.agents.nodes._get_audit_store", lambda: audit_store)

    result = audit_log(state)

    assert result[STATE_AUDIT_TRAIL][AUDIT_REQUEST_ID] == "request-audit"
    assert conversation_store.get_outbox_status("request-audit") == "processed"
    assert audit_store.get_by_request_id("request-audit") is not None
