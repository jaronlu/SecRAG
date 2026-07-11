"""Execute deterministic conversation isolation, deletion, idempotency, and audit cases."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from src.api.auth import AuthenticatedUser, build_assistant_initial_state
from src.schemas.constants import (
    ROLE_TECHNICAL,
    STATE_CITATIONS,
    STATE_COMPLIANCE,
    STATE_ENTITIES,
    STATE_FINAL_ANSWER,
    STATE_RESOLVED_QUERY,
    STATE_VERIFICATION,
)
from src.schemas.request_response import AssistantQARequest
from src.utils.audit import AuditLogger, SQLiteAuditStore
from src.utils.conversation import ConversationNotFoundError, SQLiteConversationStore

from scripts.evaluation_common import load_dataset, write_artifact


def _build_state(store: SQLiteConversationStore, item: dict, index: int):
    user = AuthenticatedUser(
        str(item.get("user_id", f"owner-{index}")),
        str(item.get("user_role", ROLE_TECHNICAL)),
        str(item.get("department", "tech")),
    )
    thread = store.create_thread(
        user_id=user.user_id,
        user_role=user.role,
        client_id=item.get("client_id"),
        title="新会话",
    )
    request_id = str(item.get("request_id", f"request-{index}"))
    state = build_assistant_initial_state(
        AssistantQARequest(
            query=str(item.get("query", "600519 的净利润是多少")),
            client_id=item.get("client_id"),
        ),
        user,
        thread_id=thread["thread_id"],
        turn_id=f"turn-{index}",
        request_id=request_id,
    )
    state[STATE_FINAL_ANSWER] = str(item.get("answer", "净利润为 747 亿元"))
    state[STATE_RESOLVED_QUERY] = state["original_query"]
    state[STATE_ENTITIES] = item.get("entities", {"stock_code": "600519"})
    state[STATE_CITATIONS] = item.get(
        "citations",
        [{"source": "report.pdf", "chunk_id": f"chunk-{index}"}],
    )
    state[STATE_VERIFICATION] = {"passed": True, "confidence": "high", "issues": []}
    state[STATE_COMPLIANCE] = {"passed": True, "flags": [], "risk_disclosure": ""}
    return user, state


def evaluate_conversations(dataset_path: str | Path) -> dict[str, float]:
    dataset = load_dataset(dataset_path)
    passed = 0
    with tempfile.TemporaryDirectory(prefix="secrag-conversation-eval-") as temp_dir:
        for index, item in enumerate(dataset):
            store = SQLiteConversationStore(Path(temp_dir) / f"conversation-{index}.db")
            audit_store = SQLiteAuditStore(Path(temp_dir) / f"audit-{index}.db")
            user, state = _build_state(store, item, index)
            thread_id = state["thread_id"]

            owner_isolated = False
            try:
                store.get_thread_for_user(thread_id=thread_id, user_id=f"other-{index}")
            except ConversationNotFoundError:
                owner_isolated = True

            store.insert_turn(state)
            store.insert_turn(state)
            messages = store.list_messages(thread_id=thread_id, user_id=user.user_id)
            request_id_idempotent = len(messages) == 2

            turns = store.get_recent_turns(thread_id=thread_id, user_id=user.user_id)
            current_turn_citations_only = (
                len(turns) == 1 and turns[0]["citations"] == state[STATE_CITATIONS]
            )

            audit_entry = AuditLogger().log(state)
            audit_store.insert(audit_entry)
            store.mark_outbox_processed(audit_entry.request_id)
            audit_complete = (
                store.get_outbox_status(audit_entry.request_id) == "processed"
                and audit_store.get_by_request_id(audit_entry.request_id) is not None
            )

            store.soft_delete_thread(thread_id=thread_id, user_id=user.user_id)
            deleted_thread_rejected = False
            try:
                store.get_thread_for_user(thread_id=thread_id, user_id=user.user_id)
            except ConversationNotFoundError:
                deleted_thread_rejected = True

            passed += int(
                owner_isolated
                and request_id_idempotent
                and current_turn_citations_only
                and audit_complete
                and deleted_thread_rejected
            )
    total = len(dataset)
    return {
        "samples": float(total),
        "conversation_case_accuracy": passed / total if total else 0.0,
    }


def admission_passed(summary: dict[str, float]) -> bool:
    return summary["samples"] > 0 and summary["conversation_case_accuracy"] == 1.0


def main() -> None:
    parser = argparse.ArgumentParser(description="评估会话隔离、删除、幂等与审计结果")
    parser.add_argument("dataset_path")
    parser.add_argument("--output-root", default="artifacts/evaluation")
    args = parser.parse_args()
    summary = evaluate_conversations(args.dataset_path)
    artifact = write_artifact(
        name="conversations",
        dataset_path=args.dataset_path,
        summary=summary,
        output_root=args.output_root,
    )
    print(summary)
    print(f"评估产物: {artifact}")
    if not admission_passed(summary):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
