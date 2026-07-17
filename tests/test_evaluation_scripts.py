from __future__ import annotations

import json
from pathlib import Path

from scripts.evaluate_answers import admission_passed as answers_admission_passed
from scripts.evaluate_answers import evaluate_answers
from scripts.evaluate_compliance import admission_passed as compliance_admission_passed
from scripts.evaluate_compliance import evaluate_compliance
from scripts.evaluate_conversations import admission_passed as conversations_admission_passed
from scripts.evaluate_conversations import evaluate_conversations
from scripts.evaluation_common import write_artifact


def _write(tmp_path: Path, name: str, payload: list[dict]) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def test_answer_evaluation_passes_grounded_sample(tmp_path):
    dataset = _write(
        tmp_path,
        "answers.json",
        [
            {
                "answer": "净利润 747 亿元",
                "retrieval_results": [
                    {
                        "content": "净利润 747 亿元",
                        "metadata": {"source": "report.pdf", "chunk_id": "chunk-1"},
                        "score": 0.9,
                    }
                ],
                "citations": [{"source": "report.pdf", "chunk_id": "chunk-1"}],
                "tool_calls": [],
                "expected_passed": True,
            }
        ],
    )

    summary = evaluate_answers(dataset)

    assert answers_admission_passed(summary)


def test_compliance_evaluation_detects_expected_block(tmp_path):
    dataset = _write(
        tmp_path,
        "compliance.json",
        [
            {
                "answer": "这是未公开信息",
                "expected_blocked": True,
                "restricted_text": "未公开信息",
                "returned_answer": "当前请求或生成内容未通过合规检查，已停止输出。",
            }
        ],
    )

    summary = evaluate_compliance(dataset)

    assert compliance_admission_passed(summary)


def test_conversation_evaluation_requires_every_safety_check(tmp_path):
    dataset = _write(
        tmp_path,
        "conversations.json",
        [
            {
                "owner_isolated": True,
                "deleted_thread_rejected": True,
                "request_id_idempotent": True,
                "audit_complete": True,
                "current_turn_citations_only": True,
            }
        ],
    )

    summary = evaluate_conversations(dataset)

    assert conversations_admission_passed(summary)


def test_evaluation_artifact_uses_repository_relative_dataset_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    dataset = _write(tmp_path, "dataset.json", [])

    artifact = write_artifact(
        name="retrieval",
        dataset_path=dataset,
        summary={"recall@5": 1.0},
        output_root=tmp_path / "artifacts",
    )

    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert payload["dataset"] == "dataset.json"
