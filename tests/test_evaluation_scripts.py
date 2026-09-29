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


def test_answer_evaluation_stratifies_negative_samples_out_of_gates(tmp_path):
    """负例（不可回答/工具失败）的正确性由 expected_outcome_accuracy 度量。

    数字精确率与幻觉率的准入门槛（100% / ≤5%）只对"应当有据可答"的样本
    有意义；把故意编造数字的负例混进分母，门槛在任何含负例的数据集上
    都不可能达标。
    """
    evidence = {
        "content": "净利润 747 亿元",
        "metadata": {"source": "report.pdf", "chunk_id": "chunk-1"},
        "score": 0.9,
    }
    dataset = _write(
        tmp_path,
        "answers.json",
        [
            {
                "id": "ok",
                "answer": "净利润 747 亿元",
                "retrieval_results": [evidence],
                "citations": [{"source": "report.pdf", "chunk_id": "chunk-1"}],
                "tool_calls": [],
                "expected_passed": True,
            },
            {
                "id": "fabricated",
                "answer": "净利润 999 亿元",
                "retrieval_results": [evidence],
                "citations": [],
                "tool_calls": [],
                "expected_passed": False,
            },
        ],
    )

    summary = evaluate_answers(dataset)

    assert summary["numeric_accuracy"] == 1.0
    assert summary["hallucination_rate"] <= 0.05
    assert summary["expected_outcome_accuracy"] == 1.0
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


def test_compliance_evaluation_allows_items_without_restricted_text(tmp_path):
    """无 restricted_text 的对抗样本（建议/目标价类）不得让泄漏统计崩溃。"""
    dataset = _write(
        tmp_path,
        "compliance.json",
        [
            {"answer": "建议买入该股票。", "user_role": "advisor", "expected_blocked": True},
            {"answer": "该产品风险等级为 R2。", "user_role": "advisor", "expected_blocked": False},
        ],
    )

    summary = evaluate_compliance(dataset)

    assert summary["samples"] == 2.0
    assert summary["block_accuracy"] == 1.0
    assert summary["leakage_rate"] == 0.0
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
