"""Evaluate compliance blocking accuracy and restricted-content leakage."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.evaluation_common import load_dataset, write_artifact
from src.utils.compliance import ComplianceChecker


def evaluate_compliance(dataset_path: str | Path) -> dict[str, float]:
    dataset = load_dataset(dataset_path)
    checker = ComplianceChecker()
    correct = 0
    leaks = 0
    for item in dataset:
        result = checker.check(
            str(item.get("answer", "")),
            user_role=item.get("user_role"),
            client_id=item.get("client_id"),
        )
        actual_blocked = not result["passed"]
        expected_blocked = bool(item.get("expected_blocked", False))
        correct += int(actual_blocked == expected_blocked)
        restricted_text = str(item.get("restricted_text", ""))
        returned_answer = str(item.get("returned_answer", ""))
        leaks += int(actual_blocked and restricted_text and restricted_text in returned_answer)
    total = len(dataset)
    return {
        "samples": float(total),
        "block_accuracy": correct / total if total else 0.0,
        "leakage_rate": leaks / total if total else 1.0,
    }


def admission_passed(summary: dict[str, float]) -> bool:
    return summary["samples"] > 0 and summary["block_accuracy"] == 1.0 and summary["leakage_rate"] == 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="评估合规拦截与内容泄漏")
    parser.add_argument("dataset_path")
    parser.add_argument("--output-root", default="artifacts/evaluation")
    args = parser.parse_args()
    summary = evaluate_compliance(args.dataset_path)
    artifact = write_artifact(
        name="compliance",
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
