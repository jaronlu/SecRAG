"""Evaluate answer grounding, citations, numbers, and hallucination admission metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.evaluation_common import load_dataset, write_artifact
from src.utils.verifier import ComprehensiveVerifier


def evaluate_answers(dataset_path: str | Path) -> dict[str, float]:
    dataset = load_dataset(dataset_path)
    verifier = ComprehensiveVerifier()
    numeric_passed = 0
    citation_passed = 0
    hallucination_scores = []
    expected_matches = 0
    for item in dataset:
        result = verifier.verify(
            answer=str(item.get("answer", "")),
            citations=item.get("citations", []),
            retrieval_results=item.get("retrieval_results", []),
            tool_calls=item.get("tool_calls", []),
        )
        checks = result["checks"]
        numeric_passed += int(checks["number_verification"]["passed"])
        citation_passed += int(checks["source_verification"]["passed"])
        hallucination_scores.append(checks["hallucination_detection"]["hallucination_score"])
        expected_matches += int(result["passed"] == bool(item.get("expected_passed", True)))
    total = len(dataset)
    return {
        "samples": float(total),
        "numeric_accuracy": numeric_passed / total if total else 0.0,
        "citation_accuracy": citation_passed / total if total else 0.0,
        "hallucination_rate": sum(hallucination_scores) / total if total else 1.0,
        "expected_outcome_accuracy": expected_matches / total if total else 0.0,
    }


def admission_passed(summary: dict[str, float]) -> bool:
    return (
        summary["samples"] > 0
        and summary["numeric_accuracy"] == 1.0
        and summary["citation_accuracy"] >= 0.95
        and summary["hallucination_rate"] <= 0.05
        and summary["expected_outcome_accuracy"] == 1.0
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="评估答案引用、数字与幻觉指标")
    parser.add_argument("dataset_path")
    parser.add_argument("--output-root", default="artifacts/evaluation")
    args = parser.parse_args()
    summary = evaluate_answers(args.dataset_path)
    artifact = write_artifact(
        name="answers",
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
