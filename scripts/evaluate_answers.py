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
    grounded_total = 0
    # 数字精确率/引用准确率/幻觉率的准入门槛（§2：100% / ≥95% / ≤5%）只对
    # "应当有据可答"的样本有意义；负例（不可回答、工具失败）的正确性由
    # expected_outcome_accuracy 度量，混进分母会让门槛永远无法达标
    for item in dataset:
        expected_passed = bool(item.get("expected_passed", True))
        result = verifier.verify(
            answer=str(item.get("answer", "")),
            citations=item.get("citations", []),
            retrieval_results=item.get("retrieval_results", []),
            tool_calls=item.get("tool_calls", []),
        )
        checks = result["checks"]
        expected_matches += int(result["passed"] == expected_passed)
        if not expected_passed:
            continue
        grounded_total += 1
        numeric_passed += int(checks["number_verification"]["passed"])
        citation_passed += int(checks["source_verification"]["passed"])
        hallucination_scores.append(checks["hallucination_detection"]["hallucination_score"])
    total = len(dataset)
    return {
        "samples": float(total),
        "grounded_samples": float(grounded_total),
        "numeric_accuracy": numeric_passed / grounded_total if grounded_total else 0.0,
        "citation_accuracy": citation_passed / grounded_total if grounded_total else 0.0,
        "hallucination_rate": (
            sum(hallucination_scores) / grounded_total if grounded_total else 1.0
        ),
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
