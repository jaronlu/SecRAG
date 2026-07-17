"""检索评估脚本：按评估集运行 HybridRetriever 并汇总核心指标。"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from src.retrieval.hybrid_retriever import HybridRetriever
from src.schemas.constants import (
    META_CHUNK_ID,
    PLAN_DENIED,
    PLAN_FILTERS,
    PLAN_QUERY,
    PLAN_REASON,
    PLAN_SOURCE,
    PLAN_TOP_K,
    QT_FAQ_INQUIRY,
    QT_PRODUCT_INQUIRY,
    QT_REGULATION_INQUIRY,
    QT_REPORT_INQUIRY,
    QT_RULE_INQUIRY,
    QT_TECHNICAL_INQUIRY,
    ROLE_ADVISOR,
    ROLE_DATA_PERMISSIONS,
    RR_DENIED,
    RR_METADATA,
    SOURCE_FAQ,
    SOURCE_PRODUCT,
    SOURCE_REGULATION,
    SOURCE_REPORT,
)
from src.schemas.typed_dicts import RetrievalPlanStep, RetrievalResult
from scripts.evaluation_common import write_artifact

_QUERY_TYPE_TO_SOURCE = {
    QT_PRODUCT_INQUIRY: SOURCE_PRODUCT,
    QT_RULE_INQUIRY: SOURCE_REGULATION,
    QT_REGULATION_INQUIRY: SOURCE_REGULATION,
    QT_REPORT_INQUIRY: SOURCE_REPORT,
    QT_FAQ_INQUIRY: SOURCE_FAQ,
    QT_TECHNICAL_INQUIRY: SOURCE_FAQ,
}
DEFAULT_DATASET_PATH = Path(__file__).with_name("evaluate_retrieval.sample.json")
_PLACEHOLDER_CHUNK_ID_PREFIXES = ("replace_me_", "example_", "sample_")
ADMISSION_THRESHOLDS = {
    "recall@5": 0.80,
    "recall@10": 0.90,
    "permission_block_accuracy": 1.0,
}


def _load_dataset(dataset_path: str | Path) -> list[dict[str, Any]]:
    dataset = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    if not isinstance(dataset, list):
        raise ValueError("评估集必须是 JSON 数组")
    return dataset


def _contains_placeholder_chunk_ids(dataset: list[dict[str, Any]]) -> bool:
    for item in dataset:
        for doc_id in item.get("relevant_chunk_ids", item.get("relevant_doc_ids", [])):
            if isinstance(doc_id, str) and doc_id.startswith(_PLACEHOLDER_CHUNK_ID_PREFIXES):
                return True
    return False


def _normalize_plan(item: dict[str, Any]) -> list[RetrievalPlanStep]:
    if "plan" in item:
        raw_plan = item["plan"]
        if not isinstance(raw_plan, list):
            raise ValueError("评估样本中的 plan 必须是列表")
        normalized_plan: list[RetrievalPlanStep] = []
        for raw_step in raw_plan:
            if not isinstance(raw_step, dict):
                raise ValueError("评估样本中的 plan step 必须是对象")

            step = RetrievalPlanStep(
                source=str(raw_step.get(PLAN_SOURCE, "")),
                query=str(raw_step.get(PLAN_QUERY, item.get("query", ""))),
                top_k=int(raw_step.get(PLAN_TOP_K, item.get("top_k", 10))),
            )
            filters = raw_step.get(PLAN_FILTERS)
            if PLAN_FILTERS in raw_step:
                step[PLAN_FILTERS] = filters if isinstance(filters, dict) else None
            if PLAN_DENIED in raw_step:
                step[PLAN_DENIED] = bool(raw_step[PLAN_DENIED])
            reason = raw_step.get(PLAN_REASON)
            if isinstance(reason, str):
                step[PLAN_REASON] = reason
            normalized_plan.append(step)
        return normalized_plan

    query = item.get("query", "")
    source = item.get("source")
    if source is None:
        query_type = item.get("expected_query_type")
        if isinstance(query_type, str):
            source = _QUERY_TYPE_TO_SOURCE.get(query_type)

    if source is None:
        raise ValueError("评估样本缺少检索计划。请提供 plan、source 或 expected_query_type。")

    return [
        RetrievalPlanStep(
            source=source,
            query=query,
            top_k=item.get("top_k", 10),
        )
    ]


def _collect_chunk_ids(results: Sequence[RetrievalResult], limit: int) -> set[str]:
    ids = set()
    for result in results[:limit]:
        chunk_id = result.get(RR_METADATA, {}).get(META_CHUNK_ID)
        if chunk_id:
            ids.add(str(chunk_id))
    return ids


def _first_relevant_rank(
    results: Sequence[RetrievalResult], relevant_doc_ids: set[str]
) -> int | None:
    for index, result in enumerate(results, start=1):
        chunk_id = result.get(RR_METADATA, {}).get(META_CHUNK_ID)
        if chunk_id and str(chunk_id) in relevant_doc_ids:
            return index
    return None


def _has_permission_denied(results: Sequence[RetrievalResult]) -> bool:
    return any(bool(result.get(RR_DENIED)) for result in results)


def _average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def evaluate_retrieval(dataset_path: str | Path) -> dict[str, float]:
    """评估检索效果，返回聚合后的指标结果。"""
    dataset = _load_dataset(dataset_path)
    metrics: dict[str, list[float]] = {
        "recall@5": [],
        "recall@10": [],
        "mrr": [],
        "precision@5": [],
        "coverage": [],
        "permission_block_accuracy": [],
    }

    for item in dataset:
        relevant_doc_ids = {
            str(doc_id)
            for doc_id in item.get("relevant_chunk_ids", item.get("relevant_doc_ids", []))
        }
        user_role = item.get("user_role", ROLE_ADVISOR)
        expect_permission_denied = bool(item.get("expected_permission_denied", False))
        plan = _normalize_plan(item)

        retriever = HybridRetriever(
            user_role=user_role,
            data_permissions=ROLE_DATA_PERMISSIONS.get(user_role, []),
        )
        results = retriever.retrieve(plan=plan)

        top5_ids = _collect_chunk_ids(results, limit=5)
        top10_ids = _collect_chunk_ids(results, limit=10)
        actual_permission_denied = _has_permission_denied(results)

        if relevant_doc_ids:
            recall5 = len(top5_ids & relevant_doc_ids) / len(relevant_doc_ids)
            recall10 = len(top10_ids & relevant_doc_ids) / len(relevant_doc_ids)
            first_rank = _first_relevant_rank(results, relevant_doc_ids)
            mrr = 1 / first_rank if first_rank is not None else 0.0
            precision5 = len(top5_ids & relevant_doc_ids) / 5
            coverage = 1.0 if top10_ids & relevant_doc_ids else 0.0
            metrics["recall@5"].append(recall5)
            metrics["recall@10"].append(recall10)
            metrics["mrr"].append(mrr)
            metrics["precision@5"].append(precision5)
        else:
            coverage = 1.0 if expect_permission_denied == actual_permission_denied else 0.0

        permission_block_accuracy = (
            1.0 if actual_permission_denied == expect_permission_denied else 0.0
        )

        metrics["coverage"].append(coverage)
        metrics["permission_block_accuracy"].append(permission_block_accuracy)

    return {
        "samples": float(len(dataset)),
        "recall@5": _average(metrics["recall@5"]),
        "recall@10": _average(metrics["recall@10"]),
        "mrr": _average(metrics["mrr"]),
        "precision@5": _average(metrics["precision@5"]),
        "coverage": _average(metrics["coverage"]),
        "permission_block_accuracy": _average(metrics["permission_block_accuracy"]),
    }


def admission_passed(summary: dict[str, float]) -> bool:
    return all(summary.get(metric, 0.0) >= threshold for metric, threshold in ADMISSION_THRESHOLDS.items())


def main() -> None:
    parser = argparse.ArgumentParser(description="评估 HybridRetriever 检索效果")
    parser.add_argument(
        "dataset_path",
        nargs="?",
        default=str(DEFAULT_DATASET_PATH),
        help=f"评估集 JSON 文件路径，默认使用 {DEFAULT_DATASET_PATH.name}",
    )
    parser.add_argument("--output-root", default="artifacts/evaluation")
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    dataset = _load_dataset(dataset_path)
    summary = evaluate_retrieval(dataset_path)
    print(f"评估集: {dataset_path}")
    if _contains_placeholder_chunk_ids(dataset):
        print(
            "提示: 当前评估集包含占位 relevant_doc_ids，请替换为真实 chunk_id 后再解读召回/MRR 指标。"
        )
    print("检索评估结果：")
    for metric, value in summary.items():
        if metric == "samples":
            print(f"  {metric}: {int(value)}")
        else:
            print(f"  {metric}: {value:.3f}")
    artifact = write_artifact(
        name="retrieval",
        dataset_path=dataset_path,
        summary=summary,
        output_root=args.output_root,
    )
    print(f"评估产物: {artifact}")
    if _contains_placeholder_chunk_ids(dataset) or not admission_passed(summary):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
