"""检索评估脚本：按评估集运行 HybridRetriever 并汇总核心指标。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.retrieval.hybrid_retriever import HybridRetriever
from src.schemas.constants import (
    META_CHUNK_ID,
    PLAN_QUERY,
    PLAN_SOURCE,
    PLAN_TOP_K,
    QT_FAQ_INQUIRY,
    QT_PRODUCT_INQUIRY,
    QT_REGULATION_INQUIRY,
    QT_REPORT_INQUIRY,
    QT_RULE_INQUIRY,
    QT_TECHNICAL_INQUIRY,
    ROLE_ADVISOR,
    RR_DENIED,
    RR_METADATA,
    SOURCE_FAQ,
    SOURCE_PRODUCT,
    SOURCE_REGULATION,
    SOURCE_REPORT,
)

_QUERY_TYPE_TO_SOURCE = {
    QT_PRODUCT_INQUIRY: SOURCE_PRODUCT,
    QT_RULE_INQUIRY: SOURCE_REGULATION,
    QT_REGULATION_INQUIRY: SOURCE_REGULATION,
    QT_REPORT_INQUIRY: SOURCE_REPORT,
    QT_FAQ_INQUIRY: SOURCE_FAQ,
    QT_TECHNICAL_INQUIRY: SOURCE_FAQ,
}


def _load_dataset(dataset_path: str | Path) -> list[dict[str, Any]]:
    dataset = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    if not isinstance(dataset, list):
        raise ValueError("评估集必须是 JSON 数组")
    return dataset


def _normalize_plan(item: dict[str, Any]) -> list[dict[str, Any]]:
    if "plan" in item:
        plan = item["plan"]
        if not isinstance(plan, list):
            raise ValueError("评估样本中的 plan 必须是列表")
        return plan

    query = item.get("query", "")
    source = item.get("source")
    if source is None:
        query_type = item.get("expected_query_type")
        if isinstance(query_type, str):
            source = _QUERY_TYPE_TO_SOURCE.get(query_type)

    if source is None:
        raise ValueError(
            "评估样本缺少检索计划。请提供 plan、source 或 expected_query_type。"
        )

    return [{
        PLAN_SOURCE: source,
        PLAN_QUERY: query,
        PLAN_TOP_K: item.get("top_k", 10),
    }]


def _collect_chunk_ids(results: list[dict[str, Any]], limit: int) -> set[str]:
    ids = set()
    for result in results[:limit]:
        chunk_id = result.get(RR_METADATA, {}).get(META_CHUNK_ID)
        if chunk_id:
            ids.add(str(chunk_id))
    return ids


def _first_relevant_rank(results: list[dict[str, Any]], relevant_doc_ids: set[str]) -> int | None:
    for index, result in enumerate(results, start=1):
        chunk_id = result.get(RR_METADATA, {}).get(META_CHUNK_ID)
        if chunk_id and str(chunk_id) in relevant_doc_ids:
            return index
    return None


def _has_permission_denied(results: list[dict[str, Any]]) -> bool:
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
        relevant_doc_ids = {str(doc_id) for doc_id in item.get("relevant_doc_ids", [])}
        user_role = item.get("user_role", ROLE_ADVISOR)
        expect_permission_denied = bool(item.get("expected_permission_denied", False))
        plan = _normalize_plan(item)

        retriever = HybridRetriever(user_role=user_role)
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
        else:
            recall5 = 0.0
            recall10 = 0.0
            mrr = 0.0
            precision5 = 0.0
            coverage = 1.0 if expect_permission_denied == actual_permission_denied else 0.0

        permission_block_accuracy = 1.0 if actual_permission_denied == expect_permission_denied else 0.0

        metrics["recall@5"].append(recall5)
        metrics["recall@10"].append(recall10)
        metrics["mrr"].append(mrr)
        metrics["precision@5"].append(precision5)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="评估 HybridRetriever 检索效果")
    parser.add_argument("dataset_path", help="评估集 JSON 文件路径")
    args = parser.parse_args()

    summary = evaluate_retrieval(args.dataset_path)
    print("检索评估结果：")
    for metric, value in summary.items():
        if metric == "samples":
            print(f"  {metric}: {int(value)}")
        else:
            print(f"  {metric}: {value:.3f}")


if __name__ == "__main__":
    main()
