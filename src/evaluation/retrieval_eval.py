"""P2-5: 检索质量评测脚本——评估向量检索+BM25+RRF 的召回质量。

用法:
    python -m src.evaluation.retrieval_eval
    python -m src.evaluation.retrieval_eval --top-k 5 --min-score 0.3

评测指标:
    - Hit@K: 前 K 个结果中包含预期关键词的查询比例
    - MRR: 第一个相关结果的倒数排名均值
    - Avg Score: 所有结果的平均相似度分数
    - Empty Rate: 返回空结果的查询比例
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field

from src.retrieval.hybrid_retriever import HybridRetriever
from src.schemas.constants import (
    PLAN_QUERY,
    ROLE_ADVISOR,
    RR_CONTENT,
    RR_METADATA,
    RR_SCORE,
    SOURCE_PRODUCT,
    SOURCE_REGULATION,
)
from src.schemas.typed_dicts import RetrievalPlanStep

# 评测用例：query + 预期关键词（结果内容或 metadata 中应包含）
EVAL_CASES: list[dict] = [
    {
        "query": "理财产品风险等级划分标准",
        "expected_keywords": ["风险等级", "R1", "R2", "R3", "R4", "R5"],
        "source": SOURCE_PRODUCT,
    },
    {
        "query": "公募基金销售适当性管理要求",
        "expected_keywords": ["适当性", "风险承受能力", "匹配"],
        "source": SOURCE_REGULATION,
    },
    {
        "query": "私募投资基金合格投资者标准",
        "expected_keywords": ["合格投资者", "净资产", "金融资产", "100万"],
        "source": SOURCE_REGULATION,
    },
    {
        "query": "股票质押式回购业务规则",
        "expected_keywords": ["质押", "回购", "股票"],
        "source": SOURCE_REGULATION,
    },
    {
        "query": "货币市场基金投资范围",
        "expected_keywords": ["货币", "现金", "短期", "国债"],
        "source": SOURCE_PRODUCT,
    },
]


@dataclass
class EvalResult:
    query: str
    hit: bool
    first_relevant_rank: int | None
    result_count: int
    avg_score: float
    top_results: list[dict] = field(default_factory=list)


def run_eval(top_k: int = 5, min_score: float = 0.0) -> list[EvalResult]:
    """运行评测，返回每个用例的结果。"""
    retriever = HybridRetriever(user_role=ROLE_ADVISOR)
    results: list[EvalResult] = []

    for case in EVAL_CASES:
        plan = [
            RetrievalPlanStep(
                source=case["source"],
                query=case[PLAN_QUERY],
                top_k=top_k,
            )
        ]
        retrieved = retriever.retrieve(plan)
        # 过滤被拒绝的结果和低分结果
        usable = [r for r in retrieved if not r.get("denied") and r.get(RR_SCORE, 0) >= min_score]

        # 检查是否命中预期关键词
        first_relevant_rank = None
        hit = False
        for rank, r in enumerate(usable):
            content = r.get(RR_CONTENT, "")
            meta = r.get(RR_METADATA, {})
            text = content + " " + " ".join(str(v) for v in meta.values())
            if any(kw in text for kw in case["expected_keywords"]):
                hit = True
                first_relevant_rank = rank + 1
                break

        avg_score = sum(r.get(RR_SCORE, 0) for r in usable) / len(usable) if usable else 0.0
        results.append(
            EvalResult(
                query=case["query"],
                hit=hit,
                first_relevant_rank=first_relevant_rank,
                result_count=len(usable),
                avg_score=avg_score,
                top_results=[
                    {
                        "score": round(r.get(RR_SCORE, 0), 4),
                        "content": r.get(RR_CONTENT, "")[:100],
                    }
                    for r in usable[:3]
                ],
            )
        )
    return results


def print_report(results: list[EvalResult]) -> None:
    """打印评测报告。"""
    total = len(results)
    hits = sum(1 for r in results if r.hit)
    mrr = (
        sum(1.0 / r.first_relevant_rank for r in results if r.first_relevant_rank is not None)
        / total
        if total
        else 0
    )
    empty = sum(1 for r in results if r.result_count == 0)
    avg_score = sum(r.avg_score for r in results) / total if total else 0

    print("=" * 60)
    print("检索质量评测报告")
    print("=" * 60)
    print(f"用例总数:    {total}")
    print(f"Hit@K:       {hits}/{total} ({hits / total * 100:.1f}%)" if total else "Hit@K: N/A")
    print(f"MRR:         {mrr:.4f}")
    print(
        f"空结果率:    {empty}/{total} ({empty / total * 100:.1f}%)" if total else "空结果率: N/A"
    )
    print(f"平均分数:    {avg_score:.4f}")
    print("-" * 60)
    for r in results:
        status = "HIT" if r.hit else "MISS"
        rank = f"rank={r.first_relevant_rank}" if r.first_relevant_rank else "no relevant"
        print(f"[{status}] {r.query} ({r.result_count} results, {rank})")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="SecRAG 检索质量评测")
    parser.add_argument("--top-k", type=int, default=5, help="每个查询返回的结果数")
    parser.add_argument("--min-score", type=float, default=0.0, help="最低相似度分数过滤")
    parser.add_argument("--json", action="store_true", help="以 JSON 格式输出")
    args = parser.parse_args()

    results = run_eval(top_k=args.top_k, min_score=args.min_score)

    if args.json:
        print(json.dumps([r.__dict__ for r in results], ensure_ascii=False, indent=2))
    else:
        print_report(results)

    # CI 用：Hit@K < 60% 时返回非零退出码
    hits = sum(1 for r in results if r.hit)
    if results and hits / len(results) < 0.6:
        print(f"WARNING: Hit@K={hits / len(results) * 100:.1f}% 低于 60% 阈值", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
