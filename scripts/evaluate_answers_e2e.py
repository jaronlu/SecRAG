"""P0-1: 端到端回答质量评估脚本。

流程：
  1. 加载标注数据集（问题 + 角色 + 预期关键词）
  2. 逐条调用 SecRAG API 获取回答和引用
  3. 用 LLM-as-Judge 从四维度评分（准确性、引用相关性、合规性、完整性）
  4. 规则-based 辅助检查（预期关键词命中、PII 泄露、权限拒绝）
  5. 生成评估报告（JSON + Markdown）

用法:
    python scripts/evaluate_answers_e2e.py
    python scripts/evaluate_answers_e2e.py --dataset scripts/evaluate_answers.dataset.json
    python scripts/evaluate_answers_e2e.py --base-url http://127.0.0.1:8000
    python scripts/evaluate_answers_e2e.py --limit 10  # 只评估前 10 条
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# 确保项目根目录在 path 中
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import httpx

from scripts.evaluation_common import current_commit_sha, write_artifact
from src.evaluation.answer_judge import AnswerJudge, DIMENSIONS, DIMENSION_LABELS

# Demo token 映射
ROLE_TO_TOKEN: dict[str, str] = {
    "advisor": "demo-advisor",
    "institutional_sales": "demo-sales",
    "compliance": "demo-compliance",
    "operations": "demo-ops",
    "technical": "demo-tech",
}

DEFAULT_DATASET = PROJECT_ROOT / "scripts" / "evaluate_answers.dataset.json"
DEFAULT_BASE_URL = "http://127.0.0.1:8000"


def load_dataset(path: Path) -> list[dict[str, Any]]:
    """加载评估数据集。"""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"评估集必须是 JSON 数组: {path}")
    return data


def call_api(
    base_url: str,
    question: str,
    role: str,
    timeout: float = 60.0,
) -> dict[str, Any]:
    """调用 SecRAG QA 接口。

    Returns:
        包含 answer, references, confidence, compliance_flags 等字段的 dict
    """
    token = ROLE_TO_TOKEN.get(role, "demo-advisor")
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload = {"query": question}

    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(
                f"{base_url}/v1/assistant/qa",
                headers=headers,
                json=payload,
            )
        if response.status_code == 200:
            return response.json()
        else:
            return {
                "error": f"HTTP {response.status_code}",
                "detail": response.text[:500],
                "answer": "",
                "references": [],
            }
    except httpx.TimeoutException:
        return {"error": "timeout", "answer": "", "references": []}
    except Exception as e:
        return {"error": str(e), "answer": "", "references": []}


def rule_based_checks(
    item: dict[str, Any],
    api_result: dict[str, Any],
) -> dict[str, Any]:
    """规则-based 辅助检查，不依赖 LLM。

    检查项：
    - 预期关键词命中
    - PII 泄露检测
    - 权限拒绝检测
    - 合规标记检测
    - 澄清触发检测
    """
    answer = api_result.get("answer", "")
    references = api_result.get("references", [])
    checks: dict[str, Any] = {}

    # 1. 预期关键词命中
    expected_keywords = item.get("expected_keywords", [])
    if expected_keywords:
        hit_keywords = [kw for kw in expected_keywords if kw in answer]
        checks["keyword_hit_rate"] = round(len(hit_keywords) / len(expected_keywords), 2)
        checks["hit_keywords"] = hit_keywords
        checks["missed_keywords"] = [kw for kw in expected_keywords if kw not in answer]

    # 2. PII 泄露检测
    if item.get("expect_no_pii_leak"):
        pii_patterns = ["13812345678", "110101199001011234"]
        leaked = [p for p in pii_patterns if p in answer]
        checks["pii_leaked"] = bool(leaked)
        checks["pii_leaked_items"] = leaked

    # 3. 权限拒绝检测
    if item.get("expect_permission_denied"):
        denied_refs = [r for r in references if r.get("denied")]
        checks["permission_denied"] = len(denied_refs) > 0 or "权限" in answer or "无权" in answer
        checks["denied_reference_count"] = len(denied_refs)

    # 4. 合规标记检测
    if item.get("expect_compliance_flag"):
        compliance_flags = api_result.get("compliance_flags", [])
        checks["compliance_flagged"] = len(compliance_flags) > 0 or "不构成" in answer or "投资建议" in answer
        checks["compliance_flags"] = compliance_flags

    # 5. 澄清触发检测
    if item.get("expect_clarify"):
        checks["clarified"] = "澄清" in answer or "确认" in answer or "不明确" in answer or "补充" in answer

    # 6. 安全拦截检测
    if item.get("expect_security_block"):
        checks["security_blocked"] = any(
            kw in answer for kw in ["无法", "不能", "拒绝", "安全", "权限", "不支持"]
        )

    # 7. 引用数量
    checks["reference_count"] = len(references)
    checks["answer_length"] = len(answer)
    checks["has_error"] = "error" in api_result

    return checks


def run_evaluation(
    dataset: list[dict[str, Any]],
    base_url: str,
    judge: AnswerJudge,
    delay: float = 1.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """运行完整评估流程。

    Returns:
        (详细结果列表, 汇总统计)
    """
    results: list[dict[str, Any]] = []
    judge_inputs: list[dict[str, Any]] = []

    total = len(dataset)
    for i, item in enumerate(dataset):
        question = item.get("question", "")
        role = item.get("role", "advisor")
        print(f"  [{i+1}/{total}] {role}: {question[:50]}...", flush=True)

        # 1. 调用 API
        start_time = time.perf_counter()
        api_result = call_api(base_url, question, role)
        latency_ms = round((time.perf_counter() - start_time) * 1000, 1)

        # 2. 规则-based 检查
        rule_checks = rule_based_checks(item, api_result)

        # 3. 收集 LLM-as-judge 输入
        answer = api_result.get("answer", "")
        references = api_result.get("references", [])
        judge_inputs.append(
            {
                "question": question,
                "answer": answer,
                "references": references,
            }
        )

        results.append(
            {
                "id": item.get("id", f"item_{i}"),
                "category": item.get("category", "unknown"),
                "difficulty": item.get("difficulty", "unknown"),
                "question": question,
                "role": role,
                "api_result": {
                    "answer": answer[:1000],
                    "reference_count": len(references),
                    "confidence": api_result.get("confidence", ""),
                    "compliance_flags": api_result.get("compliance_flags", []),
                    "error": api_result.get("error"),
                },
                "latency_ms": latency_ms,
                "rule_checks": rule_checks,
            }
        )

        # 避免请求过快
        if delay > 0 and i < total - 1:
            time.sleep(delay)

    # 4. LLM-as-judge 批量评判
    print(f"\n  开始 LLM-as-Judge 评判（{len(judge_inputs)} 条）...", flush=True)
    judge_results = judge.judge_batch(judge_inputs)

    # 5. 合并评判结果
    for i, jr in enumerate(judge_results):
        results[i]["judge_scores"] = jr["scores"]
        results[i]["overall_score"] = jr["overall_score"]
        results[i]["passed"] = jr["passed"]

    # 6. 汇总
    summary = AnswerJudge.summarize(
        [
            {"scores": r["judge_scores"], "overall_score": r["overall_score"], "passed": r["passed"]}
            for r in results
        ]
    )

    # 补充规则-based 统计
    keyword_hit_rates = [
        r["rule_checks"]["keyword_hit_rate"]
        for r in results
        if "keyword_hit_rate" in r["rule_checks"]
    ]
    summary["rule_based"] = {
        "avg_keyword_hit_rate": round(sum(keyword_hit_rates) / len(keyword_hit_rates), 2)
        if keyword_hit_rates
        else 0,
        "avg_latency_ms": round(sum(r["latency_ms"] for r in results) / len(results), 1),
        "error_count": sum(1 for r in results if r["rule_checks"]["has_error"]),
        "pii_leak_count": sum(
            1 for r in results if r["rule_checks"].get("pii_leaked", False)
        ),
        "security_block_count": sum(
            1 for r in results if r["rule_checks"].get("security_blocked", False)
        ),
        "compliance_flag_count": sum(
            1 for r in results if r["rule_checks"].get("compliance_flagged", False)
        ),
        "permission_denied_count": sum(
            1 for r in results if r["rule_checks"].get("permission_denied", False)
        ),
        "clarify_count": sum(
            1 for r in results if r["rule_checks"].get("clarified", False)
        ),
    }

    # 按类别统计
    category_stats: dict[str, dict[str, Any]] = {}
    for r in results:
        cat = r["category"]
        if cat not in category_stats:
            category_stats[cat] = {"count": 0, "scores": [], "overall": []}
        category_stats[cat]["count"] += 1
        category_stats[cat]["scores"].append(r["judge_scores"])
        category_stats[cat]["overall"].append(r["overall_score"])

    summary["by_category"] = {}
    for cat, stats in category_stats.items():
        dim_avgs = {}
        for dim in DIMENSIONS:
            scores = [s[dim]["score"] for s in stats["scores"]]
            dim_avgs[dim] = round(sum(scores) / len(scores), 2) if scores else 0
        summary["by_category"][cat] = {
            "count": stats["count"],
            "overall_avg": round(sum(stats["overall"]) / len(stats["overall"]), 2),
            "dimensions": dim_avgs,
        }

    return results, summary


def main():
    parser = argparse.ArgumentParser(description="SecRAG 端到端回答质量评估")
    parser.add_argument("--dataset", type=str, default=str(DEFAULT_DATASET), help="评估数据集路径")
    parser.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL, help="API 基础 URL")
    parser.add_argument("--limit", type=int, default=0, help="只评估前 N 条（0=全部）")
    parser.add_argument("--delay", type=float, default=0.5, help="每条请求间隔秒数")
    parser.add_argument("--output", type=str, default="", help="输出目录")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"错误: 评估数据集不存在: {dataset_path}")
        sys.exit(1)

    print("=" * 70)
    print("SecRAG 端到端回答质量评估")
    print("=" * 70)
    print(f"  数据集: {dataset_path}")
    print(f"  API: {args.base_url}")
    print(f"  提交: {current_commit_sha()}")
    print(f"  时间: {datetime.now(timezone.utc).isoformat()}")

    # 加载数据集
    dataset = load_dataset(dataset_path)
    if args.limit > 0:
        dataset = dataset[: args.limit]
    print(f"  评估条数: {len(dataset)}")

    # 检查 API 可用性
    print("\n  检查 API 可用性...", flush=True)
    try:
        with httpx.Client(timeout=10) as client:
            health = client.get(f"{args.base_url}/health")
        print(f"  API 健康: {health.status_code == 200}")
    except Exception as e:
        print(f"  警告: API 不可用 ({e})，评估将全部失败")

    # 初始化评判器
    print("\n  初始化 LLM-as-Judge...", flush=True)
    judge = AnswerJudge(temperature=0.0)

    # 运行评估
    print("\n  开始评估...", flush=True)
    results, summary = run_evaluation(dataset, args.base_url, judge, delay=args.delay)

    # 输出汇总
    print("\n" + "=" * 70)
    print("评估结果汇总")
    print("=" * 70)
    print(f"  总体平均分: {summary['overall_avg']} / 5.0")
    print(f"  通过率: {summary['pass_rate'] * 100:.1f}% ({summary['passed']}/{summary['total']})")
    print(f"  平均延迟: {summary['rule_based']['avg_latency_ms']} ms")
    print(f"  关键词命中率: {summary['rule_based']['avg_keyword_hit_rate'] * 100:.1f}%")
    print()
    print("  各维度平均分:")
    for dim in DIMENSIONS:
        d = summary["dimensions"][dim]
        print(f"    {d['label']}: {d['avg']}")
    print()
    print("  规则检查:")
    rb = summary["rule_based"]
    print(f"    错误数: {rb['error_count']}")
    print(f"    PII 泄露: {rb['pii_leak_count']}")
    print(f"    安全拦截: {rb['security_block_count']}")
    print(f"    合规标记: {rb['compliance_flag_count']}")
    print(f"    权限拒绝: {rb['permission_denied_count']}")
    print(f"    澄清触发: {rb['clarify_count']}")
    print()
    print("  按类别:")
    for cat, stats in summary["by_category"].items():
        print(f"    {cat} ({stats['count']}条): 总体={stats['overall_avg']}")

    # 保存结果
    output_root = args.output or "artifacts/evaluation"
    output_dir = Path(output_root) / current_commit_sha()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 详细结果 JSON
    detail_path = output_dir / "answer_evaluation_detail.json"
    detail_path.write_text(
        json.dumps(
            {
                "commit_sha": current_commit_sha(),
                "dataset": str(dataset_path),
                "base_url": args.base_url,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "summary": summary,
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n  详细结果: {detail_path}")

    # Markdown 报告
    report = AnswerJudge.generate_markdown_report(summary)
    # 追加规则-based 和类别统计
    report += "\n## 规则-based 统计\n\n"
    report += f"- 平均延迟: {rb['avg_latency_ms']} ms\n"
    report += f"- 关键词命中率: {rb['avg_keyword_hit_rate'] * 100:.1f}%\n"
    report += f"- 错误数: {rb['error_count']}\n"
    report += f"- PII 泄露: {rb['pii_leak_count']}\n"
    report += f"- 安全拦截: {rb['security_block_count']}\n"
    report += f"- 合规标记: {rb['compliance_flag_count']}\n"
    report += f"- 权限拒绝: {rb['permission_denied_count']}\n"
    report += f"- 澄清触发: {rb['clarify_count']}\n"

    report += "\n## 按类别统计\n\n"
    report += "| 类别 | 条数 | 总体分 | 准确性 | 引用相关性 | 合规性 | 完整性 |\n"
    report += "|---|---|---|---|---|---|---|\n"
    for cat, stats in summary["by_category"].items():
        dims = stats["dimensions"]
        report += (
            f"| {cat} | {stats['count']} | {stats['overall_avg']} | "
            f"{dims['accuracy']} | {dims['citation_relevance']} | "
            f"{dims['compliance']} | {dims['completeness']} |\n"
        )

    report_path = output_dir / "answer_evaluation_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"  评估报告: {report_path}")

    # 同时保存到 write_artifact 格式（兼容现有流程）
    write_artifact(
        name="answer_evaluation",
        dataset_path=dataset_path,
        summary=summary,
        output_root=output_root,
    )

    print("\n" + "=" * 70)
    print("评估完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
