"""消融对照实验：同一批问题分别跑简化检索路径与完整 Agent 链路。

对应 eva.md §5 与 §10 里程碑二（"证明复杂度有收益"）：比较正确率
（预期关键词命中率）、拒答误判、延迟和单次 LLM 调用数（成本代理），
判断哪些问题值得支付最复杂路径的成本、哪些走短链路即可。

四条路径（同一批问题、同一 LLM、同一角色权限过滤，差别只在检索深度与编排）：
- direct_tool:  单轮检索原始结果，不过滤不重排，直接进 prompt
- plain_rag:    单轮检索 + 阈值过滤去重（grade_and_filter 的确定性子集）
- rerank_rag:   plain_rag 基础上加 BGE 语义重排
- agent:        完整 Agent 图（规划/多跳/验证/合规）

权限说明：简化路径与 Agent 路径使用同一 HybridRetriever 执行层角色过滤，
不在实验中旁路权限边界。

用法:
    python scripts/evaluate_ablation.py scripts/evaluate_answers.dataset.json
    python scripts/evaluate_ablation.py ... --paths direct_tool,agent --limit 10
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Callable

# 确保项目根目录在 path 中
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluation_common import load_dataset, write_artifact
from src.agents.nodes import _try_rerank_candidates
from src.retrieval.hybrid_retriever import HybridRetriever
from src.schemas.constants import (
    DEFAULT_TOP_K,
    GRADE_TOP_K,
    META_CHUNK_ID,
    META_SOURCE,
    RETRIEVAL_MIN_SCORE,
    RR_CONTENT,
    RR_DENIED,
    RR_METADATA,
    RR_SCORE,
    SOURCE_FAQ,
    SOURCE_PRODUCT,
    SOURCE_REGULATION,
    SOURCE_REPORT,
)
from src.schemas.typed_dicts import RetrievalPlanStep, RetrievalResult

ALL_PATHS = ("direct_tool", "plain_rag", "rerank_rag", "agent")

# 安全兜底文案特征（compose/权限拒绝/验证失败）——拒答判定依据
REFUSAL_MARKERS = ("无法安全返回", "已停止输出", "无权限访问", "无法回答")

CATEGORY_TO_SOURCE = {
    "product": SOURCE_PRODUCT,
    "regulation": SOURCE_REGULATION,
    "report": SOURCE_REPORT,
    "faq": SOURCE_FAQ,
}


class _CountingLLM:
    """包装 LLM 并统计真实模型调用次数，作为单次成本的可比代理。

    ReAct 循环经 bind_tools 得到的 runnable 由 _Bound 代理计数。
    """

    def __init__(self, inner: Any):
        self._inner = inner
        self.calls = 0

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        return self._inner.invoke(*args, **kwargs)

    def bind_tools(self, *args: Any, **kwargs: Any) -> Any:
        bound = self._inner.bind_tools(*args, **kwargs)
        counter = self

        class _Bound:
            def invoke(self, *a: Any, **kw: Any) -> Any:
                counter.calls += 1
                return bound.invoke(*a, **kw)

            def __getattr__(self, name: str) -> Any:
                return getattr(bound, name)

        return _Bound()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def build_plan(item: dict[str, Any]) -> list[RetrievalPlanStep]:
    """按 dataset 类别映射检索源；无单一映射的类别（多跳/歧义等）查全部四源。"""
    source = CATEGORY_TO_SOURCE.get(str(item.get("category", "")))
    query = str(item.get("question", ""))
    if source:
        return [RetrievalPlanStep(source=source, query=query, top_k=DEFAULT_TOP_K)]
    return [
        RetrievalPlanStep(source=one_source, query=query, top_k=DEFAULT_TOP_K)
        for one_source in (SOURCE_REPORT, SOURCE_PRODUCT, SOURCE_REGULATION, SOURCE_FAQ)
    ]


def _grade_results(results: list[RetrievalResult]) -> list[RetrievalResult]:
    """plain_rag 用的确定性过滤：阈值过滤 → 来源+内容指纹去重 → 分数降序截断。

    与 grade_and_filter 的确定性子集对齐（同一组 RETRIEVAL_MIN_SCORE /
    GRADE_TOP_K / 去重键契约），差异仅在不做语义重排。
    """
    candidates: list[RetrievalResult] = []
    seen_evidence: set[tuple[Any, Any]] = set()
    ordered = sorted(
        (r for r in results if not r.get(RR_DENIED)),
        key=lambda r: float(r.get(RR_SCORE, 0) or 0.0),
        reverse=True,
    )
    for result in ordered:
        if float(result.get(RR_SCORE, 0) or 0.0) < RETRIEVAL_MIN_SCORE:
            continue
        metadata = result.get(RR_METADATA, {})
        key = (metadata.get(META_SOURCE), metadata.get(META_CHUNK_ID) or result.get(RR_CONTENT, ""))
        if key in seen_evidence:
            continue
        seen_evidence.add(key)
        candidates.append(result)
        if len(candidates) >= GRADE_TOP_K:
            break
    return candidates


def _format_context(results: list[RetrievalResult]) -> str:
    if not results:
        return "（未检索到可用资料）"
    return "\n\n".join(
        f"[{index + 1}] {result.get(RR_CONTENT, '')}" for index, result in enumerate(results)
    )


def _build_prompt(question: str, context: str) -> str:
    return (
        "你是机构内部投研助手。仅根据以下资料回答问题；"
        "资料不足以回答时，直接说明无法回答，不得编造。\n\n"
        f"问题：{question}\n\n资料：\n{context}"
    )


def is_refusal(answer: str) -> bool:
    return any(marker in answer for marker in REFUSAL_MARKERS)


def keyword_recall(answer: str, expected_keywords: list[str]) -> float | None:
    """预期关键词命中率；无标注（期望拒答的条目）返回 None，不进入正确率均值。"""
    if not expected_keywords:
        return None
    lowered = answer.lower()
    hit = sum(1 for keyword in expected_keywords if keyword.lower() in lowered)
    return hit / len(expected_keywords)


def _run_simplified_path(
    item: dict[str, Any],
    llm: Any,
    retriever_factory: Callable[[str], HybridRetriever],
    mode: str,
) -> dict[str, Any]:
    """direct_tool / plain_rag / rerank_rag 共用骨架，差异只在检索后处理。"""
    started = time.perf_counter()
    calls_before = getattr(llm, "calls", 0)
    retriever = retriever_factory(str(item.get("role", "operations")))
    results = retriever.retrieve(build_plan(item))

    usable = [result for result in results if not result.get(RR_DENIED)]
    if mode == "direct_tool":
        usable = usable[:DEFAULT_TOP_K]
    else:
        usable = _grade_results(usable)
    if mode == "rerank_rag" and usable:
        usable, _status = _try_rerank_candidates(str(item.get("question", "")), usable)

    answer = llm.invoke(_build_prompt(str(item.get("question", "")), _format_context(usable)))
    answer_text = answer.content if hasattr(answer, "content") else str(answer)
    return {
        "answer": answer_text,
        "llm_calls": getattr(llm, "calls", 0) - calls_before,
        "elapsed_ms": max((time.perf_counter() - started) * 1000, 0.0),
        "retrieved": len(usable),
    }


def _run_agent_path(
    item: dict[str, Any],
    llm: Any,
    retriever_factory: Callable[[str], HybridRetriever],
) -> dict[str, Any]:
    """完整 Agent 图链路；临时替换 nodes.llm 为计数代理统计模型调用次数。"""
    import src.agents.nodes as agent_nodes
    from src.agents.graph import build_agent_graph
    from src.api.auth import AuthenticatedUser, build_assistant_initial_state
    from src.schemas.constants import AGENT_RECURSION_LIMIT, ROLE_OPERATIONS
    from src.schemas.request_response import AssistantQARequest

    request = AssistantQARequest(query=str(item.get("question", "")))
    user = AuthenticatedUser(f"eval-{item.get('role', ROLE_OPERATIONS)}", str(item.get("role", ROLE_OPERATIONS)), "eval")
    initial_state = build_assistant_initial_state(request, user)

    original_llm = agent_nodes.llm
    agent_nodes.llm = llm
    started = time.perf_counter()
    calls_before = getattr(llm, "calls", 0)
    try:
        result = build_agent_graph().compile().invoke(
            initial_state,
            {"recursion_limit": AGENT_RECURSION_LIMIT},
        )
    finally:
        agent_nodes.llm = original_llm
    return {
        "answer": str(result.get("final_answer", "")),
        "llm_calls": getattr(llm, "calls", 0) - calls_before,
        "elapsed_ms": max((time.perf_counter() - started) * 1000, 0.0),
        "retrieved": len(result.get("retrieval_results", [])),
    }


# agent 路径执行器可被测试替换（默认用真实 Agent 图）
AgentRunner = Callable[[dict[str, Any], Any, Callable[[str], HybridRetriever]], dict[str, Any]]


def evaluate_ablation(
    dataset_path: str | Path,
    *,
    paths: tuple[str, ...] = ALL_PATHS,
    limit: int | None = None,
    llm: Any | None = None,
    retriever_factory: Callable[[str], HybridRetriever] | None = None,
    agent_runner: AgentRunner | None = None,
) -> dict[str, dict[str, float]]:
    """跑消融对照实验并按路径汇总指标。

    llm / retriever_factory / agent_runner 可注入替身供测试使用；
    默认用真实 LLM、检索组件与 Agent 图。
    """
    dataset = load_dataset(dataset_path)
    if limit is not None:
        dataset = dataset[:limit]
    if not dataset:
        raise ValueError("评估集为空")

    if llm is None:
        from src.agents.nodes import llm as graph_llm

        llm = _CountingLLM(graph_llm)
    if retriever_factory is None:
        retriever_factory = lambda role: HybridRetriever(user_role=role)  # noqa: E731
    if agent_runner is None:
        agent_runner = _run_agent_path

    per_path: dict[str, list[dict[str, Any]]] = {path: [] for path in paths}
    for item in dataset:
        for path in paths:
            if path == "agent":
                outcome = agent_runner(item, llm, retriever_factory)
            elif path in ("direct_tool", "plain_rag", "rerank_rag"):
                outcome = _run_simplified_path(item, llm, retriever_factory, mode=path)
            else:
                raise ValueError(f"未知路径: {path}")
            outcome["recall"] = keyword_recall(outcome["answer"], list(item.get("expected_keywords", [])))
            outcome["refused"] = is_refusal(outcome["answer"])
            per_path[path].append(outcome)

    summary: dict[str, dict[str, float]] = {}
    for path, outcomes in per_path.items():
        labeled = [o for o in outcomes if o["recall"] is not None]
        # 拒答误判：有预期答案标注（expected_keywords 非空）却被安全兜底文案拒绝
        refusals_on_labeled = sum(1 for o in labeled if o["refused"])
        summary[path] = {
            "samples": float(len(outcomes)),
            "keyword_recall": (
                sum(o["recall"] for o in labeled) / len(labeled) if labeled else 0.0
            ),
            "refusal_false_positive_rate": (
                refusals_on_labeled / len(labeled) if labeled else 0.0
            ),
            "avg_latency_ms": sum(o["elapsed_ms"] for o in outcomes) / len(outcomes),
            "avg_llm_calls": sum(o["llm_calls"] for o in outcomes) / len(outcomes),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="消融对照：简化检索路径 vs 完整 Agent 链路")
    parser.add_argument("dataset_path")
    parser.add_argument(
        "--paths",
        default=",".join(ALL_PATHS),
        help=f"逗号分隔的参与路径（默认全部）：{', '.join(ALL_PATHS)}",
    )
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 条")
    parser.add_argument("--output-root", default="artifacts/evaluation")
    args = parser.parse_args()

    paths = tuple(p.strip() for p in args.paths.split(",") if p.strip())
    summary = evaluate_ablation(args.dataset_path, paths=paths, limit=args.limit)
    artifact = write_artifact(
        name="ablation",
        dataset_path=args.dataset_path,
        summary=summary,
        output_root=args.output_root,
    )
    for path, metrics in summary.items():
        print(path, metrics)
    print(f"评估产物: {artifact}")


if __name__ == "__main__":
    main()
