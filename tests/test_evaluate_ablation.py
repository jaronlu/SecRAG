"""消融对照实验脚本测试：全部使用替身组件，不依赖真实 LLM 与检索库。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from scripts.evaluate_ablation import (
    ALL_PATHS,
    _CountingLLM,
    _grade_results,
    build_plan,
    evaluate_ablation,
    is_refusal,
    keyword_recall,
)
from src.retrieval.hybrid_retriever import HybridRetriever
from src.schemas.constants import (
    SOURCE_FAQ,
    SOURCE_PRODUCT,
    SOURCE_REGULATION,
    SOURCE_REPORT,
)
from src.schemas.typed_dicts import RetrievalResult


def _write(tmp_path: Path, payload: list[dict]) -> Path:
    path = tmp_path / "ablation.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def _result(
    content: str, score: float = 0.9, source: str = "report_search", chunk: str = "c1"
) -> RetrievalResult:
    return {
        "content": content,
        "score": score,
        "denied": False,
        "metadata": {"source": source, "chunk_id": chunk},
    }


class _FakeRetriever:
    """按角色返回固定候选，记录收到的检索计划。"""

    def __init__(self, results: list[RetrievalResult]):
        self.results = results
        self.plans: list[list[dict]] = []

    def retrieve(self, plan):
        self.plans.append(plan)
        return [dict(r) for r in self.results]


class _FakeMessage:
    def __init__(self, content: str):
        self.content = content


class _FakeLLM:
    """固定回答的替身 LLM，经 _CountingLLM 包装后应产生调用计数。"""

    def __init__(self, answer: str = "货币基金风险等级为 R1 低风险。"):
        self.answer = answer
        self.prompts: list[str] = []

    def invoke(self, prompt, *args, **kwargs):
        self.prompts.append(str(prompt))
        return _FakeMessage(self.answer)

    def bind_tools(self, tools, *args, **kwargs):
        counter_self = self

        class _Bound:
            def invoke(self, messages, *a, **kw):
                counter_self.prompts.append(str(messages))
                return _FakeMessage(counter_self.answer)

        return _Bound()


# ══════════════════════════════════════════════════════════════════════
# 纯函数单元
# ══════════════════════════════════════════════════════════════════════


def test_keyword_recall_skips_unlabeled_items():
    assert keyword_recall("R1 低风险", ["R1", "低风险"]) == 1.0
    assert keyword_recall("完全无关", ["R1"]) == 0.0
    assert keyword_recall("任意答案", []) is None


def test_is_refusal_matches_safety_fallback_text():
    assert is_refusal("当前答案未通过来源或数字验证，无法安全返回。请重试。")
    assert is_refusal("当前角色无权限访问完成该请求所需的数据源。")
    assert not is_refusal("货币基金风险等级为 R1。")


def test_build_plan_maps_category_and_expands_unknown_categories():
    plan = build_plan({"category": "report", "question": "Q"})
    assert len(plan) == 1 and plan[0].get("source") == SOURCE_REPORT

    expanded = build_plan({"category": "multi_hop", "question": "Q"})
    assert {step.get("source") for step in expanded} == {
        SOURCE_REPORT, SOURCE_PRODUCT, SOURCE_REGULATION, SOURCE_FAQ,
    }


def test_grade_results_applies_threshold_dedupe_and_topk():
    results = [
        _result("低分被滤掉", score=0.3, chunk="c0"),
        _result("重复证据", score=0.9, chunk="c1"),
        _result("重复证据", score=0.8, chunk="c1"),
        _result("正常证据", score=0.7, chunk="c2"),
    ]

    graded = _grade_results(results)

    contents = [r["content"] for r in graded]
    assert "低分被滤掉" not in contents
    assert contents.count("重复证据") == 1
    assert "正常证据" in contents


# ══════════════════════════════════════════════════════════════════════
# 端到端汇总（替身注入）
# ══════════════════════════════════════════════════════════════════════


def test_counting_llm_counts_direct_and_bound_invokes():
    counter = _CountingLLM(_FakeLLM())
    counter.invoke("a")
    counter.bind_tools([]).invoke("b")

    assert counter.calls == 2


def test_evaluate_ablation_reports_all_paths_with_fake_components(tmp_path):
    dataset = _write(
        tmp_path,
        [
            {
                "id": "abl_001",
                "category": "report",
                "question": "货币基金的风险等级是什么？",
                "role": "advisor",
                "expected_keywords": ["R1", "低风险"],
            }
        ],
    )
    llm = _CountingLLM(_FakeLLM())
    retriever = _FakeRetriever([_result("货币基金风险等级为 R1 低风险。")])

    def fake_agent_runner(item, llm, retriever_factory):
        before = getattr(llm, "calls", 0)
        llm.invoke("agent prompt")
        return {
            "answer": "货币基金风险等级为 R1 低风险。",
            "llm_calls": getattr(llm, "calls", 0) - before,
            "elapsed_ms": 1.0,
            "retrieved": 1,
        }

    summary = evaluate_ablation(
        dataset,
        paths=ALL_PATHS,
        llm=llm,
        retriever_factory=lambda role: cast(HybridRetriever, retriever),
        agent_runner=fake_agent_runner,
    )

    assert set(summary) == set(ALL_PATHS)
    for path, metrics in summary.items():
        assert metrics["samples"] == 1.0
        assert metrics["keyword_recall"] == 1.0, f"{path} 应命中预期关键词"
        assert metrics["refusal_false_positive_rate"] == 0.0
    # 简化路径每次恰好 1 次 LLM 调用；agent 路径按替身实现计数
    assert summary["direct_tool"]["avg_llm_calls"] == 1.0
    assert summary["plain_rag"]["avg_llm_calls"] == 1.0
    assert summary["rerank_rag"]["avg_llm_calls"] == 1.0
    assert summary["agent"]["avg_llm_calls"] >= 1.0


def test_evaluate_ablation_counts_refusal_false_positive(tmp_path):
    dataset = _write(
        tmp_path,
        [
            {
                "id": "abl_002",
                "category": "report",
                "question": "普通问题",
                "role": "advisor",
                "expected_keywords": ["R1"],
            }
        ],
    )
    # 简化路径返回拒答文案 → 应答被拒计为误判
    llm = _CountingLLM(_FakeLLM(answer="当前答案未通过来源或数字验证，无法安全返回。"))
    retriever = _FakeRetriever([])

    summary = evaluate_ablation(
        dataset,
        paths=("plain_rag",),
        llm=llm,
        retriever_factory=lambda role: cast(HybridRetriever, retriever),
    )

    assert summary["plain_rag"]["refusal_false_positive_rate"] == 1.0
    assert summary["plain_rag"]["keyword_recall"] == 0.0


def test_evaluate_ablation_rejects_unknown_path(tmp_path):
    dataset = _write(tmp_path, [{"question": "Q", "expected_keywords": []}])

    with pytest.raises(ValueError, match="未知路径"):
        evaluate_ablation(
            dataset,
            paths=("nope",),
            llm=_CountingLLM(_FakeLLM()),
            retriever_factory=lambda role: cast(HybridRetriever, _FakeRetriever([])),
        )
