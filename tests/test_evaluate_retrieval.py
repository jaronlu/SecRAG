from __future__ import annotations

import json
from pathlib import Path

from scripts import evaluate_retrieval as eval_script
from src.schemas.constants import (
    META_CHUNK_ID,
    PLAN_QUERY,
    PLAN_SOURCE,
    PLAN_TOP_K,
    ROLE_ADVISOR,
    ROLE_TECHNICAL,
    RR_DENIED,
    RR_METADATA,
    SOURCE_FAQ,
    SOURCE_PRODUCT,
)


def _write_dataset(tmp_path: Path, payload: list[dict]) -> Path:
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return dataset_path


def test_evaluate_retrieval_uses_explicit_plan(monkeypatch, tmp_path):
    captured = {}

    class FakeHybridRetriever:
        def __init__(self, user_role: str):
            captured["user_role"] = user_role

        def retrieve(self, plan):
            captured["plan"] = plan
            return [{
                RR_METADATA: {META_CHUNK_ID: "chunk-1"},
            }]

    monkeypatch.setattr(eval_script, "HybridRetriever", FakeHybridRetriever)

    dataset_path = _write_dataset(tmp_path, [{
        "query": "产品风险等级",
        "user_role": ROLE_ADVISOR,
        "plan": [{
            PLAN_SOURCE: SOURCE_PRODUCT,
            PLAN_QUERY: "显式计划查询",
            PLAN_TOP_K: 3,
        }],
        "relevant_doc_ids": ["chunk-1"],
    }])

    summary = eval_script.evaluate_retrieval(dataset_path)

    assert captured["user_role"] == ROLE_ADVISOR
    assert captured["plan"] == [{
        PLAN_SOURCE: SOURCE_PRODUCT,
        PLAN_QUERY: "显式计划查询",
        PLAN_TOP_K: 3,
    }]
    assert summary["recall@5"] == 1.0
    assert summary["mrr"] == 1.0


def test_evaluate_retrieval_infers_source_from_query_type(monkeypatch, tmp_path):
    captured = {}

    class FakeHybridRetriever:
        def __init__(self, user_role: str):
            captured["user_role"] = user_role

        def retrieve(self, plan):
            captured["plan"] = plan
            return []

    monkeypatch.setattr(eval_script, "HybridRetriever", FakeHybridRetriever)

    dataset_path = _write_dataset(tmp_path, [{
        "query": "怎么走系统操作流程",
        "user_role": ROLE_TECHNICAL,
        "expected_query_type": "technical_inquiry",
        "relevant_doc_ids": [],
    }])

    eval_script.evaluate_retrieval(dataset_path)

    assert captured["user_role"] == ROLE_TECHNICAL
    assert captured["plan"] == [{
        PLAN_SOURCE: SOURCE_FAQ,
        PLAN_QUERY: "怎么走系统操作流程",
        PLAN_TOP_K: 10,
    }]


def test_evaluate_retrieval_computes_permission_accuracy(monkeypatch, tmp_path):
    class FakeHybridRetriever:
        def __init__(self, user_role: str):
            self.user_role = user_role

        def retrieve(self, plan):
            if self.user_role == ROLE_TECHNICAL:
                return [{RR_DENIED: True, RR_METADATA: {}}]
            return [{RR_METADATA: {META_CHUNK_ID: "chunk-2"}}]

    monkeypatch.setattr(eval_script, "HybridRetriever", FakeHybridRetriever)

    dataset_path = _write_dataset(tmp_path, [
        {
            "query": "内部资料",
            "user_role": ROLE_TECHNICAL,
            "expected_query_type": "report_inquiry",
            "expected_permission_denied": True,
            "relevant_doc_ids": [],
        },
        {
            "query": "产品风险",
            "user_role": ROLE_ADVISOR,
            "source": SOURCE_PRODUCT,
            "relevant_doc_ids": ["chunk-2"],
        },
    ])

    summary = eval_script.evaluate_retrieval(dataset_path)

    assert summary["permission_block_accuracy"] == 1.0
    assert summary["coverage"] == 1.0
    assert summary["precision@5"] == 0.1


def test_evaluate_retrieval_requires_plan_or_source_hint(tmp_path):
    dataset_path = _write_dataset(tmp_path, [{
        "query": "缺少计划",
        "user_role": ROLE_ADVISOR,
        "relevant_doc_ids": [],
    }])

    try:
        eval_script.evaluate_retrieval(dataset_path)
    except ValueError as exc:
        assert "plan" in str(exc)
    else:
        raise AssertionError("expected ValueError")
