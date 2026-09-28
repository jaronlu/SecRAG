"""date_day 存量迁移脚本测试（todo/issues.md ISSUE-1）。"""

import chromadb
import pytest

from scripts.migrate_date_day import migrate_collection

_DIM = 4


def _vector(seed: int) -> list[float]:
    return [((seed + i) % 7) / 7 for i in range(_DIM)]


@pytest.fixture()
def collection():
    client = chromadb.EphemeralClient()
    col = client.create_collection("migrate_date_day_test")
    yield col
    client.delete_collection("migrate_date_day_test")


def _seed(col, records: list[dict]) -> None:
    col.add(
        ids=[record["id"] for record in records],
        embeddings=[_vector(i) for i in range(len(records))],
        documents=[f"doc-{i}" for i in range(len(records))],
        metadatas=[record["metadata"] for record in records],
    )


def _metadata_of(col, chunk_id: str) -> dict:
    return col.get(ids=[chunk_id], include=["metadatas"])["metadatas"][0]


def test_dry_run_counts_without_writing(collection):
    _seed(collection, [
        {"id": "a", "metadata": {"doc_type": "research_report", "date": "2026-05-25"}},
        {"id": "b", "metadata": {"doc_type": "research_report", "date": "2024.0"}},
        {"id": "c", "metadata": {"doc_type": "faq", "date": ""}},
        {"id": "d", "metadata": {"doc_type": "product", "date": "2026-03-21", "date_day": 20260321}},
    ])

    summary = migrate_collection(collection, apply=False)

    assert summary["scanned"] == 4
    assert summary["backfilled"] == 2
    assert summary["already_had_date_day"] == 1
    assert summary["unparseable"] == 1
    assert summary["unparseable|faq|''"] == 1
    # dry-run 不写库
    assert _metadata_of(collection, "a").get("date_day") is None


def test_apply_backfills_and_is_idempotent(collection):
    _seed(collection, [
        {"id": "a", "metadata": {"doc_type": "research_report", "date": "2026-05-25", "title": "t"}},
        {"id": "b", "metadata": {"doc_type": "research_report", "date": "2024.0"}},
        {"id": "c", "metadata": {"doc_type": "faq", "date": "2025"}},
    ])

    summary = migrate_collection(collection, apply=True)

    assert summary["backfilled"] == 3
    assert _metadata_of(collection, "a")["date_day"] == 20260525
    assert _metadata_of(collection, "b")["date_day"] == 20240101
    assert _metadata_of(collection, "c")["date_day"] == 20250101
    # 原有元数据不丢失
    assert _metadata_of(collection, "a")["title"] == "t"

    # 幂等：重跑不重复写
    second = migrate_collection(collection, apply=True)
    assert second["backfilled"] == 0
    assert second["already_had_date_day"] == 3


def test_unparseable_dates_reported_by_doc_type(collection):
    _seed(collection, [
        {"id": "a", "metadata": {"doc_type": "faq", "date": "最近三个月"}},
        {"id": "b", "metadata": {"doc_type": "faq", "date": ""}},
    ])

    summary = migrate_collection(collection, apply=True)

    assert summary["unparseable"] == 2
    assert summary["backfilled"] == 0
    assert summary["unparseable|faq|'最近三个月'"] == 1
    assert summary["unparseable|faq|''"] == 1
    assert _metadata_of(collection, "a").get("date_day") is None
