"""SemanticCache 快照持久化测试（P1-1）。

缓存条目必须保存 store 时的 compliance/verification 终态快照，
命中时原样返回，替换 main.py 曾硬编码的 {"passed": True}。
"""

from __future__ import annotations

import sqlite3

import numpy as np
import pytest

from src.utils.semantic_cache import SemanticCache


@pytest.fixture
def cache(monkeypatch, tmp_path):
    """启用且嵌入恒定的缓存实例——同一查询再次 lookup 必然命中。"""
    cache = SemanticCache(db_path=str(tmp_path / "cache.db"), enabled=True)
    monkeypatch.setattr(cache, "_embed", lambda text: np.array([1.0, 0.0]))
    return cache


def _drop_snapshot_columns(db_path):
    """把新 schema 退化成旧版本（无快照列），模拟历史数据库。"""
    conn = sqlite3.connect(db_path)
    conn.execute("ALTER TABLE cache_entries DROP COLUMN compliance")
    conn.execute("ALTER TABLE cache_entries DROP COLUMN verification")
    conn.commit()
    conn.close()


def test_store_persists_compliance_and_verification_snapshots(cache):
    compliance = {"passed": True, "flags": [], "risk_disclosure": "【风险提示】市场有风险。"}
    verification = {"passed": True, "issues": [], "confidence": "high"}

    assert cache.store(
        "货币基金风险等级",
        "货币基金风险等级为低。",
        citations=[{"source": "a.pdf"}],
        confidence="high",
        role="advisor",
        compliance=compliance,
        verification=verification,
    ) is True

    hit = cache.lookup("货币基金风险等级", role="advisor")
    assert hit is not None
    assert hit["compliance"] == compliance
    assert hit["verification"] == verification


def test_migrated_legacy_rows_report_empty_snapshots(monkeypatch, tmp_path):
    """旧库行（无快照列）经迁移补列后命中，按空快照返回而不是伪造 passed=True。"""
    db_path = tmp_path / "legacy.db"
    cache = SemanticCache(db_path=str(db_path), enabled=True)
    monkeypatch.setattr(cache, "_embed", lambda text: np.array([1.0, 0.0]))
    assert cache.store("旧问题", "旧回答", role="advisor") is True
    _drop_snapshot_columns(db_path)

    reopened = SemanticCache(db_path=str(db_path), enabled=True)
    monkeypatch.setattr(reopened, "_embed", lambda text: np.array([1.0, 0.0]))

    hit = reopened.lookup("旧问题", role="advisor")
    assert hit is not None
    assert hit["compliance"] == {}
    assert hit["verification"] == {}


def test_init_db_adds_snapshot_columns_to_legacy_table(tmp_path):
    """旧库（无快照列）打开时自动补列。"""
    db_path = tmp_path / "legacy.db"
    SemanticCache(db_path=str(db_path), enabled=True).close()
    _drop_snapshot_columns(db_path)

    SemanticCache(db_path=str(db_path), enabled=True)

    columns = {
        row[1] for row in sqlite3.connect(db_path).execute("PRAGMA table_info(cache_entries)")
    }
    assert {"compliance", "verification"} <= columns


def _cache_with_distinct_embeddings(monkeypatch, tmp_path, name="stats.db"):
    """embedding 按查询文本区分的缓存实例，可构造真实的 miss。"""
    cache = SemanticCache(db_path=str(tmp_path / name), enabled=True)
    embeddings = {
        "货币基金风险等级": np.array([1.0, 0.0]),
        "股票交易手续费": np.array([0.0, 1.0]),
    }
    monkeypatch.setattr(cache, "_embed", lambda text: embeddings[text])
    return cache


def test_hit_rate_counts_real_lookup_misses(monkeypatch, tmp_path):
    """命中率按真实 lookup 请求口径统计，而不是用条目数近似 miss。"""
    cache = _cache_with_distinct_embeddings(monkeypatch, tmp_path)
    cache.store("货币基金风险等级", "货币基金风险等级为低。", role="advisor")

    assert cache.lookup("股票交易手续费", role="advisor") is None
    assert cache.lookup("货币基金风险等级", role="advisor") is not None

    stats = cache.get_stats()
    assert stats["lookup_total"] == 2
    assert stats["lookup_hits"] == 1
    assert stats["lookup_misses"] == 1
    assert stats["hit_rate"] == 0.5


def test_disabled_cache_does_not_count_lookups(tmp_path):
    """enabled=False 或空查询的短路返回不构成真实请求，不进入命中口径。"""
    cache = SemanticCache(db_path=str(tmp_path / "off.db"), enabled=False)

    assert cache.lookup("任何问题") is None

    stats = cache.get_stats()
    assert stats["lookup_total"] == 0
    assert stats["hit_rate"] == 0.0


def test_clear_all_resets_lookup_counters(monkeypatch, tmp_path):
    cache = _cache_with_distinct_embeddings(monkeypatch, tmp_path, name="reset.db")
    cache.store("货币基金风险等级", "货币基金风险等级为低。", role="advisor")
    cache.lookup("货币基金风险等级", role="advisor")

    assert cache.clear_all() >= 1

    stats = cache.get_stats()
    assert stats["lookup_total"] == 0
    assert stats["hit_rate"] == 0.0


# ══════════════════════════════════════════════════════════════════════
# ISSUE-26：缓存绑定身份、授权范围、客户上下文、上下文摘要与知识库版本
# ══════════════════════════════════════════════════════════════════════


def _bound_cache(monkeypatch, tmp_path, name="bound.db") -> SemanticCache:
    """嵌入恒定的启用缓存：命中与否只由绑定维度决定。"""
    cache = SemanticCache(db_path=str(tmp_path / name), enabled=True)
    monkeypatch.setattr(cache, "_embed", lambda text: np.array([1.0, 0.0]))
    return cache


def _store_with_binding(cache: SemanticCache, query: str, **binding: object) -> bool:
    from src.utils.semantic_cache import build_cache_binding

    return cache.store(
        query,
        "缓存答案。",
        role=str(binding.get("role", "advisor")),
        binding=build_cache_binding(query=query, **binding),  # type: ignore[arg-type]
    )


def _lookup_with_binding(cache: SemanticCache, query: str, **binding: object):
    from src.utils.semantic_cache import build_cache_binding

    return cache.lookup(
        query,
        role=str(binding.get("role", "advisor")),
        binding=build_cache_binding(query=query, **binding),  # type: ignore[arg-type]
    )


def test_hit_requires_same_user(monkeypatch, tmp_path):
    """同角色不同用户不得互相命中（身份绑定）。"""
    cache = _bound_cache(monkeypatch, tmp_path)
    _store_with_binding(cache, "客户持仓情况", user_id="user_a")

    assert _lookup_with_binding(cache, "客户持仓情况", user_id="user_b") is None
    assert _lookup_with_binding(cache, "客户持仓情况", user_id="user_a") is not None


def test_hit_requires_same_client_context(monkeypatch, tmp_path):
    cache = _bound_cache(monkeypatch, tmp_path)
    _store_with_binding(cache, "该客户适当性评级", client_id="client_1")

    assert _lookup_with_binding(cache, "该客户适当性评级", client_id="client_2") is None
    assert _lookup_with_binding(cache, "该客户适当性评级", client_id="client_1") is not None


def test_hit_requires_same_permission_scope(monkeypatch, tmp_path):
    cache = _bound_cache(monkeypatch, tmp_path)
    _store_with_binding(cache, "内部制度要点", data_permissions=("internal",))

    assert (
        _lookup_with_binding(cache, "内部制度要点", data_permissions=("internal", "confidential"))
        is None
    )
    assert _lookup_with_binding(cache, "内部制度要点", data_permissions=("internal",)) is not None


def test_hit_requires_same_context_summary(monkeypatch, tmp_path):
    """会话上下文摘要变化（追问指代不同）不得复用旧答案。"""
    cache = _bound_cache(monkeypatch, tmp_path)
    _store_with_binding(cache, "它的风险等级是多少", conversation_summary="产品：稳健增利")

    assert (
        _lookup_with_binding(cache, "它的风险等级是多少", conversation_summary="产品：货币基金")
        is None
    )
    assert (
        _lookup_with_binding(cache, "它的风险等级是多少", conversation_summary="产品：稳健增利")
        is not None
    )


def test_hit_requires_same_knowledge_base_version(monkeypatch, tmp_path):
    """知识库版本变化后旧条目不得命中（重新入库即失效）。"""
    cache = _bound_cache(monkeypatch, tmp_path)
    _store_with_binding(cache, "茅台2026半年报营业收入", kb_version="kb-v1")

    assert _lookup_with_binding(cache, "茅台2026半年报营业收入", kb_version="kb-v2") is None
    assert _lookup_with_binding(cache, "茅台2026半年报营业收入", kb_version="kb-v1") is not None


def test_hit_requires_same_normalized_query(monkeypatch, tmp_path):
    """语义相似不等于同一问题：规范化问题不同即视为不同条目。"""
    cache = _bound_cache(monkeypatch, tmp_path)
    _store_with_binding(cache, "货币基金风险等级")

    assert _lookup_with_binding(cache, "股票基金风险等级") is None
    assert _lookup_with_binding(cache, "货币基金风险等级") is not None


def test_normalize_query_collapses_case_and_whitespace():
    from src.utils.semantic_cache import normalize_query

    # NFKC 把全角问号折成半角，空白折叠为单空格，英文大小写归一
    assert normalize_query("  MONEY   Fund 风险等级？ ") == "money fund 风险等级?"
    assert normalize_query("MONEY  FUND") == normalize_query("money fund")


def test_build_cache_binding_is_stable_and_discriminating():
    from src.utils.semantic_cache import build_cache_binding

    base = dict(query="q", role="advisor", user_id="u1")
    assert build_cache_binding(**base) == build_cache_binding(**base)
    assert build_cache_binding(**base) != build_cache_binding(**{**base, "user_id": "u2"})


def test_permission_scope_is_order_insensitive():
    from src.utils.semantic_cache import build_cache_binding

    left = build_cache_binding(query="q", role="advisor", data_permissions=("a", "b"))
    right = build_cache_binding(query="q", role="advisor", data_permissions=("b", "a"))

    assert left.permission_scope == right.permission_scope


def test_knowledge_base_version_tracks_registry_content(tmp_path):
    """知识库版本指纹来自入库注册表，入库后必然变化（跨进程可见）。"""
    from src.ingestion.registry import DocumentRegistryStore, DocumentRegistryUpdate
    from src.utils.semantic_cache import knowledge_base_version

    db_path = tmp_path / "registry.db"
    store = DocumentRegistryStore(db_path)
    empty_version = knowledge_base_version(db_path)
    assert empty_version

    store.upsert_success(
        DocumentRegistryUpdate(
            doc_id="d1",
            source_uri="file:///tmp/report.pdf",
            relative_path="report.pdf",
            doc_type="research_report",
            title="跟踪报告",
            stock_code="600519",
            publish_date="2026-05-20",
            file_hash="file-hash",
            metadata_hash="metadata-hash",
            parse_hash="parse-hash",
            parser_version="pv",
            chunker_version="cv",
            embedding_model="em",
            chunk_count=3,
            doc_version=1,
            last_seen_at="2026-09-29T00:00:00Z",
            last_ingested_at="2026-09-29T00:00:00Z",
        )
    )

    assert knowledge_base_version(db_path) != empty_version


def test_knowledge_base_version_is_empty_without_registry(tmp_path):
    from src.utils.semantic_cache import knowledge_base_version

    assert knowledge_base_version(tmp_path / "missing.db") == ""


def test_cache_enabled_by_default_after_issue_26():
    from src.utils.semantic_cache import DEFAULT_CACHE_ENABLED

    assert DEFAULT_CACHE_ENABLED is True
