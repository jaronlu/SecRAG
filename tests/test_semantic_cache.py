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
