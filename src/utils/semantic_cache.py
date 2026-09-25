"""P1-4: 语义缓存层——基于 embedding 相似度的查询结果缓存。

用 embedding 向量计算查询相似度，相似度超过阈值时复用历史回答，
避免重复的检索 + LLM 调用，提升响应速度、降低 API 成本。

设计要点：
- 相似度匹配：cosine similarity > threshold（默认 0.9）
- 角色隔离：不同角色的缓存独立，避免权限越权
- TTL 过期：默认 24 小时，过期自动失效
- 命中率统计：hit/miss 计数，可查询命中率
- 线程安全：SQLite WAL 模式 + 连接池

用法:
    from src.utils.semantic_cache import SemanticCache
    cache = SemanticCache()
    hit = cache.lookup("货币基金风险等级", role="advisor")
    if hit:
        answer = hit["answer"]
    else:
        answer = ... # 正常 RAG 流程
        cache.store("货币基金风险等级", answer, citations, confidence, role="advisor")
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from src.config import config

# ══════════════════════════════════════════════════════════════════════
# 配置常量
# ══════════════════════════════════════════════════════════════════════

DEFAULT_CACHE_THRESHOLD = 0.90  # cosine similarity 阈值，超过则命中
DEFAULT_CACHE_TTL_SECONDS = 86400  # 24 小时
DEFAULT_CACHE_DB_PATH = "./data/semantic_cache.db"
DEFAULT_CACHE_ENABLED = False  # 答案缓存默认关闭：缓存未绑定会话上下文与知识库版本，
# 且命中路径无法复现会话保存/审计流程；重新启用前需满足 issues.md 一.1 的条件

# 嵌入模型懒加载（避免启动时加载模型）
_embedding_model = None
_embedding_lock = threading.Lock()


def _get_embedding_model():
    """懒加载 sentence-transformers 模型，线程安全。"""
    global _embedding_model
    if _embedding_model is None:
        with _embedding_lock:
            if _embedding_model is None:
                from sentence_transformers import SentenceTransformer

                _embedding_model = SentenceTransformer(config.embedding_model)
    return _embedding_model


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """计算两个向量的余弦相似度。"""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class SemanticCache:
    """语义缓存——基于 embedding 相似度的查询结果缓存。

    线程安全，支持多角色隔离、TTL 过期、命中率统计。
    """

    def __init__(
        self,
        db_path: str | None = None,
        threshold: float = DEFAULT_CACHE_THRESHOLD,
        ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS,
        enabled: bool = DEFAULT_CACHE_ENABLED,
    ):
        """初始化语义缓存。

        Args:
            db_path: SQLite 数据库路径
            threshold: 余弦相似度阈值（0-1），超过则命中
            ttl_seconds: 缓存过期时间（秒）
            enabled: 是否启用缓存
        """
        self.db_path = db_path or DEFAULT_CACHE_DB_PATH
        self.threshold = threshold
        self.ttl_seconds = ttl_seconds
        self.enabled = enabled
        self._local = threading.local()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """获取线程本地 SQLite 连接。"""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = conn
        return conn

    def _init_db(self):
        """初始化数据库表和索引。"""
        conn = self._get_conn()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cache_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                query_embedding TEXT NOT NULL,
                answer TEXT NOT NULL,
                citations TEXT DEFAULT '[]',
                confidence TEXT DEFAULT '',
                role TEXT NOT NULL DEFAULT '',
                created_at REAL NOT NULL,
                expires_at REAL NOT NULL,
                hit_count INTEGER DEFAULT 0
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_cache_role_expires ON cache_entries(role, expires_at)"
        )
        # P1-1: 终态快照列——旧库自动补列，历史条目读空串时按空快照处理
        existing_columns = {row[1] for row in conn.execute("PRAGMA table_info(cache_entries)")}
        if "compliance" not in existing_columns:
            conn.execute("ALTER TABLE cache_entries ADD COLUMN compliance TEXT DEFAULT ''")
        if "verification" not in existing_columns:
            conn.execute("ALTER TABLE cache_entries ADD COLUMN verification TEXT DEFAULT ''")
        conn.commit()

    def _embed(self, text: str) -> np.ndarray:
        """计算文本的 embedding 向量。"""
        model = _get_embedding_model()
        result = model.encode(text, normalize_embeddings=True)
        return np.asarray(result)

    def lookup(
        self,
        query: str,
        role: str = "",
    ) -> Optional[dict[str, Any]]:
        """查询缓存，命中则返回缓存结果，未命中返回 None。

        Args:
            query: 用户查询
            role: 用户角色（用于角色隔离）

        Returns:
            命中时返回 {query, answer, citations, confidence, similarity, hit_count,
            compliance, verification}；未命中返回 None
        """
        if not self.enabled or not query:
            return None

        now = time.time()
        conn = self._get_conn()

        # 查询该角色下未过期的所有缓存
        cursor = conn.execute(
            "SELECT id, query, query_embedding, answer, citations, confidence, hit_count, compliance, verification FROM cache_entries WHERE role = ? AND expires_at > ?",
            (role, now),
        )
        rows = cursor.fetchall()

        if not rows:
            return None

        # 计算查询 embedding
        query_emb = self._embed(query)

        # 找最相似的缓存条目
        best_similarity = 0.0
        best_row = None
        for row in rows:
            cached_emb = np.array(json.loads(row[2]))
            sim = _cosine_similarity(query_emb, cached_emb)
            if sim > best_similarity:
                best_similarity = sim
                best_row = row

        if best_similarity >= self.threshold and best_row is not None:
            # 命中：更新 hit_count
            conn.execute(
                "UPDATE cache_entries SET hit_count = hit_count + 1 WHERE id = ?",
                (best_row[0],),
            )
            conn.commit()
            return {
                "query": best_row[1],
                "answer": best_row[3],
                "citations": json.loads(best_row[4]),
                "confidence": best_row[5],
                "similarity": round(best_similarity, 4),
                "hit_count": best_row[6] + 1,
                # P1-1: 返回 store 时的终态快照；旧条目空串按空快照处理
                "compliance": json.loads(best_row[7]) if best_row[7] else {},
                "verification": json.loads(best_row[8]) if best_row[8] else {},
            }

        return None

    def store(
        self,
        query: str,
        answer: str,
        citations: list[dict[str, Any]] | None = None,
        confidence: str = "",
        role: str = "",
        compliance: dict[str, Any] | None = None,
        verification: dict[str, Any] | None = None,
    ) -> bool:
        """存储查询结果到缓存。

        Args:
            query: 用户查询
            answer: 回答内容
            citations: 引用列表
            confidence: 置信度
            role: 用户角色
            compliance: 终态合规结果快照（P1-1）
            verification: 终态验证结果快照（P1-1）

        Returns:
            是否存储成功
        """
        if not self.enabled or not query or not answer:
            return False

        now = time.time()
        query_emb = self._embed(query)

        conn = self._get_conn()
        try:
            conn.execute(
                "INSERT INTO cache_entries (query, query_embedding, answer, citations, confidence, role, created_at, expires_at, compliance, verification) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    query,
                    json.dumps(query_emb.tolist()),
                    answer,
                    json.dumps(citations or []),
                    confidence,
                    role,
                    now,
                    now + self.ttl_seconds,
                    # P1-1: 终态快照随条目一起落库
                    json.dumps(compliance or {}),
                    json.dumps(verification or {}),
                ),
            )
            conn.commit()
            return True
        except Exception:
            conn.rollback()
            return False

    def get_stats(self) -> dict[str, Any]:
        """获取缓存统计信息。

        Returns:
            {total_entries, active_entries, total_hits, hit_rate, avg_similarity}
        """
        conn = self._get_conn()
        now = time.time()

        total = conn.execute("SELECT COUNT(*) FROM cache_entries").fetchone()[0]
        active = conn.execute(
            "SELECT COUNT(*) FROM cache_entries WHERE expires_at > ?", (now,)
        ).fetchone()[0]
        total_hits = conn.execute(
            "SELECT COALESCE(SUM(hit_count), 0) FROM cache_entries"
        ).fetchone()[0]

        # 命中率 = total_hits / (total_hits + total_entries) 近似
        # 因为每次 store 算一次 miss，每次 lookup 命中算一次 hit
        total_requests = total_hits + total
        hit_rate = round(total_hits / total_requests, 4) if total_requests > 0 else 0.0

        return {
            "total_entries": total,
            "active_entries": active,
            "expired_entries": total - active,
            "total_hits": total_hits,
            "hit_rate": hit_rate,
            "threshold": self.threshold,
            "ttl_seconds": self.ttl_seconds,
            "enabled": self.enabled,
        }

    def clear_expired(self) -> int:
        """清理过期缓存条目。

        Returns:
            清理的条目数
        """
        conn = self._get_conn()
        now = time.time()
        cursor = conn.execute("DELETE FROM cache_entries WHERE expires_at <= ?", (now,))
        conn.commit()
        return cursor.rowcount

    def clear_all(self) -> int:
        """清空所有缓存。

        Returns:
            清理的条目数
        """
        conn = self._get_conn()
        cursor = conn.execute("DELETE FROM cache_entries")
        conn.commit()
        return cursor.rowcount

    def close(self):
        """关闭数据库连接。"""
        conn = getattr(self._local, "conn", None)
        if conn:
            conn.close()
            self._local.conn = None


# 全局单例
_semantic_cache: Optional[SemanticCache] = None
_semantic_cache_lock = threading.Lock()


def get_semantic_cache() -> SemanticCache:
    """获取语义缓存单例。

    开关由 config.semantic_cache_enabled 控制，默认关闭。
    """
    global _semantic_cache
    if _semantic_cache is None:
        with _semantic_cache_lock:
            if _semantic_cache is None:
                _semantic_cache = SemanticCache(enabled=config.semantic_cache_enabled)
    return _semantic_cache
