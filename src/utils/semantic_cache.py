"""P1-4: 语义缓存层——基于 embedding 相似度的查询结果缓存。

用 embedding 向量计算查询相似度，相似度超过阈值时复用历史回答，
避免重复的检索 + LLM 调用，提升响应速度、降低 API 成本。

设计要点：
- 相似度匹配：cosine similarity > threshold（默认 0.9）
- 绑定维度（ISSUE-26）：身份（user_id）、授权范围（permission_scope）、
  客户上下文（client_id）、规范化问题、上下文摘要哈希、知识库版本。
  只有六个维度全部一致才允许命中，避免跨用户/跨客户/跨知识库版本复用
- TTL 过期：默认 24 小时，过期自动失效
- 命中率统计：hit/miss 计数，可查询命中率
- 线程安全：SQLite WAL 模式 + 连接池

用法:
    from src.utils.semantic_cache import SemanticCache, build_cache_binding
    cache = SemanticCache()
    binding = build_cache_binding(
        query=question,
        role=role,
        user_id=user_id,
        conversation_summary=summary,
    )
    hit = cache.lookup(question, role=role, binding=binding)
    if hit is None:
        answer = run_rag(question)
        cache.store(question, answer, role=role, binding=binding)
    else:
        answer = hit["answer"]
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

from src.config import config
from src.schemas.constants import INGEST_REGISTRY_DB_PATH
from src.utils.sqlite_support import connect_sqlite

# ══════════════════════════════════════════════════════════════════════
# 配置常量
# ══════════════════════════════════════════════════════════════════════

DEFAULT_CACHE_THRESHOLD = 0.90  # cosine similarity 阈值，超过则命中
DEFAULT_CACHE_TTL_SECONDS = 86400  # 24 小时
DEFAULT_CACHE_DB_PATH = "./data/semantic_cache.db"
# ISSUE-26：启用条件（原 issues.md 一.1）已全部落地——缓存绑定身份与授权范围、
# 客户上下文、规范化问题、上下文摘要哈希与知识库版本；只缓存验证与合规均通过的
# 成功终态；命中路径仍写审计事件并保存会话回合。因此默认启用。
DEFAULT_CACHE_ENABLED = True

_WHITESPACE_RE = re.compile(r"\s+")

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


def normalize_query(query: str) -> str:
    """规范化问题文本（ISSUE-26）：NFKC 归一 + 折叠空白 + 去首尾 + 英文小写。"""
    if not isinstance(query, str):
        return ""
    normalized = unicodedata.normalize("NFKC", query)
    normalized = _WHITESPACE_RE.sub(" ", normalized).strip()
    return normalized.casefold()


def _sha256_prefix(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def context_hash(conversation_summary: str) -> str:
    """会话上下文摘要哈希（ISSUE-26）：摘要变化即视为不同上下文。"""
    return _sha256_prefix(normalize_query(conversation_summary or ""))


def permission_scope(data_permissions: Iterable[str] | None) -> str:
    """授权范围指纹（ISSUE-26）：与顺序无关，权限集合变化即失效。"""
    values = sorted({str(value) for value in (data_permissions or [])})
    return _sha256_prefix("|".join(values))


def knowledge_base_version(db_path: str | Path | None = None) -> str:
    """知识库版本指纹（ISSUE-26）：文档数 + 最近入库时间。

    入库成功会写 document_registry.last_ingested_at，因此知识库内容变化后
    指纹必然改变，缓存条目随之自然失效——不依赖进程内通知（入库 CLI 与
    API 是两个进程，进程内失效信号传不过去）。注册表缺失或不可读时返回
    空串，此时所有条目共享同一"未知版本"，行为与未启用版本绑定一致。
    """
    path = Path(db_path or INGEST_REGISTRY_DB_PATH)
    if not path.exists():
        return ""
    try:
        conn = connect_sqlite(path)
        try:
            row = conn.execute(
                "SELECT COUNT(*), COALESCE(MAX(last_ingested_at), '') FROM document_registry"
            ).fetchone()
        finally:
            conn.close()
    except sqlite3.Error:
        return ""
    if row is None:
        return ""
    return _sha256_prefix(f"{row[0]}:{row[1]}")


@dataclass(frozen=True)
class CacheBinding:
    """缓存条目的绑定维度（ISSUE-26）。

    六个维度全部参与匹配：身份、授权范围、客户上下文、规范化问题、
    上下文摘要哈希、知识库版本。
    """

    role: str
    user_id: str
    client_id: str
    permission_scope: str
    normalized_query: str
    context_hash: str
    kb_version: str


def build_cache_binding(
    *,
    query: str,
    role: str = "",
    user_id: str = "",
    client_id: str = "",
    data_permissions: Iterable[str] | None = None,
    conversation_summary: str = "",
    kb_version: str | None = None,
) -> CacheBinding:
    """从请求上下文构造缓存绑定（ISSUE-26）。"""
    return CacheBinding(
        role=str(role or ""),
        user_id=str(user_id or ""),
        client_id=str(client_id or ""),
        permission_scope=permission_scope(data_permissions),
        normalized_query=normalize_query(query),
        context_hash=context_hash(conversation_summary),
        kb_version=knowledge_base_version() if kb_version is None else str(kb_version),
    )


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
        # 命中率按真实 lookup 请求口径统计（进程内计数，重启归零）；
        # 独立于条目 hit_count，不受过期清理影响
        self._stats_lock = threading.Lock()
        self._lookup_hits = 0
        self._lookup_misses = 0
        self._init_db()

    def _record_lookup(self, *, hit: bool) -> None:
        with self._stats_lock:
            if hit:
                self._lookup_hits += 1
            else:
                self._lookup_misses += 1

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
        if "user_id" not in existing_columns:
            conn.execute("ALTER TABLE cache_entries ADD COLUMN user_id TEXT DEFAULT ''")
        if "client_id" not in existing_columns:
            conn.execute("ALTER TABLE cache_entries ADD COLUMN client_id TEXT DEFAULT ''")
        if "permission_scope" not in existing_columns:
            conn.execute("ALTER TABLE cache_entries ADD COLUMN permission_scope TEXT DEFAULT ''")
        if "normalized_query" not in existing_columns:
            conn.execute("ALTER TABLE cache_entries ADD COLUMN normalized_query TEXT DEFAULT ''")
        if "context_hash" not in existing_columns:
            conn.execute("ALTER TABLE cache_entries ADD COLUMN context_hash TEXT DEFAULT ''")
        if "kb_version" not in existing_columns:
            conn.execute("ALTER TABLE cache_entries ADD COLUMN kb_version TEXT DEFAULT ''")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_cache_binding "
            "ON cache_entries(role, user_id, client_id, kb_version, expires_at)"
        )
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
        *,
        binding: CacheBinding | None = None,
    ) -> Optional[dict[str, Any]]:
        """查询缓存，命中则返回缓存结果，未命中返回 None。

        Args:
            query: 用户查询
            role: 用户角色（未传 binding 时用于构造默认绑定）
            binding: 缓存绑定（ISSUE-26）。六个维度必须全部一致才参与相似度比较；
                未传时按 (query, role) 构造默认绑定，其余维度为空串

        Returns:
            命中时返回 {query, answer, citations, confidence, similarity, hit_count,
            compliance, verification}；未命中返回 None
        """
        if not self.enabled or not query:
            return None

        effective = binding or build_cache_binding(query=query, role=role)
        now = time.time()
        conn = self._get_conn()

        # 只有绑定维度全部一致的条目才是候选，再做语义相似度比较
        cursor = conn.execute(
            "SELECT id, query, query_embedding, answer, citations, confidence, hit_count, compliance, verification FROM cache_entries WHERE role = ? AND user_id = ? AND client_id = ? AND permission_scope = ? AND normalized_query = ? AND context_hash = ? AND kb_version = ? AND expires_at > ?",
            (
                effective.role,
                effective.user_id,
                effective.client_id,
                effective.permission_scope,
                effective.normalized_query,
                effective.context_hash,
                effective.kb_version,
                now,
            ),
        )
        rows = cursor.fetchall()

        if not rows:
            self._record_lookup(hit=False)
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
            self._record_lookup(hit=True)
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

        self._record_lookup(hit=False)
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
        *,
        binding: CacheBinding | None = None,
    ) -> bool:
        """存储查询结果到缓存。

        Args:
            query: 用户查询
            answer: 回答内容
            citations: 引用列表
            confidence: 置信度
            role: 用户角色（未传 binding 时用于构造默认绑定）
            compliance: 终态合规结果快照（P1-1）
            verification: 终态验证结果快照（P1-1）
            binding: 缓存绑定（ISSUE-26）；未传时按 (query, role) 构造

        Returns:
            是否存储成功
        """
        if not self.enabled or not query or not answer:
            return False

        effective = binding or build_cache_binding(query=query, role=role)
        now = time.time()
        query_emb = self._embed(query)

        conn = self._get_conn()
        try:
            conn.execute(
                "INSERT INTO cache_entries (query, query_embedding, answer, citations, confidence, role, created_at, expires_at, compliance, verification, user_id, client_id, permission_scope, normalized_query, context_hash, kb_version) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    query,
                    json.dumps(query_emb.tolist()),
                    answer,
                    json.dumps(citations or []),
                    confidence,
                    effective.role,
                    now,
                    now + self.ttl_seconds,
                    # P1-1: 终态快照随条目一起落库
                    json.dumps(compliance or {}),
                    json.dumps(verification or {}),
                    # ISSUE-26：绑定维度随条目一起落库，命中时逐维比对
                    effective.user_id,
                    effective.client_id,
                    effective.permission_scope,
                    effective.normalized_query,
                    effective.context_hash,
                    effective.kb_version,
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
            {total_entries, active_entries, total_hits, lookup_total, lookup_hits,
            lookup_misses, hit_rate, threshold, ttl_seconds, enabled}

        hit_rate 按真实 lookup 请求口径计算：lookup_hits / lookup_total，
        只统计 enabled 且查询非空的真实请求；条目级 total_hits 是历史
        hit_count 汇总，仅作参考，不参与命中率计算。
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

        with self._stats_lock:
            lookup_hits = self._lookup_hits
            lookup_misses = self._lookup_misses
        lookup_total = lookup_hits + lookup_misses
        hit_rate = round(lookup_hits / lookup_total, 4) if lookup_total > 0 else 0.0

        return {
            "total_entries": total,
            "active_entries": active,
            "expired_entries": total - active,
            "total_hits": total_hits,
            "lookup_total": lookup_total,
            "lookup_hits": lookup_hits,
            "lookup_misses": lookup_misses,
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
        with self._stats_lock:
            self._lookup_hits = 0
            self._lookup_misses = 0
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

    开关由 config.semantic_cache_enabled 控制。ISSUE-26 起启用条件已落地
    （绑定身份/授权范围/客户上下文/规范化问题/上下文摘要/知识库版本，命中仍
    写审计并保存会话回合），默认启用。
    """
    global _semantic_cache
    if _semantic_cache is None:
        with _semantic_cache_lock:
            if _semantic_cache is None:
                _semantic_cache = SemanticCache(enabled=config.semantic_cache_enabled)
    return _semantic_cache
