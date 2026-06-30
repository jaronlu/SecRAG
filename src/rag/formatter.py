"""引用格式化和置信度评估

⚡ 字段统一：读取 retrieval_results 和 metadata 时使用 src/schemas/constants 中的常量。
输出 dict 的字段名对应 SCHEMA-REFERENCE §3.1 Citation 模型。
"""

from typing import Dict, List

from src.schemas.constants import (
    # retrieval_results 键名 (§1.2)
    RR_CONTENT,
    RR_METADATA,
    RR_SCORE,
    # metadata 键名 (§1.1)
    META_TITLE,
    META_SOURCE,
    META_DOC_TYPE,
    META_CHUNK_ID,
    META_PAGE_NUMBER,
    META_PERMISSION_LEVEL,
    # 枚举 (§2.3, §2.4)
    CONFIDENCE_HIGH,
    CONFIDENCE_MEDIUM,
    CONFIDENCE_LOW,
    PERMISSION_PUBLIC,
    # 阈值
    CONFIDENCE_HIGH_THRESHOLD,
    CONFIDENCE_HIGH_MIN_RESULTS,
    CONFIDENCE_MEDIUM_THRESHOLD,
)


def format_citations(retrieval_results: List[Dict]) -> List[Dict]:
    """生成引用列表

    输出 dict 字段对应 SCHEMA-REFERENCE §3.1 Citation 模型。
    """
    citations = []
    for i, doc in enumerate(retrieval_results, 1):
        meta = doc.get(RR_METADATA, {})
        content = doc.get(RR_CONTENT, "")
        citations.append({
            "citation_id": f"cite_{i:03d}",
            "doc_title": meta.get(META_TITLE, "未知"),
            "source": meta.get(META_SOURCE, ""),
            "doc_type": meta.get(META_DOC_TYPE, ""),
            "chunk_id": meta.get(META_CHUNK_ID, ""),
            "quote": content[:200] + "..." if len(content) > 200 else content,
            "relevance_score": round(doc.get(RR_SCORE, 0), 4),
            "page_number": meta.get(META_PAGE_NUMBER),
            "permission_level": meta.get(META_PERMISSION_LEVEL, PERMISSION_PUBLIC),
        })
    return citations


def estimate_confidence(retrieval_results: List[Dict]) -> str:
    """
    简单置信度估计（规则版）
    后续可替换为 LLM judge
    """
    if not retrieval_results:
        return CONFIDENCE_LOW

    top_score = retrieval_results[0][RR_SCORE]
    if top_score >= CONFIDENCE_HIGH_THRESHOLD and len(retrieval_results) >= CONFIDENCE_HIGH_MIN_RESULTS:
        return CONFIDENCE_HIGH
    elif top_score >= CONFIDENCE_MEDIUM_THRESHOLD:
        return CONFIDENCE_MEDIUM
    else:
        return CONFIDENCE_LOW
