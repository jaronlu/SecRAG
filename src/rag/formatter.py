"""引用格式化和置信度评估

⚡ 字段统一：读取 retrieval_results 和 metadata 时使用 src/schemas/constants 中的常量。
输出 dict 的字段名对应 SCHEMA-REFERENCE §3.1 Citation 模型。
"""

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

from src.schemas.constants import (
    # 枚举 (§2.3, §2.4)
    CONFIDENCE_HIGH,
    CONFIDENCE_HIGH_MIN_RESULTS,
    # 阈值
    CONFIDENCE_HIGH_THRESHOLD,
    CONFIDENCE_LOW,
    CONFIDENCE_MEDIUM,
    CONFIDENCE_MEDIUM_THRESHOLD,
    META_CHUNK_ID,
    META_DOC_TYPE,
    META_PAGE_NUMBER,
    META_PERMISSION_LEVEL,
    META_SOURCE,
    # metadata 键名 (§1.1)
    META_TITLE,
    PERMISSION_PUBLIC,
    # retrieval_results 键名 (§1.2)
    RR_CONTENT,
    RR_METADATA,
    RR_SCORE,
)


@dataclass
class Citation:
    """引用标注 — 权威定义，字段对应 SCHEMA-REFERENCE §3.1"""

    citation_id: str
    doc_title: str
    source: str
    doc_type: str
    chunk_id: str
    quote: str
    relevance_score: float
    permission_level: str
    page_number: Optional[int] = None


def format_citations(retrieval_results: List[Dict]) -> List[Dict]:
    """生成引用列表

    输出 dict 字段对应 SCHEMA-REFERENCE §3.1 Citation 模型。
    """
    citations = []
    for i, doc in enumerate(retrieval_results, 1):
        meta = doc.get(RR_METADATA, {})
        content = doc.get(RR_CONTENT, "")
        citation = Citation(
            citation_id=f"cite_{i:03d}",
            doc_title=meta.get(META_TITLE, "未知"),
            source=meta.get(META_SOURCE, ""),
            doc_type=meta.get(META_DOC_TYPE, ""),
            chunk_id=meta.get(META_CHUNK_ID, ""),
            quote=content[:200] + "..." if len(content) > 200 else content,
            relevance_score=round(doc.get(RR_SCORE, 0), 4),
            page_number=meta.get(META_PAGE_NUMBER),
            permission_level=meta.get(META_PERMISSION_LEVEL, PERMISSION_PUBLIC),
        )
        citations.append(asdict(citation))
    return citations


def estimate_confidence(retrieval_results: List[Dict]) -> str:
    """
    简单置信度估计（规则版）
    后续可替换为 LLM judge
    """
    if not retrieval_results:
        return CONFIDENCE_LOW

    top_score = retrieval_results[0][RR_SCORE]
    if (
        top_score >= CONFIDENCE_HIGH_THRESHOLD
        and len(retrieval_results) >= CONFIDENCE_HIGH_MIN_RESULTS
    ):
        return CONFIDENCE_HIGH
    elif top_score >= CONFIDENCE_MEDIUM_THRESHOLD:
        return CONFIDENCE_MEDIUM
    else:
        return CONFIDENCE_LOW
