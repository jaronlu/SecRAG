"""引用格式化和置信度评估

⚡ 字段统一：读取 retrieval_results 和 metadata 时使用 src/schemas/constants 中的常量。
输出 dict 的字段名对应 SCHEMA-REFERENCE §3.1 Citation 模型。
"""

from src.schemas.constants import (
    # 枚举 (§2.3, §2.4)
    CONFIDENCE_HIGH,
    CONFIDENCE_HIGH_MIN_RESULTS,
    # 阈值
    CONFIDENCE_HIGH_THRESHOLD,
    CONFIDENCE_LOW,
    CONFIDENCE_MEDIUM,
    CONFIDENCE_MEDIUM_THRESHOLD,
    # retrieval_results 键名 (§1.2)
    RR_SCORE,
)
from src.schemas.typed_dicts import CitationDict, RetrievalResult


def format_citations(retrieval_results: list[RetrievalResult]) -> list[CitationDict]:
    """Generate citations for the development-only basic RAG route."""
    from src.utils.verifier import CitationExtractor

    return CitationExtractor().extract(retrieval_results, query="")


def estimate_confidence(retrieval_results: list[RetrievalResult]) -> str:
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
