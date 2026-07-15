"""引用格式化和置信度评估

Retriever 只管检索，返回原始结果。
formatter 负责把检索结果转成：
- 带引用标注的答案结构（citations）
- 置信度评估（confidence）

下游接口和前端都消费这两个字段。
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

    # 取最高分的检索结果
    top_score = retrieval_results[0][RR_SCORE]
    if (
        # 最高分达到高置信度阈值 且 结果数量足够多
        top_score >= CONFIDENCE_HIGH_THRESHOLD
        and len(retrieval_results) >= CONFIDENCE_HIGH_MIN_RESULTS
    ):
        return CONFIDENCE_HIGH
    elif top_score >= CONFIDENCE_MEDIUM_THRESHOLD:
        # 最高分达到中置信度阈值
        return CONFIDENCE_MEDIUM
    else:
        return CONFIDENCE_LOW
