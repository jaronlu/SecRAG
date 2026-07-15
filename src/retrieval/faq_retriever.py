"""FAQ 检索：封装 ChromaVectorRetriever，按 doc_type 过滤"""

from typing import Dict, Optional

from src.retrieval.base import BaseRetriever
from src.retrieval.filters import build_chroma_where
from src.retrieval.vector_retriever import ChromaVectorRetriever
from src.schemas.constants import DEFAULT_TOP_K, DOC_TYPE_FAQ
from src.schemas.typed_dicts import RetrievalResult


class FAQRetriever(BaseRetriever):
    """FAQ 检索：封装 ChromaVectorRetriever，按 doc_type=faq 过滤"""

    def __init__(self, engine: Optional[BaseRetriever] = None):
        self._engine = engine or ChromaVectorRetriever()

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        filters: Optional[Dict] = None,
    ) -> list[RetrievalResult]:
        return self._engine.retrieve(
            query,
            top_k=top_k,
            filters=build_chroma_where(DOC_TYPE_FAQ, filters),
        )


# ──────────────────────────────────────────────
# 调用示例
# ──────────────────────────────────────────────
#
# 示例 1：使用默认引擎（ChromaVectorRetriever），只查 FAQ
#   faq = FAQRetriever()
#   results = faq.retrieve("怎么退款？")
#   # 内部实际调用：
#   #   ChromaVectorRetriever().retrieve("怎么退款？", filters={"doc_type": "faq"})
#
# 示例 2：同时加额外过滤条件
#   results = faq.retrieve("怎么退款？", filters={"category": "售后"})
#   # 内部实际调用：
#   #   build_chroma_where("faq", {"category": "售后"})
#   #   → {"$and": [{"doc_type": "faq"}, {"category": "售后"}]}
#
# 示例 3：传入自定义引擎（比如测试时用 Mock，或换用 BM25）
#   mock_engine = MockRetriever()
#   faq = FAQRetriever(engine=mock_engine)
#   results = faq.retrieve("怎么退款？")
#   # 底层走 mock_engine，但自动带上 doc_type=faq 过滤
