"""FAQ 检索：封装 ChromaVectorRetriever，按 doc_type 过滤"""

from typing import Dict, List, Optional

from src.retrieval.base import BaseRetriever
from src.retrieval.vector_retriever import ChromaVectorRetriever
from src.schemas.constants import DEFAULT_TOP_K, DOC_TYPE_FAQ, META_DOC_TYPE


class FAQRetriever(BaseRetriever):
    """FAQ 检索：封装 ChromaVectorRetriever，按 doc_type=faq 过滤。
    FAQ 通常较短且答案集中，默认 top_k 与通用检索一致；
    需要更少结果时由调用方（如 faq_search tool）传入更小的 top_k。
    """

    def __init__(self):
        self._engine = ChromaVectorRetriever()

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        filters: Optional[Dict] = None,
    ) -> List[Dict]:
        effective_filters = {META_DOC_TYPE: DOC_TYPE_FAQ}
        if filters:
            effective_filters.update(filters)
        return self._engine.retrieve(query, top_k=top_k, filters=effective_filters)
