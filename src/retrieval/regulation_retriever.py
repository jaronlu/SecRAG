"""法规库检索：封装 ChromaVectorRetriever，按 doc_type 过滤"""

from typing import Dict, List, Optional

from src.retrieval.base import BaseRetriever
from src.retrieval.vector_retriever import ChromaVectorRetriever
from src.schemas.constants import DEFAULT_TOP_K, DOC_TYPE_REGULATION, META_DOC_TYPE


class RegulationRetriever(BaseRetriever):
    """法规库检索：封装 ChromaVectorRetriever，按 doc_type=regulation 过滤"""

    def __init__(self):
        self._engine = ChromaVectorRetriever()

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        filters: Optional[Dict] = None,
    ) -> List[Dict]:
        effective_filters = {META_DOC_TYPE: DOC_TYPE_REGULATION}
        if filters:
            effective_filters.update(filters)
        return self._engine.retrieve(query, top_k=top_k, filters=effective_filters)
