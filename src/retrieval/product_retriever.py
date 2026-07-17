"""产品知识库检索：封装 ChromaVectorRetriever，按 doc_type 过滤"""

from typing import Dict, Optional

from src.retrieval.base import BaseRetriever
from src.retrieval.filters import build_retrieval_source_where
from src.retrieval.vector_retriever import ChromaVectorRetriever
from src.schemas.constants import DEFAULT_TOP_K, SOURCE_PRODUCT
from src.schemas.typed_dicts import RetrievalResult


class ProductRetriever(BaseRetriever):
    """产品知识库检索：按 retrieval_source=product_search 过滤。"""

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
            filters=build_retrieval_source_where(SOURCE_PRODUCT, filters),
        )
