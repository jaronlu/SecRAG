"""产品知识库检索：封装 ChromaVectorRetriever，按 doc_type 过滤"""

from typing import Dict, List, Optional

from src.retrieval.base import BaseRetriever
from src.retrieval.filters import build_chroma_where
from src.retrieval.vector_retriever import ChromaVectorRetriever
from src.schemas.constants import DEFAULT_TOP_K, DOC_TYPE_PRODUCT


class ProductRetriever(BaseRetriever):
    """产品知识库检索：封装 ChromaVectorRetriever，按 doc_type=product 过滤"""

    def __init__(self, engine: Optional[BaseRetriever] = None):
        self._engine = engine or ChromaVectorRetriever()

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        filters: Optional[Dict] = None,
    ) -> List[Dict]:
        return self._engine.retrieve(
            query,
            top_k=top_k,
            filters=build_chroma_where(DOC_TYPE_PRODUCT, filters),
        )
