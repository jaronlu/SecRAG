from abc import ABC, abstractmethod
from typing import Dict, Optional

from src.schemas.constants import DEFAULT_TOP_K
from src.schemas.typed_dicts import RetrievalResult


class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        filters: Optional[Dict] = None,
    ) -> list[RetrievalResult]:
        """检索，返回带 metadata 的 chunk 列表"""
        pass
