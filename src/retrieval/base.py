from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from src.schemas.constants import DEFAULT_TOP_K


class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        filters: Optional[Dict] = None,
    ) -> List[Dict]:
        """检索，返回带 metadata 的 chunk 列表"""
        pass
