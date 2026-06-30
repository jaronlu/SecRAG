from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict] = None,
    ) -> List[Dict]:
        """检索，返回带 metadata 的 chunk 列表"""
        pass
