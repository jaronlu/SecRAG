"""Retriever 抽象层。
Retriever 管的就是一件事：给定 query，返回结果列表。它不关心你是去查 ChromaDB、BM25 索引、ES，还是去调用某个远程搜索 API。

每个子类各自分析自己的后端：
- ChromaDB 实现：算 embedding、拼 where 条件、调 collection.query()
- BM25 实现：分词、算 TF-IDF、倒排表打分
- 混合实现：并行调两个后端、归一化分数、重排序

上游代码永远只看到同一个接口：
results = retriever.retrieve(query, top_k=5)

这就是典型的策略模式：接口统一，算法/后端可替换。
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional

from src.schemas.constants import DEFAULT_TOP_K
from src.schemas.typed_dicts import RetrievalResult


class BaseRetriever(ABC):  #  Abstract Base Class（抽象基类）
    @abstractmethod  # 这个装饰器：注释标记方法子类必须自己实现，没有标记就是普通方法子类可以直接调用
    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        filters: Optional[Dict] = None,  # 可选的过滤条件，不传就是None
    ) -> list[RetrievalResult]:  # 返回值是 RetrievalResult 的列表
        """检索，返回带 metadata 的 chunk 列表"""
        pass
