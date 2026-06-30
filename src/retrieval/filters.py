"""ChromaDB where 子句构建工具"""

from typing import Dict, Optional

from src.schemas.constants import META_DOC_TYPE


def build_chroma_where(doc_type: str, extra: Optional[Dict] = None) -> Optional[Dict]:
    """构建 ChromaDB where 子句：单条件直接返回，多条件用 $and 包装"""
    conditions = [{META_DOC_TYPE: doc_type}]
    if extra:
        conditions.extend([{k: v} for k, v in extra.items()])
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}
