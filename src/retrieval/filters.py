"""ChromaDB where 子句构建工具"""

from typing import Dict, Optional

from src.schemas.constants import META_DOC_TYPE, META_RETRIEVAL_SOURCE


def _build_required_where(field: str, value: str, extra: Optional[Dict] = None) -> Dict:
    conditions = [{field: value}]
    if extra:
        conditions.extend([{key: item} for key, item in extra.items()])
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def build_chroma_where(doc_type: str, extra: Optional[Dict] = None) -> Optional[Dict]:
    """构建 ChromaDB where 子句：单条件直接返回，多条件用 $and 包装"""
    return _build_required_where(META_DOC_TYPE, doc_type, extra)


def build_retrieval_source_where(source: str, extra: Optional[Dict] = None) -> Dict:
    """构建强制 retrieval_source 的 ChromaDB where 子句。"""
    return _build_required_where(META_RETRIEVAL_SOURCE, source, extra)


# ──────────────────────────────────────────────
# 调用示例
# ──────────────────────────────────────────────
#
# 示例 1：只有一个条件 → 直接返回
#   build_chroma_where("pdf")
#   # 返回: {"doc_type": "pdf"}
#
# 示例 2：传入 extra → 多条件
#   build_chroma_where("pdf", {"source": "arxiv"})
#   # conditions = [{"doc_type": "pdf"}, {"source": "arxiv"}]
#   # 返回: {"$and": [{"doc_type": "pdf"}, {"source": "arxiv"}]}
#
# 设计意图：
#   - ChromaDB 的 where 参数要求单个条件是字典，多个条件必须用 $and 包一层
#   - 这个函数帮你自动判断用哪种格式，调用方不用关心底层语法
