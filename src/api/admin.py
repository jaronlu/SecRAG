"""Admin 管理端点（P2-7/P2-9/P1-4）：查询统计、语义缓存运维、知识库文档管理。"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from src.api.auth import AuthenticatedUser, authenticate_user
from src.api.deps import _get_conversation_store
from src.utils.semantic_cache import get_semantic_cache

router = APIRouter()


@router.get("/v1/admin/stats/queries")
async def query_stats(
    days: int = 7,
    user: AuthenticatedUser = Depends(authenticate_user),
):
    """P2-9: 查询统计端点——返回查询量、命中率、无引用查询示例。

    仅 admin 角色可访问。
    """
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="仅管理员可访问统计数据")
    return _get_conversation_store().query_stats(days=days)


@router.get("/v1/admin/cache/stats")
async def semantic_cache_stats(
    user: AuthenticatedUser = Depends(authenticate_user),
):
    """P1-4: 语义缓存统计端点——返回缓存条目数、命中率、阈值等。

    仅 admin/technical 角色可访问。
    """
    if user.role not in ("admin", "technical"):
        raise HTTPException(status_code=403, detail="仅管理员可访问缓存统计")
    return get_semantic_cache().get_stats()


@router.post("/v1/admin/cache/clear")
async def semantic_cache_clear(
    clear_expired_only: bool = False,
    user: AuthenticatedUser = Depends(authenticate_user),
):
    """P1-4: 语义缓存清理端点——清理过期缓存或全部清空。

    仅 admin/technical 角色可访问。
    """
    if user.role not in ("admin", "technical"):
        raise HTTPException(status_code=403, detail="仅管理员可清理缓存")
    cache = get_semantic_cache()
    if clear_expired_only:
        cleared = cache.clear_expired()
        return {"cleared": cleared, "mode": "expired_only"}
    cleared = cache.clear_all()
    return {"cleared": cleared, "mode": "all"}


@router.delete("/v1/admin/documents")
async def delete_document(
    source: str,
    user: AuthenticatedUser = Depends(authenticate_user),
):
    """P2-7: 文档删除端点——按 source 路径删除该文档的所有 chunk。

    仅 admin 角色可访问。用于文档版本管理：删除旧版本后重新入库。
    """
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="仅管理员可删除文档")
    from src.retrieval.vector_retriever import ChromaVectorRetriever

    engine = ChromaVectorRetriever()
    deleted = engine.delete_by_source(source)
    return {"source": source, "deleted_chunks": deleted}


@router.get("/v1/admin/documents")
async def list_documents(
    doc_type: str | None = None,
    limit: int = 100,
    offset: int = 0,
    user: AuthenticatedUser = Depends(authenticate_user),
):
    """P2: 知识库文档列表——列出所有已入库文档（按 source 分组）。

    仅 admin/technical 角色可访问。支持按 doc_type 筛选、分页。
    """
    if user.role not in ("admin", "technical"):
        raise HTTPException(status_code=403, detail="仅管理员可查看文档列表")
    from src.utils.knowledge_base import get_kb_manager

    return get_kb_manager().list_documents(doc_type=doc_type, limit=limit, offset=offset)


@router.get("/v1/admin/documents/stats")
async def knowledge_base_stats(
    user: AuthenticatedUser = Depends(authenticate_user),
):
    """P2: 知识库统计——返回文档数、chunk 数、按类型分布。

    仅 admin/technical 角色可访问。
    """
    if user.role not in ("admin", "technical"):
        raise HTTPException(status_code=403, detail="仅管理员可查看统计")
    from src.utils.knowledge_base import get_kb_manager

    return get_kb_manager().get_stats()


@router.get("/v1/admin/documents/chunks")
async def get_document_chunks(
    source: str,
    limit: int = 50,
    offset: int = 0,
    user: AuthenticatedUser = Depends(authenticate_user),
):
    """P2: 文档 chunk 详情——查看某文档的所有 chunk 内容和元数据。

    仅 admin/technical 角色可访问。用于排查检索质量问题。
    """
    if user.role not in ("admin", "technical"):
        raise HTTPException(status_code=403, detail="仅管理员可查看 chunk 详情")
    from src.utils.knowledge_base import get_kb_manager

    return get_kb_manager().get_document_chunks(source=source, limit=limit, offset=offset)


@router.get("/v1/admin/documents/search")
async def search_knowledge_base(
    query: str,
    top_k: int = 5,
    user: AuthenticatedUser = Depends(authenticate_user),
):
    """P2: 知识库语义搜索——直接在向量库中搜索，用于预览检索效果。

    仅 admin/technical 角色可访问。不经过 Agent 流程，直接返回检索结果。
    issues.md 一.3：前端以 GET 查询参数调用，这里注册为 GET 保持契约一致。
    """
    if user.role not in ("admin", "technical"):
        raise HTTPException(status_code=403, detail="仅管理员可搜索知识库")
    from src.utils.knowledge_base import get_kb_manager

    return {"query": query, "results": get_kb_manager().search_documents(query=query, top_k=top_k)}
