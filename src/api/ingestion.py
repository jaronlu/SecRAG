"""Technical-only document ingestion management API."""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status

from src.api.auth import AuthenticatedUser, authenticate_user
from src.ingestion.service import (
    ActiveIngestRunError,
    CategoryPreflightError,
    IngestDocumentNotFoundError,
    IngestionService,
    IngestRunNotFoundError,
    UnknownIngestionCategoryError,
    UnsafeIngestionPathError,
)
from src.schemas.constants import (
    API_ROUTE_INGESTION_CATEGORIES,
    API_ROUTE_INGESTION_CATEGORY_FILES,
    API_ROUTE_INGESTION_CHUNKS,
    API_ROUTE_INGESTION_RUN,
    API_ROUTE_INGESTION_RUN_ITEMS,
    API_ROUTE_INGESTION_RUNS,
    ROLE_TECHNICAL,
)
from src.schemas.request_response import (
    IngestionCategoriesResponse,
    IngestionCategoryResponse,
    IngestionChunkResponse,
    IngestionDocumentChunksResponse,
    IngestionFileResponse,
    IngestionFilesResponse,
    IngestionRunCreate,
    IngestionRunCreateResponse,
    IngestionRunItemResponse,
    IngestionRunItemsResponse,
    IngestionRunResponse,
    IngestionRunsResponse,
)

router = APIRouter(tags=["ingestion"])


def _get_ingestion_service() -> IngestionService:
    return IngestionService()


def require_technical_user(
    user: AuthenticatedUser = Depends(authenticate_user),
) -> AuthenticatedUser:
    if user.role != ROLE_TECHNICAL:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="technical role required")
    return user


def _translate_error(exc: Exception) -> HTTPException:
    if isinstance(exc, UnknownIngestionCategoryError):
        return HTTPException(status_code=404, detail="文档分类不存在")
    if isinstance(exc, IngestRunNotFoundError):
        return HTTPException(status_code=404, detail="入库任务不存在")
    if isinstance(exc, IngestDocumentNotFoundError):
        return HTTPException(status_code=404, detail="该文档没有可用 Chunks")
    if isinstance(exc, ActiveIngestRunError):
        return HTTPException(
            status_code=409,
            detail={"message": "已有入库任务运行", "active_run_id": exc.run_id},
        )
    if isinstance(exc, (CategoryPreflightError, UnsafeIngestionPathError, FileNotFoundError)):
        return HTTPException(status_code=422, detail=str(exc))
    raise exc


@router.get(API_ROUTE_INGESTION_CATEGORIES, response_model=IngestionCategoriesResponse)
def get_ingestion_categories(
    _: AuthenticatedUser = Depends(require_technical_user),
    service: IngestionService = Depends(_get_ingestion_service),
):
    categories, active_run_id = service.list_categories()
    return IngestionCategoriesResponse(
        categories=[IngestionCategoryResponse(**category) for category in categories],
        active_run_id=active_run_id,
    )


@router.get(API_ROUTE_INGESTION_CATEGORY_FILES, response_model=IngestionFilesResponse)
def get_ingestion_category_files(
    category_id: str,
    _: AuthenticatedUser = Depends(require_technical_user),
    service: IngestionService = Depends(_get_ingestion_service),
):
    try:
        files = service.list_category_files(category_id)
    except Exception as exc:
        raise _translate_error(exc) from exc
    return IngestionFilesResponse(
        category_id=category_id,
        files=[IngestionFileResponse(**file) for file in files],
    )


@router.get(API_ROUTE_INGESTION_CHUNKS, response_model=IngestionDocumentChunksResponse)
def get_ingestion_document_chunks(
    doc_id: str = Query(min_length=1, max_length=500),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=100),
    _: AuthenticatedUser = Depends(require_technical_user),
    service: IngestionService = Depends(_get_ingestion_service),
):
    try:
        chunks, total_chunks = service.list_document_chunks(
            doc_id,
            offset=offset,
            limit=limit,
        )
    except Exception as exc:
        raise _translate_error(exc) from exc
    return IngestionDocumentChunksResponse(
        doc_id=doc_id,
        total_chunks=total_chunks,
        offset=offset,
        limit=limit,
        chunks=[IngestionChunkResponse(**chunk) for chunk in chunks],
    )


@router.post(
    API_ROUTE_INGESTION_RUNS,
    response_model=IngestionRunCreateResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def create_ingestion_run(
    request: IngestionRunCreate,
    background_tasks: BackgroundTasks,
    user: AuthenticatedUser = Depends(require_technical_user),
    service: IngestionService = Depends(_get_ingestion_service),
):
    try:
        summary = service.create_run(request.category_id, requested_by=user.user_id)
    except Exception as exc:
        raise _translate_error(exc) from exc
    background_tasks.add_task(service.execute_run, summary["run_id"])
    return IngestionRunCreateResponse(
        run_id=summary["run_id"],
        category_id=summary["category_id"],
        status=summary["status"],
        queued_at=summary["queued_at"],
    )


@router.get(API_ROUTE_INGESTION_RUNS, response_model=IngestionRunsResponse)
def get_recent_ingestion_runs(
    limit: int = Query(default=10, ge=1, le=50),
    _: AuthenticatedUser = Depends(require_technical_user),
    service: IngestionService = Depends(_get_ingestion_service),
):
    return IngestionRunsResponse(
        runs=[IngestionRunResponse(**run) for run in service.list_recent_runs(limit)]
    )


@router.get(API_ROUTE_INGESTION_RUN, response_model=IngestionRunResponse)
def get_ingestion_run(
    run_id: str,
    _: AuthenticatedUser = Depends(require_technical_user),
    service: IngestionService = Depends(_get_ingestion_service),
):
    try:
        return IngestionRunResponse(**service.get_run(run_id))
    except Exception as exc:
        raise _translate_error(exc) from exc


@router.get(API_ROUTE_INGESTION_RUN_ITEMS, response_model=IngestionRunItemsResponse)
def get_ingestion_run_items(
    run_id: str,
    _: AuthenticatedUser = Depends(require_technical_user),
    service: IngestionService = Depends(_get_ingestion_service),
):
    try:
        items = service.list_run_items(run_id)
    except Exception as exc:
        raise _translate_error(exc) from exc
    return IngestionRunItemsResponse(
        run_id=run_id,
        items=[IngestionRunItemResponse(**item) for item in items],
    )
