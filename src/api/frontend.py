"""React 前端静态页面与 SPA fallback 路由（P1-6 / ISSUE-7）。

SPA 通配 catch-all 必须是全应用最后注册的 GET 路由（issues.md 一.2）：
Starlette 按注册顺序匹配路径，main.py 中本 router 必须在所有业务 router
之后 include。
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

# P1-6: React 前端静态文件挂载（frontend/dist 必须已构建，缺失即启动失败）
_FRONTEND_DIST = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"


def _ensure_frontend_dist() -> None:
    """fail-fast：React 构建产物缺失时启动即报错，不再静默兜底旧 HTML UI（ISSUE-7）。"""
    if not _FRONTEND_DIST.exists():
        raise RuntimeError(
            f"React frontend build not found at {_FRONTEND_DIST}; "
            "run `cd frontend && npm run build` first"
        )


def mount_frontend(app: FastAPI) -> None:
    """挂载 /assets 静态资源；必须在业务 router 注册前调用。"""
    _ensure_frontend_dist()
    app.mount("/assets", StaticFiles(directory=str(_FRONTEND_DIST / "assets")), name="assets")
    logging.getLogger("secrag.audit").info("React frontend mounted from %s", _FRONTEND_DIST)


router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def ui():
    return FileResponse(str(_FRONTEND_DIST / "index.html"))


@router.get("/admin", response_class=HTMLResponse)
async def admin_ui():
    """知识库管理后台页面（React AdminPage 路由）。"""
    return FileResponse(str(_FRONTEND_DIST / "index.html"))


@router.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)


@router.get("/.well-known/appspecific/com.chrome.devtools.json", include_in_schema=False)
async def chrome_devtools_probe():
    return {}


@router.get("/{full_path:path}", response_class=HTMLResponse)
async def spa_catch_all(full_path: str):
    """React Router catch-all——非 API 路径返回 index.html。"""
    if _FRONTEND_DIST.exists() and not full_path.startswith(
        ("v1/", "health", "metrics", "docs", "openapi.json")
    ):
        index_file = _FRONTEND_DIST / "index.html"
        if index_file.exists():
            return FileResponse(str(index_file))
    raise HTTPException(status_code=404, detail="Not Found")
