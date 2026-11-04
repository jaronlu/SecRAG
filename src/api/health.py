"""运维观测端点：健康检查（P2-4）与 Prometheus 指标导出（P1-5）。"""

from __future__ import annotations

import asyncio
import functools
import subprocess
import time
from typing import Any

from fastapi import APIRouter
from fastapi.responses import Response

from src.utils.metrics import get_metrics

router = APIRouter()


@functools.lru_cache(maxsize=1)
def _deployment_version() -> dict[str, Any]:
    """部署代码版本标识（ENV-2：部署时效缺口）。

    GUI 实测暴露：修复提交晚于服务进程启动时，实机跑的是旧代码，
    验收结论整体失真。/health 返回 git 提交与工作区是否干净，
    供发布流程核对"部署版本 == HEAD"。版本在启动时已定型，
    结果缓存只解析一次，避免健康检查高频执行 git 子进程；
    git 不可用（如容器内无 .git）时返回 unknown，按未校验处理。
    """

    def _git_out(args: list[str]) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            )
        except Exception:
            return None
        return result.stdout

    version: dict[str, Any] = {"commit": "unknown", "dirty": False}
    head = _git_out(["rev-parse", "--short", "HEAD"])
    if head is not None:
        version["commit"] = head.strip() or "unknown"
        changed = _git_out(["status", "--porcelain"])
        version["dirty"] = bool(changed and changed.strip())
    return version


@router.get("/health")
async def health_check():
    """P2-4: 健康检查端点——检查 ChromaDB 连通性和文档计数。

    返回 200 表示服务存活；chroma 字段为 "error" 时不影响整体 200，
    避免 ChromaDB 短暂不可用导致负载均衡器摘除节点。
    """
    status = {"status": "ok", "timestamp": time.time(), "version": _deployment_version()}

    def _probe_chroma() -> dict[str, Any]:
        # ISSUE-18：Chroma 连接与 count() 是同步 IO，移入线程池避免阻塞事件循环
        from src.retrieval.vector_retriever import ChromaVectorRetriever

        engine = ChromaVectorRetriever()
        return {"status": "ok", "doc_count": engine.collection.count()}

    try:
        status["chroma"] = await asyncio.to_thread(_probe_chroma)
    except Exception as exc:
        status["chroma"] = {"status": "error", "error": str(exc)[:200]}

    # P1-5: 可观测性——健康检查中加入关键指标摘要
    try:
        status["metrics"] = get_metrics().get_summary()
    except Exception:
        pass

    return status


@router.get("/metrics")
async def prometheus_metrics():
    """P1-5: Prometheus 指标导出端点——输出标准 Prometheus 文本格式。

    可被 Prometheus 抓取，对接 Grafana 仪表盘监控。
    """
    metrics_text = get_metrics().export_prometheus()
    return Response(
        content=metrics_text,
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
