"""运维观测端点：健康检查（P2-4）与 Prometheus 指标导出（P1-5）。"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from fastapi import APIRouter
from fastapi.responses import Response

from src.utils.metrics import get_metrics

router = APIRouter()


@router.get("/health")
async def health_check():
    """P2-4: 健康检查端点——检查 ChromaDB 连通性和文档计数。

    返回 200 表示服务存活；chroma 字段为 "error" 时不影响整体 200，
    避免 ChromaDB 短暂不可用导致负载均衡器摘除节点。
    """
    status = {"status": "ok", "timestamp": time.time()}

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
