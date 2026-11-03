"""FastAPI 应用组装：lifespan 预热、中间件与各路由域 router 注册。

路由实现按域拆分：ingestion.py / admin.py / conversation.py / qa.py /
profile.py / health.py / frontend.py；跨域共享的懒加载单例在 deps.py。
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.admin import router as admin_router
from src.api.conversation import router as conversation_router
from src.api.deps import _get_agent_app
from src.api.frontend import mount_frontend, router as frontend_router
from src.api.health import router as health_router
from src.api.ingestion import router as ingestion_router
from src.api.profile import router as profile_router
from src.api.qa import router as qa_router
from src.config import config

# 追踪日志记录器（结构化 JSON，可对接 ELK / Loki）
warmup_logger = logging.getLogger("secrag.warmup")


def _warmup_retrieval_stack() -> None:
    """启动预热（ISSUE-16）：图编译、向量引擎、embedding 模型、BM25 全索引。

    冷启动开销（torch import、jieba 全量分词 10-30s、embedding 权重加载）
    此前全部落在重启后的首个请求上；预热在 lifespan 启动时的后台线程执行。
    每步独立容错：单步失败只记日志，不影响服务可用性。
    """
    started = time.perf_counter()

    def _step(name: str, fn) -> None:
        step_started = time.perf_counter()
        try:
            fn()
        except Exception as exc:
            warmup_logger.warning("warmup %s failed: %s: %s", name, type(exc).__name__, exc)
            return
        warmup_logger.info(
            "warmup %s ok in %.1f ms", name, (time.perf_counter() - step_started) * 1000
        )

    _step("agent_graph", lambda: _get_agent_app())

    def _warm_vector_and_bm25() -> None:
        from src.retrieval.bm25_retriever import BM25Retriever
        from src.retrieval.vector_retriever import ChromaVectorRetriever

        engine = ChromaVectorRetriever()
        BM25Retriever(engine).warmup()

    _step("vector_and_bm25", _warm_vector_and_bm25)

    def _warm_embedding_model() -> None:
        from src.ingestion.embedder import get_embedding_model

        get_embedding_model(config.embedding.model)

    _step("embedding_model", _warm_embedding_model)

    def _warm_reranker() -> None:
        from src.tools.rerank import RerankService, reranker_available

        if not reranker_available():
            return
        # 与 grade_and_filter 相同的调用面：预热一次真实 forward
        RerankService().rerank("预热", [{"content": "预热", "score": 0.0}], top_k=1)

    _step("reranker", _warm_reranker)

    warmup_logger.info("warmup finished in %.1f ms", (time.perf_counter() - started) * 1000)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """启动时后台预热检索栈（ISSUE-16），关闭时放弃残留预热任务。"""
    warmup_task = asyncio.create_task(asyncio.to_thread(_warmup_retrieval_stack))
    yield
    warmup_task.cancel()


app = FastAPI(title="机构内部投研知识平台", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

mount_frontend(app)

# 注册顺序契约（issues.md 一.2）：业务 router 在前；frontend_router 含 SPA
# 通配 catch-all，必须最后 include，否则会截走 /health、/metrics 等接口
app.include_router(ingestion_router)
app.include_router(health_router)
app.include_router(admin_router)
app.include_router(conversation_router)
app.include_router(qa_router)
app.include_router(profile_router)
app.include_router(frontend_router)
