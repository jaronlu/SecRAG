import logging
import uuid
from fastapi import FastAPI, HTTPException
from langchain_core.runnables.config import RunnableConfig
from fastapi.middleware.cors import CORSMiddleware

from src.config import config
from src.rag.chain import build_rag_chain, format_docs
from src.rag.formatter import estimate_confidence, format_citations
from src.retrieval.vector_retriever import ChromaVectorRetriever
from src.schemas.constants import (
    META_SOURCE,
    ROLE_DATA_PERMISSIONS,
    RR_METADATA,
    STATE_AMBIGUITY,
    STATE_AUDIT_TRAIL,
    STATE_CITATIONS,
    STATE_CLIENT_ID,
    STATE_COMPLIANCE,
    STATE_CONFIDENCE,
    STATE_DATA_PERMISSIONS,
    STATE_DEPARTMENT,
    STATE_ENTITIES,
    STATE_FINAL_ANSWER,
    STATE_INTERMEDIATE_STEPS,
    STATE_INTENT,
    STATE_MESSAGES,
    STATE_ORIGINAL_QUERY,
    STATE_QUERY_TYPE,
    STATE_RETRIEVAL_PLAN,
    STATE_RETRIEVAL_RESULTS,
    STATE_REWRITTEN_QUERY,
    STATE_RISK_DISCLOSURE,
    STATE_TOOL_CALLS,
    STATE_USER_ID,
    STATE_USER_ROLE,
    STATE_VERIFICATION,
)
from src.schemas.request_response import (
    AssistantQARequest,
    AssistantQAResponse,
    QARequest,
    QAResponse,
)

app = FastAPI(title="券商内部投研知识平台", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 构建 RAG 链（langchain LCEL），全局复用，避免每次请求重新编译
rag_chain = build_rag_chain()

# 检索器全局复用，避免反复创建 ChromaDB 客户端
retriever = ChromaVectorRetriever(persist_directory=config.chroma.persist_directory)

# 审计日志记录器（结构化 JSON，可对接 ELK / Loki）
audit_logger = logging.getLogger("secrag.audit")


@app.post("/v1/qa", response_model=QAResponse)
async def qa(request: QARequest):
    request_id = str(uuid.uuid4())

    try:
        # 1. 检索（仅一次，结果同时用于生成和引用格式化）
        retrieval_results = retriever.retrieve(
            query=request.query,
            top_k=request.top_k,
        )

        # 2. 将检索结果格式化为 context 传入生成链，避免链内重复检索
        context = format_docs(retrieval_results)
        answer = rag_chain.invoke({
            "question": request.query,
            "context": context,
        })

        # 3. 格式化引用与置信度
        citations = format_citations(retrieval_results)
        confidence = estimate_confidence(retrieval_results)

        # 4. 审计日志
        audit_logger.info({
            "request_id": request_id,
            "query": request.query,
            "top_k": request.top_k,
            "retrieval_count": len(retrieval_results),
            "confidence": confidence,
        })
        return QAResponse(
            answer=answer,
            citations=citations,
            confidence=confidence,
            retrieval_path=[r.get(RR_METADATA, {}).get(META_SOURCE, "") for r in retrieval_results],
        )
    except Exception:
        audit_logger.exception("QA 请求处理失败: request_id=%s", request_id)
        raise HTTPException(status_code=500, detail="内部处理错误")


# ══════════════════════════════════════════════════════════════════════
# Agent 接口（impl-03 §7）
# ══════════════════════════════════════════════════════════════════════

agent_app = None  # 懒加载，首次请求时构建


def _get_agent_app():
    """懒加载 Agent Graph（避免启动时 import 链触发 ChromaDB 连接）"""
    global agent_app
    if agent_app is None:
        from src.agents.graph import build_agent_with_checkpoint

        agent_app = build_agent_with_checkpoint()
    return agent_app


@app.post("/v1/assistant/qa", response_model=AssistantQAResponse)
async def assistant_qa(request: AssistantQARequest):
    thread_id = request.thread_id or str(uuid.uuid4())

    # 基于角色计算数据权限
    data_permissions = ROLE_DATA_PERMISSIONS.get(request.user_role, ["public"])

    initial_state = {
        STATE_USER_ID: request.user_id,
        STATE_USER_ROLE: request.user_role,
        STATE_DEPARTMENT: request.department,
        STATE_DATA_PERMISSIONS: data_permissions,
        STATE_CLIENT_ID: request.client_id,
        STATE_ORIGINAL_QUERY: request.query,
        STATE_REWRITTEN_QUERY: "",
        STATE_INTENT: "",
        STATE_ENTITIES: {},
        STATE_AMBIGUITY: [],
        STATE_QUERY_TYPE: "",
        STATE_RETRIEVAL_PLAN: [],
        STATE_RETRIEVAL_RESULTS: [],
        STATE_MESSAGES: [],
        STATE_TOOL_CALLS: [],
        STATE_INTERMEDIATE_STEPS: [],
        STATE_VERIFICATION: {},
        STATE_COMPLIANCE: {},
        STATE_FINAL_ANSWER: "",
        STATE_CITATIONS: [],
        STATE_CONFIDENCE: "low",
        STATE_RISK_DISCLOSURE: "",
        STATE_AUDIT_TRAIL: {},
    }

    app = _get_agent_app()
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}, "recursion_limit": 20}
    result = app.invoke(initial_state, config=config)

    return AssistantQAResponse(
        answer=result[STATE_FINAL_ANSWER],
        citations=result[STATE_CITATIONS],
        confidence=result[STATE_CONFIDENCE],
        compliance=result[STATE_COMPLIANCE],
        audit_trail=result[STATE_AUDIT_TRAIL],
    )
