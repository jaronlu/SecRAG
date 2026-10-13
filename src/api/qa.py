"""Agent 问答端点（impl-03 §7 / P2-1）：同步 QA 与 SSE 流式 QA。"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, cast

import httpx
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.runnables.config import RunnableConfig
from openai import APIConnectionError, APIStatusError, APITimeoutError

from src.agents.state import AssistantState
from src.api.auth import (
    AuthenticatedUser,
    authenticate_user,
    build_assistant_initial_state,
)
from src.api.deps import _conversation_http_error, _get_agent_app, _get_conversation_store
from src.config import config
from src.schemas.constants import (
    AGENT_RECURSION_LIMIT,
    API_ROUTE_ASSISTANT_QA,
    API_ROUTE_ASSISTANT_QA_STREAM,
    AUDIT_REQUEST_ID,
    ROLE_DATA_PERMISSIONS,
    STATE_AUDIT_TRAIL,
    STATE_CITATIONS,
    STATE_CLIENT_ID,
    STATE_COMPLIANCE,
    STATE_CONFIDENCE,
    STATE_ENTITIES,
    STATE_FINAL_ANSWER,
    STATE_ORIGINAL_QUERY,
    STATE_RESOLVED_QUERY,
    STATE_TERMINAL,
    STATE_THREAD_ID,
    STATE_TURN_ID,
    STATE_USER_ID,
    STATE_USER_ROLE,
    STATE_VERIFICATION,
)
from src.schemas.request_response import AssistantQARequest, AssistantQAResponse
from src.schemas.models import AuditEntry
from src.schemas.typed_dicts import AnswerOutcome, AuditQuery, AuditReasoning, AuditResponse, AuditRetrieval
from src.utils.rate_limit import check_rate_limit, get_rate_limit_key
from src.utils.semantic_cache import CacheBinding, build_cache_binding, get_semantic_cache
from src.utils.langfuse_adapter import (
    get_langfuse,
    reset_current_trace,
    set_current_trace,
)
from src.utils.metrics import get_metrics

# 追踪日志记录器（结构化 JSON，可对接 ELK / Loki）
audit_logger = logging.getLogger("secrag.audit")

router = APIRouter()


def _is_provider_unavailable(exc: Exception) -> bool:
    if isinstance(exc, (httpx.TimeoutException, httpx.TransportError)):
        return True

    if isinstance(exc, (APIConnectionError, APITimeoutError)):
        return True

    if isinstance(exc, APIStatusError):
        status_code = getattr(exc, "status_code", None)
        return status_code in {401, 403, 408, 409, 429} or (
            status_code is not None and status_code >= 500
        )

    return False


def _get_cache_hit_audit_store():
    """审计存储懒加载（P1-2 命中路径持久化审计事件），与 audit_log 节点同库。"""
    from src.agents.nodes import _get_audit_store

    return _get_audit_store()


def _build_request_cache_binding(
    request: AssistantQARequest,
    user: AuthenticatedUser,
    thread_id: str,
) -> CacheBinding:
    """构造请求级缓存绑定（ISSUE-26）。

    上下文摘要在图执行前取一次并贯穿 lookup/store：图执行会把本轮回合写入
    会话，执行后再取摘要会包含本轮实体，使 store 的绑定与 lookup 不一致而
    永不命中。会话读不到摘要时抛错（fail closed），不放宽绑定维度。
    """
    _, summary = _get_conversation_store().load_context(
        thread_id=thread_id,
        user_id=user.user_id,
    )
    return build_cache_binding(
        query=request.query,
        role=user.role,
        user_id=user.user_id,
        client_id=request.client_id or "",
        data_permissions=ROLE_DATA_PERMISSIONS.get(user.role, []),
        conversation_summary=summary,
    )


def _persist_cache_hit_turn(
    request: AssistantQARequest,
    user: AuthenticatedUser,
    *,
    thread_id: str,
    turn_id: str,
    request_id: str,
    cache_hit: dict,
) -> None:
    """命中路径保存会话回合并标记 outbox（ISSUE-26 启用条件）。

    命中跳过了图执行，persist_conversation_turn 与 audit_log 都不会运行；
    不补写会话会让多轮上下文在命中后断裂，不标记 outbox 会留下悬挂事件。
    """
    state = cast(
        AssistantState,
        {
            STATE_THREAD_ID: thread_id,
            STATE_TURN_ID: turn_id,
            STATE_USER_ID: user.user_id,
            STATE_USER_ROLE: user.role,
            STATE_CLIENT_ID: request.client_id,
            STATE_ORIGINAL_QUERY: request.query,
            STATE_RESOLVED_QUERY: request.query,
            STATE_FINAL_ANSWER: cache_hit["answer"],
            STATE_CITATIONS: cache_hit["citations"],
            STATE_ENTITIES: {},
            STATE_AUDIT_TRAIL: {AUDIT_REQUEST_ID: request_id},
        },
    )
    store = _get_conversation_store()
    store.insert_turn(state)
    if store.get_outbox_status(request_id) is not None:
        store.mark_outbox_processed(request_id)


def _persist_cache_hit_audit_event(
    user: AuthenticatedUser,
    request: AssistantQARequest,
    start_time: float,
    cache_hit: dict,
    request_id: str | None = None,
) -> None:
    """P1-2: 缓存命中跳过了图执行（audit_log 节点不会运行），在此补一条
    持久化审计事件；写入失败时复用 audit 节点的 outbox 机制落本地待重试。
    """
    import dataclasses

    from src.agents.nodes import _write_audit_outbox

    entry = AuditEntry(
        request_id=request_id or str(uuid.uuid4()),
        timestamp=datetime.now(timezone.utc).isoformat(),
        user_id=user.user_id,
        user_role=user.role,
        department=user.department,
        query=AuditQuery(original=request.query),
        retrieval=AuditRetrieval(total_chunks=0, filtered_chunks=0),
        reasoning=AuditReasoning(
            iterations=0,
            duration_ms=0.0,
            execution_path=["semantic_cache_hit"],
        ),
        verification=cache_hit["verification"],
        compliance=cache_hit["compliance"],
        response=AuditResponse(
            citations=cache_hit["citations"],
            confidence=cache_hit["confidence"],
        ),
        total_duration_ms=max((time.time() - start_time) * 1000, 0.0),
    )
    try:
        _get_cache_hit_audit_store().insert(entry)
    except Exception as exc:
        _write_audit_outbox(dataclasses.asdict(entry), str(exc))


def _qa_response_from_outcome(
    outcome: AnswerOutcome, thread_id: str, turn_id: str
) -> AssistantQAResponse:
    """统一终态出口：普通执行与缓存命中都从这里构建对外响应，
    保证两种入口返回同一种结果对象；命中/缓存相似度只进审计与指标，不进响应体。
    """
    return AssistantQAResponse(
        thread_id=thread_id,
        turn_id=turn_id,
        answer=outcome["answer"],
        citations=outcome["citations"],
        confidence=outcome["confidence"],
        compliance=outcome["compliance"],
    )


@router.post(API_ROUTE_ASSISTANT_QA, response_model=AssistantQAResponse)
async def assistant_qa(
    request: AssistantQARequest,
    user: AuthenticatedUser = Depends(authenticate_user),
):
    # P1-5: 可观测性——记录请求开始时间和活跃请求数
    start_time = time.time()
    metrics = get_metrics()
    metrics.active_requests.inc()

    def _record_metrics(status: str, is_cached: bool = False):
        """记录查询指标并减少活跃请求数。"""
        duration = time.time() - start_time
        metrics.record_query(role=user.role, status=status, duration=duration, cached=is_cached)
        metrics.active_requests.dec()

    # P2-2: 限流——按 user_id 滑动窗口，每分钟 30 次
    rate_key = get_rate_limit_key(user_id=user.user_id)
    allowed, _ = check_rate_limit(rate_key)
    if not allowed:
        _record_metrics("rate_limited")
        raise HTTPException(
            status_code=429,
            detail="请求过于频繁，请稍后再试（每分钟最多 30 次）。",
            headers={"Retry-After": "60"},
        )
    try:
        thread = await asyncio.to_thread(
            _get_conversation_store().ensure_thread_for_qa,
            thread_id=request.thread_id,
            user_id=user.user_id,
            user_role=user.role,
            client_id=request.client_id,
            title=request.query[:100],
        )
    except Exception as exc:
        raise _conversation_http_error(exc) from exc
    # ensure_thread_for_qa 返回的 thread 保证含 thread_id（查找键或新建时赋值）
    thread_id = thread["thread_id"]
    turn_id = str(uuid.uuid4())
    # request_id 显式生成后传入初始 state：Langfuse 根 trace 与 SQLite 审计
    # （STATE_AUDIT_TRAIL.request_id）共用同一 id；trace metadata 只含
    # request_id/thread_id，不携带任何用户身份信息
    request_id = str(uuid.uuid4())
    langfuse = get_langfuse()
    trace = langfuse.start_request_trace(request_id, thread_id)
    trace_token = set_current_trace(trace)
    trace_status = "ok"
    error_type: str | None = None
    try:
        initial_state = build_assistant_initial_state(
            request,
            user,
            thread_id=thread_id,
            turn_id=turn_id,
            turn_index=thread.get("turn_count", 0),
            request_id=request_id,
        )

        agent = _get_agent_app()
        runnable_config: RunnableConfig = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": AGENT_RECURSION_LIMIT,
        }
        # Langfuse：绑定根 trace 的 callback 经 RunnableConfig 传入 LangGraph，
        # 覆盖节点内 llm.invoke / 工具调用；未启用或未采样时为 None，零感知
        callback_handler = langfuse.get_callback_handler(trace)
        if callback_handler is not None:
            runnable_config["callbacks"] = [callback_handler]

        # P1-4: 语义缓存——查询前先查缓存，命中则直接返回。
        # 缓存查询涉及 embedding 计算与全表扫描，放到线程池执行，
        # 避免阻塞事件循环（issues.md 二.2）
        # ISSUE-26：绑定（身份/授权范围/客户/规范化问题/上下文摘要/知识库版本）
        # 在图执行前构造一次，lookup 与 store 复用同一绑定
        cache = get_semantic_cache()
        cache_binding = await asyncio.to_thread(
            _build_request_cache_binding, request, user, thread_id
        )
        cache_hit = await asyncio.to_thread(
            cache.lookup, request.query, user.role, binding=cache_binding
        )
        if cache_hit:
            trace.update({"cache_hit": True})
            audit_logger.info(
                "Semantic cache hit: thread_id=%s similarity=%.4f",
                thread_id, cache_hit["similarity"],
            )
            # P1-2: 命中路径补持久化审计事件（非阻塞，失败走 outbox）
            await asyncio.to_thread(
                _persist_cache_hit_audit_event,
                user,
                request,
                start_time,
                cache_hit,
                request_id,
            )
            # ISSUE-26: 命中路径仍须保存会话回合，多轮上下文不因缓存断裂
            await asyncio.to_thread(
                _persist_cache_hit_turn,
                request,
                user,
                thread_id=thread_id,
                turn_id=turn_id,
                request_id=request_id,
                cache_hit=cache_hit,
            )
            _record_metrics("success", is_cached=True)
            # P1-1: 返回 store 时保存的终态合规快照，不再硬编码 passed=True
            outcome = AnswerOutcome(
                answer=cache_hit["answer"],
                citations=cache_hit["citations"],
                confidence=cache_hit["confidence"],
                compliance=cache_hit["compliance"],
            )
            return _qa_response_from_outcome(outcome, thread_id, turn_id)

        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(agent.invoke, initial_state, runnable_config),
                timeout=config.api_request_timeout_seconds,
            )
        except asyncio.TimeoutError:
            trace_status = "error"
            error_type = "timeout"
            _record_metrics("timeout")
            audit_logger.warning(
                "Assistant QA timed out after %.1fs: thread_id=%s",
                config.api_request_timeout_seconds,
                thread_id,
            )
            raise HTTPException(
                status_code=504,
                detail=f"请求处理超时（{config.api_request_timeout_seconds:.0f}s），请简化问题或稍后重试。",
            )
        except Exception as exc:
            trace_status = "error"
            if _is_provider_unavailable(exc):
                error_type = "provider_unavailable"
                _record_metrics("provider_unavailable")
                audit_logger.warning(
                    "Assistant provider unavailable: thread_id=%s error=%s",
                    thread_id,
                    exc.__class__.__name__,
                )
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "LLM provider unavailable. Check OPENAI_API_BASE, OPENAI_API_KEY, "
                        "network/proxy settings, or start a local Ollama service and set "
                        "LLM_PROVIDER=ollama."
                    ),
                ) from exc

            error_type = type(exc).__name__
            audit_logger.exception("Assistant QA failed: thread_id=%s", thread_id)
            _record_metrics("error")
            raise HTTPException(status_code=500, detail="内部处理错误") from exc

        # P1-4: 语义缓存——仅缓存验证与合规均通过的成功终态，避免把拒答/拦截结果
        # 以"合规通过"语义缓存后再次返回（issues.md 一.1）
        answer = result.get(STATE_FINAL_ANSWER, "")
        compliance = result.get(STATE_COMPLIANCE, {})
        verification = result.get(STATE_VERIFICATION, {})
        if (
            answer
            and len(answer) > 10
            and compliance.get("passed", False)
            and verification.get("passed", False)
        ):
            await asyncio.to_thread(
                cache.store,
                query=request.query,
                answer=answer,
                citations=result.get(STATE_CITATIONS, []),
                confidence=result.get(STATE_CONFIDENCE, ""),
                role=user.role,
                compliance=compliance,
                verification=verification,
                # ISSUE-26：与 lookup 使用同一绑定，保证同请求可命中
                binding=cache_binding,
            )

        _record_metrics("success")
        outcome = AnswerOutcome(
            answer=answer,
            citations=result[STATE_CITATIONS],
            confidence=result[STATE_CONFIDENCE],
            compliance=result[STATE_COMPLIANCE],
        )
        return _qa_response_from_outcome(
            outcome,
            result.get(STATE_THREAD_ID, thread_id),
            result.get(STATE_TURN_ID, turn_id),
        )
    finally:
        reset_current_trace(trace_token)
        # 504 路径：to_thread 中的图线程不可强制取消，但 STATE_REQUEST_DEADLINE
        # 协同取消检查点（call_reason_model / 工具执行 / 合并规划节点）保证
        # 超时后不再发起新的 LLM 轮次，在有限步内收敛；finish 只结束根 span、
        # 不等待子 span（_RequestTrace 晚到事件容忍）
        trace.finish(status=trace_status, error_type=error_type)


@router.post(API_ROUTE_ASSISTANT_QA_STREAM)
async def assistant_qa_stream(
    request: AssistantQARequest,
    user: AuthenticatedUser = Depends(authenticate_user),
):
    """P2-1: 流式输出端点——SSE 逐事件返回 Agent 执行进度和最终回答。

    事件协议（issues.md 一.3：event 名与 JSON 内 type 字段一致，前端按同一契约解析）：
    - event: progress,     data: {"type": "progress", "node": "...", "status": "done"}
    - event: answer_delta, data: {"type": "answer_delta", "delta": "..."}
    - event: answer,       data: {"type": "answer", "answer": "...", "citations": [...],
                           "confidence": "...", "thread_id": "...", "turn_id": "..."}
    - event: error,        data: {"type": "error", "detail": "..."}
    - event: done,        data: {"type": "done"}

    answer_delta 携带 reason 节点 LLM 的 token 级增量（ISSUE-9），先于 answer 终态
    事件流出；answer 事件仍承载完整终态文本，前端以其为准覆盖已流出的增量。
    answer 事件来自统一终态：compose（正常回答/验证失败/合规拦截）、
    clarify（澄清）、permission_denied_response（权限拒绝）——
    后两者不经过 compose，直接产出 final_answer。
    """
    # P2-2: 限流
    rate_key = get_rate_limit_key(user_id=user.user_id)
    allowed, _ = check_rate_limit(rate_key)
    if not allowed:
        return StreamingResponse(
            iter([
                "event: error\n"
                f"data: {json.dumps({'type': 'error', 'detail': '请求过于频繁'})}\n\n"
            ]),
            media_type="text/event-stream",
            status_code=429,
        )

    async def event_generator():
        try:
            # ISSUE-18：同步 SQLite 调用移入线程池
            thread = await asyncio.to_thread(
                _get_conversation_store().ensure_thread_for_qa,
                thread_id=request.thread_id,
                user_id=user.user_id,
                user_role=user.role,
                client_id=request.client_id,
                title=request.query[:100],
            )
        except Exception as exc:
            yield f"event: error\ndata: {json.dumps({'detail': str(exc)})}\n\n"
            return

        thread_id = thread["thread_id"]
        turn_id = str(uuid.uuid4())
        # ISSUE-23：TTFT 从生成器开始计时，到首个 answer_delta 事件为止，
        # 只记录一次（设计线 P95 ≤5s，impl-08 §2）
        stream_started = time.time()
        ttft_recorded = False
        # request_id 显式生成后传入初始 state：Langfuse 根 trace 与 SQLite 审计
        # 共用同一 id；metadata 只含 request_id/thread_id，无用户身份信息
        request_id = str(uuid.uuid4())
        langfuse = get_langfuse()
        trace = langfuse.start_request_trace(request_id, thread_id)
        trace_token = set_current_trace(trace)
        trace_status = "ok"
        error_type: str | None = None
        try:
            initial_state = build_assistant_initial_state(
                request,
                user,
                thread_id=thread_id,
                turn_id=turn_id,
                turn_index=thread.get("turn_count", 0),
                request_id=request_id,
            )

            agent = _get_agent_app()
            runnable_config: RunnableConfig = {
                "configurable": {"thread_id": thread_id},
                "recursion_limit": AGENT_RECURSION_LIMIT,
            }
            # Langfuse：绑定根 trace 的 callback 经 RunnableConfig 传入 LangGraph
            callback_handler = langfuse.get_callback_handler(trace)
            if callback_handler is not None:
                runnable_config["callbacks"] = [callback_handler]
            # 进度节点集合由图模块声明，传输层不解释节点语义
            from src.agents.graph import CLIENT_PROGRESS_NODES

            try:
                # 流式获取节点状态更新与 reason 节点 LLM token；整条流受请求级
                # 总超时约束（issues.md 一.8：SSE 此前没有总超时包装）。
                # subgraphs=True 才能透出 reason 子图内的 token（ISSUE-9），
                # 此时每项为 (namespace, mode, data) 三元组：外层图 namespace 为
                # 空元组，子图内非空。
                async with asyncio.timeout(config.api_request_timeout_seconds):
                    async for stream_item in agent.astream(
                        initial_state,
                        runnable_config,
                        stream_mode=["updates", "messages"],
                        subgraphs=True,
                    ):
                        # 多模式 + subgraphs 的产出契约：(namespace, mode, data)
                        namespace, mode, data = cast(
                            "tuple[tuple[str, ...], str, Any]", stream_item
                        )
                        if mode == "messages":
                            message_chunk, chunk_metadata = data
                            # 只外发 reason 节点的生成 token；query_understand/planner
                            # 的 JSON 输出与工具调用轮次的中间文本不得进入回答流
                            if chunk_metadata.get("langgraph_node") != "call_reason_model":
                                continue
                            text = message_chunk.content
                            if not (isinstance(text, str) and text):
                                continue
                            if not ttft_recorded:
                                ttft_recorded = True
                                get_metrics().record_ttft(
                                    role=user.role, seconds=time.time() - stream_started
                                )
                            delta_data = json.dumps(
                                {"type": "answer_delta", "delta": text},
                                ensure_ascii=False,
                            )
                            yield f"event: answer_delta\ndata: {delta_data}\n\n"
                            continue

                        # updates：子图内部节点不外发进度，只转发外层图节点
                        if namespace:
                            continue
                        for node_name, node_output in data.items():
                            if not isinstance(node_output, dict):
                                continue
                            # 只发送图模块声明的客户端可见节点进度，避免事件过多
                            if node_name in CLIENT_PROGRESS_NODES:
                                yield (
                                    "event: progress\n"
                                    f"data: {json.dumps({'type': 'progress', 'node': node_name, 'status': 'done'}, ensure_ascii=False)}\n\n"
                                )
                            # 终态以节点声明的 terminal 标记 + final_answer 为准，
                            # 不按节点名推断；ReAct 尝试的中间 final_answer 不带标记，
                            # 新增终态路径（拒答/澄清等）按契约声明 STATE_TERMINAL 即自动生效
                            if node_output.get(STATE_TERMINAL) and STATE_FINAL_ANSWER in node_output:
                                answer_data = json.dumps({
                                    "type": "answer",
                                    "answer": node_output[STATE_FINAL_ANSWER],
                                    "citations": node_output.get(STATE_CITATIONS, []),
                                    "confidence": node_output.get(STATE_CONFIDENCE, "unknown"),
                                    "thread_id": thread_id,
                                    "turn_id": turn_id,
                                }, ensure_ascii=False)
                                yield f"event: answer\ndata: {answer_data}\n\n"
            except asyncio.TimeoutError:
                trace_status = "error"
                error_type = "timeout"
                yield (
                    "event: error\n"
                    f"data: {json.dumps({'type': 'error', 'detail': '请求处理超时'}, ensure_ascii=False)}\n\n"
                )
            except Exception as exc:
                trace_status = "error"
                error_type = type(exc).__name__
                yield (
                    "event: error\n"
                    f"data: {json.dumps({'type': 'error', 'detail': str(exc)[:200]}, ensure_ascii=False)}\n\n"
                )

            yield f"event: done\ndata: {json.dumps({'type': 'done'})}\n\n"
        except GeneratorExit:
            # 客户端断连：generator 在 yield 点被关闭，done 不会发出，
            # trace 必须在此收尾（不能依赖 done 事件路径）
            trace_status = "error"
            error_type = "client_disconnected"
            raise
        except asyncio.CancelledError:
            # 请求任务被取消（断连的另一形态）：同样不是 done 路径
            trace_status = "error"
            error_type = "cancelled"
            raise
        except Exception as exc:
            # 图执行前的准备阶段异常（原样上抛，与既有行为一致）
            trace_status = "error"
            error_type = type(exc).__name__
            raise
        finally:
            reset_current_trace(trace_token)
            trace.finish(status=trace_status, error_type=error_type)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
