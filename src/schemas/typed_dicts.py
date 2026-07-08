"""Shared TypedDict schemas for agent state and serialized payloads."""

from typing_extensions import TypedDict


class RetrievalPlanStep(TypedDict, total=False):
    """Single retrieval plan step stored under STATE_RETRIEVAL_PLAN."""

    source: str
    query: str
    top_k: int
    filters: dict | None
    denied: bool
    reason: str


class RetrievalResultOptional(TypedDict, total=False):
    """Optional retrieval result fields."""

    denied: bool
    reason: str


class RetrievalResult(RetrievalResultOptional):
    """Single retrieval result stored under STATE_RETRIEVAL_RESULTS."""

    content: str
    metadata: dict
    score: float


class ToolCallDict(TypedDict, total=False):
    """Serialized tool call trace stored under STATE_TOOL_CALLS."""

    tool: str
    input: dict
    output: str
    duration_ms: float
    success: bool
    error: str


class IntermediateStep(TypedDict, total=False):
    """Reserved ReAct/subgraph step record stored under STATE_INTERMEDIATE_STEPS."""

    step: str
    input: dict
    output: str
    metadata: dict


class QueryEntities(TypedDict, total=False):
    """Entities extracted by query_understand."""

    product_name: str
    product_type: str
    stock_code: str
    regulation_name: str
    client_segment: str


class VerificationResult(TypedDict, total=False):
    """Verification result stored under STATE_VERIFICATION.

    total=False keeps initialization with an empty dict valid; the verify node
    emits all fields.
    """

    passed: bool
    issues: list[str]
    confidence: str


class ComplianceResult(TypedDict, total=False):
    """Compliance result stored under STATE_COMPLIANCE.

    total=False keeps initialization with an empty dict valid; the compliance
    node emits all fields.
    """

    passed: bool
    flags: list[str]
    risk_disclosure: str
    suitability_warning: str


class CitationDict(TypedDict, total=False):
    """Serialized Citation payload."""

    citation_id: str
    doc_title: str
    source: str
    doc_type: str
    chunk_id: str
    quote: str
    relevance_score: float
    permission_level: str
    page_number: int | None


class ConversationThreadDict(TypedDict, total=False):
    """Conversation thread metadata."""

    thread_id: str
    user_id: str
    user_role: str
    client_id: str | None
    title: str
    status: str
    turn_count: int
    created_at: str
    updated_at: str
    deleted_at: str | None


class ConversationMessageDict(TypedDict, total=False):
    """User-visible conversation message."""

    message_id: str
    thread_id: str
    turn_id: str
    role: str
    content: str
    sequence: int
    created_at: str
    request_id: str | None
    deleted_at: str | None


class ConversationTurnDict(TypedDict, total=False):
    """Structured QA turn summary."""

    turn_id: str
    thread_id: str
    user_query: str
    resolved_query: str
    answer_summary: str
    entities: QueryEntities
    citations: list[CitationDict]
    request_id: str
    created_at: str


class AuditQuery(TypedDict, total=False):
    original: str
    rewritten: str
    intent: str
    query_type: str
    entities: QueryEntities


class AuditRetrieval(TypedDict, total=False):
    plan: list[RetrievalPlanStep]
    sources: list[str]
    total_chunks: int
    filtered_chunks: int


class AuditReasoning(TypedDict, total=False):
    tool_calls: list[ToolCallDict]
    iterations: int
    duration_ms: float


class AuditResponse(TypedDict, total=False):
    citations: list[CitationDict]
    confidence: str
    risk_disclosure: str


class AuditTrail(TypedDict, total=False):
    """Audit trail stored under STATE_AUDIT_TRAIL.

    Request initialization stores request_id/timestamp/_started_perf_counter;
    audit_log replaces it with the full serialized AuditEntry structure.
    """

    request_id: str
    timestamp: str
    _started_perf_counter: float
    user_id: str
    user_role: str
    department: str
    query: AuditQuery
    retrieval: AuditRetrieval
    reasoning: AuditReasoning
    verification: VerificationResult
    compliance: ComplianceResult
    response: AuditResponse
