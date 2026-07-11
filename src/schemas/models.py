"""Authoritative dataclass models defined by SCHEMA-REFERENCE."""

from dataclasses import dataclass, field
from typing import Optional

from src.schemas.typed_dicts import (
    AuditQuery,
    AuditReasoning,
    AuditResponse,
    AuditRetrieval,
    ComplianceResult,
    VerificationResult,
)


@dataclass
class Citation:
    citation_id: str
    doc_title: str
    source: str
    doc_type: str
    chunk_id: str
    quote: str
    relevance_score: float
    permission_level: str
    page_number: Optional[int] = None
    retrieval_path: list[str] = field(default_factory=lambda: ["vector_search"])
    timestamp: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class AuditEntry:
    request_id: str
    timestamp: str
    user_id: str
    user_role: str
    department: str
    query: AuditQuery
    retrieval: AuditRetrieval
    reasoning: AuditReasoning
    verification: VerificationResult
    compliance: ComplianceResult
    response: AuditResponse
    total_duration_ms: float
