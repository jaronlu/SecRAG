"""Local RBAC retrieval smoke checks.

This script does not call the LLM. It validates that role-aware retrieval and
post-retrieval grading enforce the expected visibility on the local Chroma data.
"""

from typing import cast

from src.agents.state import AssistantState
from src.agents.nodes import grade_and_filter
from src.retrieval.hybrid_retriever import HybridRetriever
from src.schemas.constants import (
    PLAN_QUERY,
    PLAN_SOURCE,
    ROLE_INSTITUTIONAL_SALES,
    ROLE_DATA_PERMISSIONS,
    ROLE_OPERATIONS,
    ROLE_TECHNICAL,
    SOURCE_FAQ,
    SOURCE_REPORT,
    STATE_RETRIEVAL_RESULTS,
    RR_DENIED,
)


def _visible_count(role: str, source: str, query: str) -> int:
    retriever = HybridRetriever(
        user_role=role,
        data_permissions=ROLE_DATA_PERMISSIONS.get(role, []),
    )
    results = retriever.retrieve([{PLAN_SOURCE: source, PLAN_QUERY: query, "top_k": 5}])
    state = cast(AssistantState, {STATE_RETRIEVAL_RESULTS: results})
    filtered = grade_and_filter(state)[STATE_RETRIEVAL_RESULTS]
    return sum(not result.get(RR_DENIED) for result in filtered)


def main() -> None:
    checks = {
        "technical_langgraph_faq": _visible_count(ROLE_TECHNICAL, SOURCE_FAQ, "LangGraph") > 0,
        "operations_langgraph_faq_blocked": _visible_count(ROLE_OPERATIONS, SOURCE_FAQ, "LangGraph")
        == 0,
        "sales_langgraph_report_blocked": _visible_count(
            ROLE_INSTITUTIONAL_SALES,
            SOURCE_REPORT,
            "LangGraph",
        )
        == 0,
    }

    failed = [name for name, passed in checks.items() if not passed]
    for name, passed in checks.items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")
    if failed:
        raise SystemExit(f"permission checks failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
