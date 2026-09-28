"""SSE progress node contract between the backend agent graph and the React frontend.

ISSUE-5: the frontend renders the 7-step progress indicator by matching the
``node`` value of SSE progress events against ``STREAM_NODES`` keys in
``frontend/src/types.ts``. When the two sides drift apart, ``findIndex``
returns -1 for every event and all steps stay grey for the whole request.

The backend LangGraph node registration names are the source of truth:
``CLIENT_PROGRESS_NODES`` in ``src/agents/graph.py``.
"""

import re
from pathlib import Path

from src.agents.graph import CLIENT_PROGRESS_NODES

FRONTEND_TYPES_PATH = Path(__file__).resolve().parents[1] / "frontend" / "src" / "types.ts"


def _stream_node_keys() -> list[str]:
    text = FRONTEND_TYPES_PATH.read_text(encoding="utf-8")
    match = re.search(r"export const STREAM_NODES = \[(.*?)\] as const", text, re.DOTALL)
    assert match, "frontend/src/types.ts must define STREAM_NODES"
    return re.findall(r"key:\s*'([^']+)'", match.group(1))


def test_frontend_stream_nodes_match_backend_progress_nodes():
    keys = _stream_node_keys()
    assert sorted(keys) == sorted(CLIENT_PROGRESS_NODES)


def test_frontend_stream_node_keys_are_unique():
    keys = _stream_node_keys()
    assert len(keys) == len(set(keys))
