"""Legacy HTML UI removal contract (ISSUE-7).

The React frontend (ChatPage + AdminPage) fully covers the legacy ui.html /
admin.html surface (the only gap, the raw JSON debug view, is covered by
curl / Langfuse instead). After removal:

- ``/legacy`` must no longer be registered;
- the ``src.api.ui`` module must be gone;
- a missing ``frontend/dist`` build must fail fast at startup instead of
  silently falling back to the legacy HTML UI.
"""

import importlib.util
from pathlib import Path

import pytest

from src.api import main as api_main
from src.api.main import app


def test_legacy_route_is_not_registered():
    paths = {getattr(route, "path", None) for route in app.routes}
    assert "/legacy" not in paths


def test_src_api_ui_module_is_deleted():
    assert importlib.util.find_spec("src.api.ui") is None


def test_missing_frontend_dist_fails_fast(monkeypatch):
    from src.api.main import _ensure_frontend_dist

    monkeypatch.setattr(
        api_main, "_FRONTEND_DIST", Path("/nonexistent/secrag-frontend-dist")
    )
    with pytest.raises(RuntimeError, match="npm run build"):
        _ensure_frontend_dist()
