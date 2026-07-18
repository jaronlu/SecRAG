from unittest.mock import patch

import httpx
import pytest

from scripts.demo import (
    DemoValidationError,
    _print_assistant_response,
    _validate_allowed_response,
    _validate_denied_response,
    build_client,
    build_client_timeout,
)


def test_build_client_timeout_separates_connect_and_read_timeout():
    timeout = build_client_timeout(150.0)

    assert isinstance(timeout, httpx.Timeout)
    assert timeout.connect == 5.0
    assert timeout.read == 150.0


def test_build_client_ignores_environment_proxy_for_local_service():
    with patch("scripts.demo.httpx.Client") as client_class:
        build_client("http://127.0.0.1:8001", 150.0)

    assert client_class.call_args.kwargs["trust_env"] is False


def test_print_assistant_response_keeps_citations_concise(capsys):
    _print_assistant_response({
        "answer": "回答",
        "confidence": "medium",
        "compliance": {"passed": True, "flags": []},
        "citations": [{
            "citation_id": "cite_001",
            "doc_title": "风险揭示书",
            "chunk_id": "chunk-1",
            "quote": "本产品风险等级为 R2",
            "relevance_score": 0.8,
            "metadata": {"internal_detail": "不应打印"},
        }],
    })

    output = capsys.readouterr().out
    assert "cite_001 | 风险揭示书 | chunk=chunk-1 | score=0.8" in output
    assert "quote: 本产品风险等级为 R2" in output
    assert "internal_detail" not in output


def test_validate_allowed_response_accepts_verified_demo_result():
    _validate_allowed_response({
        "answer": "## 结论\n\n风险等级为 R2。[来源1]",
        "confidence": "medium",
        "compliance": {"passed": True, "flags": []},
        "citations": [{"quote": "本产品风险等级评定为 R2（中低风险）"}],
    })


def test_validate_allowed_response_rejects_safe_fallback_as_demo_success():
    with pytest.raises(DemoValidationError, match="R2"):
        _validate_allowed_response({
            "answer": "## 结论\n\n当前答案未通过来源或数字验证，无法安全返回。",
            "confidence": "low",
            "compliance": {"passed": True, "flags": []},
            "citations": [],
        })


def test_validate_denied_response_accepts_permission_terminal():
    _validate_denied_response({
        "answer": "## 结论\n\n当前角色无权限访问完成该请求所需的数据源。",
        "confidence": "low",
        "compliance": {"passed": False, "flags": ["permission_denied"]},
        "citations": [],
    })


def test_validate_denied_response_rejects_leaked_citation():
    with pytest.raises(DemoValidationError, match="citations 必须为空"):
        _validate_denied_response({
            "answer": "## 结论\n\n当前角色无权限访问完成该请求所需的数据源。",
            "confidence": "low",
            "compliance": {"passed": False, "flags": ["permission_denied"]},
            "citations": [{"quote": "不应外显"}],
        })
