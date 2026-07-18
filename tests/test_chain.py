"""rag/chain.py 单元测试"""

from typing import Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from src.schemas.constants import (
    META_DATE,
    META_TITLE,
    RR_CONTENT,
    RR_METADATA,
)


# ── format_docs ──────────────────────────────────────────────────────────


class TestFormatDocs:
    """format_docs 是纯函数，无需 mock，直接测试"""

    def test_basic(self):
        from src.rag.chain import format_docs

        docs = [
            {RR_CONTENT: "示例公司净利润747亿", RR_METADATA: {META_TITLE: "年报", META_DATE: "2024"}},
            {RR_CONTENT: "对照公司净利润302亿", RR_METADATA: {META_TITLE: "年报", META_DATE: "2024"}},
        ]
        result = format_docs(docs)
        assert "[来源1]" in result
        assert "[来源2]" in result
        assert "示例公司净利润747亿" in result
        assert "对照公司净利润302亿" in result
        assert "2024" in result
        assert "年报" in result

    def test_empty(self):
        from src.rag.chain import format_docs

        assert format_docs([]) == ""

    def test_single(self):
        from src.rag.chain import format_docs

        docs = [
            {RR_CONTENT: "单条结果", RR_METADATA: {META_TITLE: "测试", META_DATE: "2025"}}
        ]
        result = format_docs(docs)
        assert "[来源1] 测试 (2025)" in result
        assert "单条结果" in result

    def test_missing_metadata_fields(self):
        """metadata 缺少 title / date 时应使用默认值"""
        from src.rag.chain import format_docs

        docs = [
            {RR_CONTENT: "无标题", RR_METADATA: {}},
            {RR_CONTENT: "无日期", RR_METADATA: {META_TITLE: "报告"}},
        ]
        result = format_docs(docs)
        assert "未知文档" in result  # title 回退
        assert "报告" in result
        assert "无标题" in result
        assert "无日期" in result

    def test_metadata_none(self):
        """metadata={'title': '文档'} 时 key lookup 使用传递值"""
        from src.rag.chain import format_docs

        docs = [{RR_CONTENT: "内容", RR_METADATA: {META_TITLE: "文档"}}]
        result = format_docs(docs)
        assert "文档" in result
        assert "内容" in result

    def test_index_starts_at_one(self):
        """来源编号从 1 开始"""
        from src.rag.chain import format_docs

        docs = [
            {RR_CONTENT: "a", RR_METADATA: {META_TITLE: "A"}},
            {RR_CONTENT: "b", RR_METADATA: {META_TITLE: "B"}},
            {RR_CONTENT: "c", RR_METADATA: {META_TITLE: "C"}},
        ]
        result = format_docs(docs)
        assert "[来源1]" in result
        assert "[来源2]" in result
        assert "[来源3]" in result
        assert "[来源0]" not in result
        assert "[来源4]" not in result


# ── build_rag_chain ──────────────────────────────────────────────────────


class FakeChatOllama:
    """ChatOllama 的 fake — 可调用，不连真实 Ollama"""

    def __init__(self, **kwargs):
        self.name = "FakeChatOllama"

    def __call__(self, input, **kwargs):
        from langchain_core.messages import AIMessage

        return AIMessage(content="模拟回答")


class FakeRetriever:
    """替换模块级 retriever，避免真实 _embed 加载 HuggingFace 模型"""

    def retrieve(self, query: str, top_k: int = 5, filters: Optional[Dict] = None) -> List[Dict]:
        return [
            {RR_CONTENT: "mock结果", RR_METADATA: {META_TITLE: "mock", META_DATE: "2025"}}
        ]


@pytest.fixture(autouse=False)
def mock_dependencies(monkeypatch):
    """mock 所有外部依赖：retriever + LLM + ChromaDB"""
    # 1) mock ChromaDB（防止 import chain 时 ChromaVectorRetriever 连真实 ChromaDB）
    monkeypatch.setattr("chromadb.PersistentClient", lambda **_: MagicMock())

    # 2) 导入 chain 模块（此时 ChromaDB 已 mock，retriever 创建时不会连真实 DB）
    import src.rag.chain as chain_module

    # 3) 替换模块级 retriever 实例为 FakeRetriever（避免 _embed 加载 HuggingFace）
    monkeypatch.setattr(chain_module, "retriever", FakeRetriever())

    # 4) 替换 _build_llm，无论 provider 配置如何都返回 fake
    monkeypatch.setattr(chain_module, "_build_llm", lambda: FakeChatOllama())

    return chain_module


class TestBuildRagChain:
    def test_returns_runnable(self, mock_dependencies):
        """build_rag_chain 返回一个 LangChain Runnable"""
        from src.rag.chain import build_rag_chain

        chain = build_rag_chain()
        from langchain_core.runnables import Runnable

        assert isinstance(chain, Runnable)

    def test_invoke_returns_string(self, mock_dependencies):
        """chain invoke 后返回字符串"""
        from src.rag.chain import build_rag_chain

        chain = build_rag_chain()
        result = chain.invoke({"question": "示例公司2024净利润是多少？"})
        assert isinstance(result, str)
        assert len(result) > 0
        assert result == "模拟回答"

    def test_invoke_with_empty_question(self, mock_dependencies):
        """空问题也能正常走通"""
        from src.rag.chain import build_rag_chain

        chain = build_rag_chain()
        result = chain.invoke({"question": ""})
        assert isinstance(result, str)


def test_rag_ollama_client_ignores_environment_proxy(monkeypatch):
    import src.rag.chain as chain_module

    chat_ollama = MagicMock()
    monkeypatch.setattr(chain_module.config, "llm_provider", "ollama")
    monkeypatch.setattr("langchain_ollama.ChatOllama", chat_ollama)

    chain_module._build_llm()

    assert chat_ollama.call_args.kwargs["reasoning"] is False
    assert chat_ollama.call_args.kwargs["client_kwargs"] == {"trust_env": False}
