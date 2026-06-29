"""rag/chain.py 单元测试"""

from typing import Dict, List
from unittest.mock import MagicMock

import pytest


# ── format_docs ──────────────────────────────────────────────────────────


class TestFormatDocs:
    """format_docs 是纯函数，无需 mock，直接测试"""

    def test_basic(self):
        from src.rag.chain import format_docs

        docs = [
            {"content": "茅台净利润747亿", "metadata": {"title": "年报", "date": "2024"}},
            {"content": "五粮液净利润302亿", "metadata": {"title": "年报", "date": "2024"}},
        ]
        result = format_docs(docs)
        assert "[来源1]" in result
        assert "[来源2]" in result
        assert "茅台净利润747亿" in result
        assert "五粮液净利润302亿" in result
        assert "2024" in result
        assert "年报" in result

    def test_empty(self):
        from src.rag.chain import format_docs

        assert format_docs([]) == ""

    def test_single(self):
        from src.rag.chain import format_docs

        docs = [
            {"content": "单条结果", "metadata": {"title": "测试", "date": "2025"}}
        ]
        result = format_docs(docs)
        assert "[来源1] 测试 (2025)" in result
        assert "单条结果" in result

    def test_missing_metadata_fields(self):
        """metadata 缺少 title / date 时应使用默认值"""
        from src.rag.chain import format_docs

        docs = [
            {"content": "无标题", "metadata": {}},
            {"content": "无日期", "metadata": {"title": "报告"}},
        ]
        result = format_docs(docs)
        assert "未知文档" in result  # title 回退
        assert "报告" in result
        assert "无标题" in result
        assert "无日期" in result

    def test_metadata_none(self):
        """metadata={'title': '文档'} 时 key lookup 使用传递值"""
        from src.rag.chain import format_docs

        docs = [{"content": "内容", "metadata": {"title": "文档"}}]
        result = format_docs(docs)
        assert "文档" in result
        assert "内容" in result

    def test_index_starts_at_one(self):
        """来源编号从 1 开始"""
        from src.rag.chain import format_docs

        docs = [
            {"content": "a", "metadata": {"title": "A"}},
            {"content": "b", "metadata": {"title": "B"}},
            {"content": "c", "metadata": {"title": "C"}},
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

    def retrieve(self, query: str, top_k: int = 5, filters: Dict = None) -> List[Dict]:
        return [
            {"content": "mock结果", "metadata": {"title": "mock", "date": "2025"}}
        ]


@pytest.fixture(autouse=True)
def mock_dependencies(monkeypatch):
    """mock 所有外部依赖：retriever + ChatOllama + ChromaDB"""
    # 1) mock ChromaDB（防止 import chain 时 FinancialVectorRetriever 连真实 ChromaDB）
    monkeypatch.setattr("chromadb.PersistentClient", lambda **_: MagicMock())

    # 2) 导入 chain 模块（此时 ChromaDB 已 mock，retriever 创建时不会连真实 DB）
    import src.rag.chain as chain_module

    # 3) 替换模块级 retriever 实例为 FakeRetriever（避免 _embed 加载 HuggingFace）
    monkeypatch.setattr(chain_module, "retriever", FakeRetriever())

    # 4) mock ChatOllama（不连 localhost:11434）
    monkeypatch.setattr(chain_module, "ChatOllama", FakeChatOllama)


class TestBuildRagChain:
    def test_returns_runnable(self):
        """build_rag_chain 返回一个 LangChain Runnable"""
        from src.rag.chain import build_rag_chain

        chain = build_rag_chain()
        from langchain_core.runnables import Runnable

        assert isinstance(chain, Runnable)

    def test_invoke_returns_string(self):
        """chain invoke 后返回字符串"""
        from src.rag.chain import build_rag_chain

        chain = build_rag_chain()
        result = chain.invoke({"question": "茅台2024净利润是多少？"})
        assert isinstance(result, str)
        assert len(result) > 0
        assert result == "模拟回答"

    def test_invoke_with_empty_question(self):
        """空问题也能正常走通"""
        from src.rag.chain import build_rag_chain

        chain = build_rag_chain()
        result = chain.invoke({"question": ""})
        assert isinstance(result, str)
