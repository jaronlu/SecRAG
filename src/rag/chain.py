"""RAG Chain：把检索 + 格式化 + 提示词 + LLM 串成 LCEL pipeline。

编排原则：
- main.py 优先提供外部 context，避免链内重复检索
- 未传入 context 时，自动检索 + 格式化后送入 LLM
"""

from typing import Any, Dict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

from src.config import config
from src.rag.prompts import prompt
from src.retrieval.vector_retriever import ChromaVectorRetriever
from src.schemas.constants import (
    DEFAULT_TOP_K,
    LLM_PROVIDER_OPENAI,
    META_DATE,
    META_TITLE,
    RR_CONTENT,
    RR_METADATA,
)
from src.schemas.typed_dicts import RetrievalResult

# 模块级检索器实例，chain 内部自动检索时复用
retriever = ChromaVectorRetriever()


def format_docs(docs: list[RetrievalResult]) -> str:
    """把检索结果拼接成 context"""
    lines = []
    for i, doc in enumerate(docs, 1):
        meta = doc.get(RR_METADATA, {})
        lines.append(
            f"[来源{i}] {meta.get(META_TITLE, '未知文档')} "
            f"({meta.get(META_DATE, '')})\n"
            f"{doc.get(RR_CONTENT, '')}\n"
        )
    return "\n".join(lines)


def _retrieve_by_question(x: Dict[str, Any]) -> list[RetrievalResult]:
    """RunnableLambda 包装函数，类型标注让 Pylance 能推断"""
    return retriever.retrieve(x["question"], top_k=DEFAULT_TOP_K)


def _build_llm():
    """根据 config.llm.provider 选择 LLM 后端"""
    if config.llm.provider == LLM_PROVIDER_OPENAI:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            base_url=config.llm.base_url,
            model=config.llm.model,
            temperature=config.llm.temperature,
            api_key=config.llm.api_key,
        )
    # fallback: ollama
    from langchain_ollama import ChatOllama

    return ChatOllama(
        base_url=config.llm.base_url,
        model=config.llm.model,
        temperature=config.llm.temperature,
    )


def _prepare_inputs(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """准备 LLM 输入：优先用外部 context，否则自动检索+格式化"""
    return {
        "context": (inputs.get("context") or format_docs(_retrieve_by_question(inputs))),
        "question": inputs["question"],
    }


def build_rag_chain():
    """编译 LCEL RAG pipeline"""
    llm = _build_llm()

    chain = RunnableLambda(_prepare_inputs) | prompt | llm | StrOutputParser()
    return chain
