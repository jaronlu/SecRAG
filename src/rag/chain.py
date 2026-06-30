from operator import itemgetter
from typing import Any, Dict, List

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_ollama import ChatOllama

from src.rag.prompts import prompt
from src.retrieval.vector_retriever import FinancialVectorRetriever

retriever = FinancialVectorRetriever()


def format_docs(docs: List[Dict]) -> str:
    """把检索结果拼接成 context"""
    lines = []
    for i, doc in enumerate(docs, 1):
        meta = doc["metadata"]
        lines.append(
            f"[来源{i}] {meta.get('title', '未知文档')} "
            f"({meta.get('date', '')})\n"
            f"{doc['content']}\n"
        )
    return "\n".join(lines)


def _retrieve_by_question(x: Dict[str, Any]) -> List[Dict]:
    """RunnableLambda 包装函数，类型标注让 Pylance 能推断"""
    return retriever.retrieve(x["question"], top_k=5)


def build_rag_chain():
    llm = ChatOllama(
        base_url="http://localhost:11434",
        model="llama3.1:8b",
        temperature=0.1,  # 金融场景低温度，减少幻觉
    )

    chain = (
        {
            "context": (
                RunnableLambda(_retrieve_by_question) | RunnableLambda(format_docs)
            ),
            "question": itemgetter("question"),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain
