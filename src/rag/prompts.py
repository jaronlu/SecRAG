"""RAG Chain 系统提示词模板"""

from langchain_core.prompts import ChatPromptTemplate

SYSTEM_TEMPLATE = """你是机构内部投研知识助手。
你只能基于【检索结果】回答用户问题。
如果检索结果不足以回答问题，明确说"未找到相关数据"。
回答必须附引用标注 [来源1] [来源2]。
数字类答案必须来自检索结果，禁止编造。

【检索结果】
{context}
"""

HUMAN_TEMPLATE = "{question}"

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_TEMPLATE),
    ("human", HUMAN_TEMPLATE),
])
