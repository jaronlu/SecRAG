"""业务文档分块策略"""

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ⚡ 字段统一：doc_type 枚举值见 src/schemas/constants.py
from src.schemas.constants import (
    DOC_TYPE_ANNOUNCEMENT,
    DOC_TYPE_FINANCIAL_DATA,
    DOC_TYPE_MEETING_MINUTES,
    DOC_TYPE_REGULATION,
    DOC_TYPE_RESEARCH_REPORT,
)


def create_financial_splitter(
    chunk_size: int = 500,  # 每个 chunk 的最大字符数；≈ NSScanner 的 scanUpToCharactersFromSet 的截断长度
    chunk_overlap: int = 100,  # 相邻 chunk 之间重叠的字符数；≈ NSAttributedString 切分时保留的上下文尾巴，确保语义不被打断
) -> RecursiveCharacterTextSplitter:
    """
    业务文档专用分块器（工厂函数）

    ObjC 类比：相当于一个配置并返回 NSScanner 实例的工厂方法
      - chunk_size: 单次扫描的最大长度
      - chunk_overlap: 两次扫描之间的重叠长度，避免关键信息在边界被切断

    设计原则：
      1. 业务文档信息密度高，chunk_size 可以小一些（500字符）
      2. 保留章节标题作为上下文
      3. 中文优先：按段落/句子切分
    """
    # RecursiveCharacterTextSplitter 是 LangChain 的分块器（Text Splitter）
    # ObjC 类比：≈ [[NSScanner alloc] initWithString:...] — 按分隔符列表递归地切分文本
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,  # chunk 最大字符数；≈ NSScanner 每次读取的最大字节
        chunk_overlap=chunk_overlap,  # chunk 间重叠字符数；≈ 相邻两次扫描共享的尾部字符
        # separators 是切分优先级列表，按顺序尝试，成功就切
        # ObjC 类比：≈ NSCharacterSet 的分组匹配，先匹配最高优先级的分隔符
        separators=[
            "\n\n",  # 最高优先级：空行（段落分隔）；≈ 两个 \n 之间是一段完整内容
            "\n",  # 次高：单换行（行分隔）
            "。",  # 中文句号；≈ 中文句子结束标记
            "；",  # 中文分号；≈ 中文分句分隔
            " ",  # 英文空格；≈ 英文单词分隔
            "",  # 兜底：逐字符切；≈ 最后手段，保证不超过 chunk_size
        ],
        length_function=len,  # 计算字符串长度的函数；≈ strlen()，默认 len() 即字符数（非字节数）
    )


def chunk_documents(
    documents: List[Document],
    doc_type: str,
) -> List[Document]:
    """根据文档类型选择分块策略

    ObjC 类比：≈ NSDictionary<NSString *, NSScanner *> dispatch
    """
    _default = create_financial_splitter

    splitters = {
        # 研究报告 / 法规：复用默认业务分块器（chunk_size=500, overlap=100）
        DOC_TYPE_RESEARCH_REPORT: _default(),
        DOC_TYPE_REGULATION: _default(),
        # 公告：小块快速切分，适合短公告
        DOC_TYPE_ANNOUNCEMENT: RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50,
            separators=["\n\n", "\n", "。", "；", " ", ""],
        ),
        # 财务数据：大块保留完整财务数据段
        DOC_TYPE_FINANCIAL_DATA: RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=200,
            separators=["\n\n", "\n", "。", "；", " ", ""],
        ),
        # 会议纪要：偏小 chunk，纪要结构松散
        DOC_TYPE_MEETING_MINUTES: RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=80,
            separators=["\n\n", "\n", "。", "；", " ", ""],
        ),
    }

    splitter = splitters.get(doc_type, _default())
    return splitter.split_documents(documents)
