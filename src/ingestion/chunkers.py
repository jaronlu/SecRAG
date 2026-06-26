"""金融文档分块策略"""

from langchain_core.documents import Document                       # ≈ @interface Document : NSObject — 文档块对象，.page_content 是 NSString，.metadata 是 NSDictionary
from langchain_text_splitters import RecursiveCharacterTextSplitter # ≈ NSScanner / NSAttributedString 文本切分工具 — 把长文本按规则切成多个小块（chunk）
from typing import List                                             # ≈ NSArray<id> — 类型注解，声明参数/返回值是对象数组


def create_financial_splitter(
    chunk_size: int = 500,        # 每个 chunk 的最大字符数；≈ NSScanner 的 scanUpToCharactersFromSet 的截断长度
    chunk_overlap: int = 100,     # 相邻 chunk 之间重叠的字符数；≈ NSAttributedString 切分时保留的上下文尾巴，确保语义不被打断
) -> RecursiveCharacterTextSplitter:
    """
    金融文档专用分块器（工厂函数）

    ObjC 类比：相当于一个配置并返回 NSScanner 实例的工厂方法
      - chunk_size: 单次扫描的最大长度
      - chunk_overlap: 两次扫描之间的重叠长度，避免关键信息在边界被切断

    设计原则：
      1. 金融文档信息密度高，chunk_size 可以小一些（500字符）
      2. 保留章节标题作为上下文
      3. 中文优先：按段落/句子切分
    """
    # RecursiveCharacterTextSplitter 是 LangChain 的分块器（Text Splitter）
    # ObjC 类比：≈ [[NSScanner alloc] initWithString:...] — 按分隔符列表递归地切分文本
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,                    # chunk 最大字符数；≈ NSScanner 每次读取的最大字节
        chunk_overlap=chunk_overlap,              # chunk 间重叠字符数；≈ 相邻两次扫描共享的尾部字符
        # separators 是切分优先级列表，按顺序尝试，成功就切
        # ObjC 类比：≈ NSCharacterSet 的分组匹配，先匹配最高优先级的分隔符
        separators=[
            "\n\n",   # 最高优先级：空行（段落分隔）；≈ 两个 \n 之间是一段完整内容
            "\n",     # 次高：单换行（行分隔）
            "。",     # 中文句号；≈ 中文句子结束标记
            "；",     # 中文分号；≈ 中文分句分隔
            " ",      # 英文空格；≈ 英文单词分隔
            "",       # 兜底：逐字符切；≈ 最后手段，保证不超过 chunk_size
        ],
        length_function=len,                      # 计算字符串长度的函数；≈ strlen()，默认 len() 即字符数（非字节数）
    )


def chunk_documents(
    documents: List[Document],  # 待分块的文档列表；≈ NSArray<Document *> — 每个 Document 是一段未切分的原始文本
    doc_type: str,              # 文档类型字符串；≈ NSString — 决定用哪种分块策略，如 "research_report"、"announcement"
) -> List[Document]:           # 返回分块后的文档列表；≈ NSArray<Document *> — chunk 数量通常 ≥ 输入数量
    """根据文档类型选择分块策略"""
    # splitters 字典：每种文档类型对应一个预配置的 TextSplitter
    # ObjC 类比：≈ NSDictionary<NSString *, NSScanner *> — 用 doc_type 作为 key 查找对应的切分器配置
    splitters = {
        # 研究报告：中等 chunk，保留较多上下文
        "research_report": RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=100,     # 500 字符/chunk，100 重叠
            separators=["\n\n", "\n", "。", "；", " ", ""],
        ),
        # 公告：小块快速切分，适合短公告
        "announcement": RecursiveCharacterTextSplitter(
            chunk_size=300, chunk_overlap=50,      # 300 字符/chunk，50 重叠
            separators=["\n\n", "\n", "。", "；", " ", ""],
        ),
        # 财务报告：大块保留完整财务数据段
        "financial_report": RecursiveCharacterTextSplitter(
            chunk_size=800, chunk_overlap=200,     # 800 字符/chunk，200 重叠
            separators=["\n\n", "\n", "。", "；", " ", ""],
        ),
        # 法规：中等 chunk，平衡精度和召回
        "regulation": RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=100,
            separators=["\n\n", "\n", "。", "；", " ", ""],
        ),
        # 会议纪要：偏小 chunk，因为纪要结构松散
        "meeting_minutes": RecursiveCharacterTextSplitter(
            chunk_size=400, chunk_overlap=80,
            separators=["\n\n", "\n", "。", "；", " ", ""],
        ),
    }

    # splitters.get(doc_type, create_financial_splitter())
    # ObjC 类比：≈ [dict objectForKey:type] ?: [self defaultSplitter]
    # 如果 doc_type 在字典中，返回对应的 splitter；否则回退到默认的金融分块器
    splitters = splitters.get(doc_type, create_financial_splitter())

    # split_documents() 执行实际切分
    # ObjC 类比：≈ [scanner scanDocuments:documents intoChunks:] — 遍历 documents，对每个 doc 按规则切分
    # 输出：List[Document]，每个 chunk 继承原始 metadata，page_content 是被切分的文本片段
    return splitters.split_documents(documents)
