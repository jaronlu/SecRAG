"""业务文档分块策略"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
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
    META_PAGE_NUMBER,
)

# 设计分块尺寸（impl-01 §4.2）：研报/法规 500/100、公告 300/50、财报 800/200、纪要 400/80
DESIGN_CHUNK_SIZES: dict[str, tuple[int, int]] = {
    DOC_TYPE_RESEARCH_REPORT: (500, 100),
    DOC_TYPE_REGULATION: (500, 100),
    DOC_TYPE_ANNOUNCEMENT: (300, 50),
    DOC_TYPE_FINANCIAL_DATA: (800, 200),
    DOC_TYPE_MEETING_MINUTES: (400, 80),
}
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 100

# ISSUE-21：UnstructuredLoader 按解析元素返回（一个元素一个 Document），页眉/页脚/
# 邮箱是版面噪声，不应进索引（现网曾把 450+414+2 条噪声写入 Chroma）。
NOISE_ELEMENT_CATEGORIES = frozenset({"Header", "Footer", "EmailAddress"})
# 表格元素整体保留：表头与单位行不得与其描述的数据行分离
TABLE_ELEMENT_CATEGORIES = frozenset({"Table"})
ELEMENT_SEPARATOR = "\n\n"
# 同一文档内重复且达到该长度的 chunk 视为重复版面（如跨页重复的表注），只保留首个
MIN_DEDUP_CHARS = 40
_BLOCK_TYPE_KEY = "block_type"
_BLOCK_TYPE_TEXT = "text"
_BLOCK_TYPE_TABLE = "table"
_HTML_ROW_RE = re.compile(r"<tr\b.*?</tr>", re.IGNORECASE | re.DOTALL)
_WHITESPACE_RE = re.compile(r"\s+")


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


def splitter_for(doc_type: str) -> RecursiveCharacterTextSplitter:
    """按 doc_type 取设计分块器；未登记的 doc_type 回落默认 500/100。"""
    chunk_size, chunk_overlap = DESIGN_CHUNK_SIZES.get(
        doc_type, (DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP)
    )
    return create_financial_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


# ══════════════════════════════════════════════════════════════════════
# ISSUE-21：解析元素聚合
# ══════════════════════════════════════════════════════════════════════


@dataclass
class _Block:
    """聚合后的分块前文档块；page_spans 记录 (起始偏移, 页码) 供 chunk 回填页码。"""

    text: str
    metadata: dict
    kind: str
    page_spans: list[tuple[int, int]] = field(default_factory=list)


def _element_category(document: Document) -> str:
    return str(document.metadata.get("category", "") or "").strip()


def has_structured_elements(documents: List[Document]) -> bool:
    """是否含解析器元素标记（category）——CSV 等结构化数据没有该字段。"""
    return any(_element_category(document) for document in documents)


def _block_metadata(metadata: dict) -> dict:
    """元素元数据转块元数据：去掉元素级 category（块是多个元素的合并）。"""
    return {key: value for key, value in metadata.items() if key != "category"}


def _page_at_offset(page_spans: list[tuple[int, int]], offset: int) -> int | None:
    page: int | None = None
    for start, candidate in page_spans:
        if start > offset:
            break
        page = candidate
    return page


def _aggregate_blocks(documents: List[Document]) -> List[_Block]:
    """按文档顺序把元素合并成块：连续文本元素合成一个流，表格元素独立成块。"""
    blocks: List[_Block] = []
    parts: list[str] = []
    spans: list[tuple[int, int]] = []
    meta: dict = {}
    offset = 0

    def flush() -> None:
        nonlocal parts, spans, meta, offset
        if not parts:
            return
        blocks.append(
            _Block(
                text=ELEMENT_SEPARATOR.join(parts),
                metadata=dict(meta),
                kind=_BLOCK_TYPE_TEXT,
                page_spans=list(spans),
            )
        )
        parts, spans, meta, offset = [], [], {}, 0

    for document in documents:
        category = _element_category(document)
        if category in NOISE_ELEMENT_CATEGORIES:
            continue
        text = document.page_content.strip()
        if not text:
            continue
        page = document.metadata.get(META_PAGE_NUMBER)
        if category in TABLE_ELEMENT_CATEGORIES:
            flush()
            blocks.append(
                _Block(
                    text=text,
                    metadata=_block_metadata(document.metadata),
                    kind=_BLOCK_TYPE_TABLE,
                    page_spans=[(0, page)] if isinstance(page, int) else [],
                )
            )
            continue
        if not parts:
            meta = _block_metadata(document.metadata)
            offset = 0
        else:
            offset += len(ELEMENT_SEPARATOR)
        parts.append(text)
        if isinstance(page, int):
            spans.append((offset, page))
        offset += len(text)

    flush()
    return blocks


def aggregate_document_elements(documents: List[Document]) -> List[Document]:
    """把解析元素聚合成分块前的文档块，并丢弃页眉/页脚/邮箱噪声（ISSUE-21）。

    无 category 标记的文档（CSV/Excel 记录）原样返回，不改变其"一行一记录"语义。
    返回块的 metadata 带 `block_type`（text/table），供切分阶段区分处理。
    """
    if not has_structured_elements(documents):
        return documents
    return [
        Document(
            page_content=block.text,
            metadata={**block.metadata, _BLOCK_TYPE_KEY: block.kind},
        )
        for block in _aggregate_blocks(documents)
    ]


# ══════════════════════════════════════════════════════════════════════
# 分块
# ══════════════════════════════════════════════════════════════════════


def _chunk_metadata(block: _Block, offset: int) -> dict:
    metadata = dict(block.metadata)
    page = _page_at_offset(block.page_spans, offset)
    if page is not None:
        metadata[META_PAGE_NUMBER] = page
    return metadata


def _fingerprint(text: str) -> str:
    return _WHITESPACE_RE.sub("", text)


def _hard_split(content: str, chunk_size: int, block: _Block) -> List[Document]:
    """单个原子单元仍超长时的兜底切分（最后手段，可能切断表格行）。"""
    return [
        Document(
            page_content=content[start : start + chunk_size],
            metadata=_chunk_metadata(block, 0),
        )
        for start in range(0, len(content), chunk_size)
    ]


def _table_rows(text: str) -> tuple[str, list[str], str, str]:
    """拆出表格的行，返回 (前缀, 行列表, 后缀, 行连接符)。"""
    matches = list(_HTML_ROW_RE.finditer(text))
    if matches:
        return (
            text[: matches[0].start()],
            [match.group(0) for match in matches],
            text[matches[-1].end() :],
            "",
        )
    lines = [line for line in text.splitlines() if line.strip()]
    return "", lines, "", "\n"


def _split_table_block(block: _Block, chunk_size: int) -> List[Document]:
    """表格块按行切分，每块重复表头行，保证表头与单位行同 chunk。"""
    text = block.text
    if chunk_size <= 0 or len(text) <= chunk_size:
        return [Document(page_content=text, metadata=_chunk_metadata(block, 0))]

    prefix, rows, suffix, separator = _table_rows(text)
    if len(rows) <= 1:
        return [Document(page_content=text, metadata=_chunk_metadata(block, 0))]

    header = rows[0]
    parts: list[str] = []
    current = header
    for row in rows[1:]:
        candidate = current + separator + row
        if len(prefix + candidate + suffix) > chunk_size and current != header:
            parts.append(current)
            current = header + separator + row
        else:
            current = candidate
    parts.append(current)

    chunks: List[Document] = []
    for part in parts:
        content = f"{prefix}{part}{suffix}"
        if len(content) > chunk_size:
            chunks.extend(_hard_split(content, chunk_size, block))
        else:
            chunks.append(Document(page_content=content, metadata=_chunk_metadata(block, 0)))
    return chunks


def _split_text_block(block: _Block, splitter: RecursiveCharacterTextSplitter) -> List[Document]:
    """文本块切分，并按 chunk 起始偏移回填其所在页码。"""
    cursor = 0
    chunks: List[Document] = []
    for piece in splitter.split_text(block.text):
        if not piece.strip():
            continue
        offset = block.text.find(piece, cursor)
        if offset < 0:
            offset = block.text.find(piece)
        if offset < 0:
            offset = cursor
        cursor = offset + len(piece)
        chunks.append(Document(page_content=piece, metadata=_chunk_metadata(block, offset)))
    return chunks


def chunk_documents(
    documents: List[Document],
    doc_type: str,
) -> List[Document]:
    """根据文档类型选择分块策略

    ObjC 类比：≈ NSDictionary<NSString *, NSScanner *> dispatch

    带解析元素标记的文档先聚合元素再切分（ISSUE-21）：碎片元素直接送切分器时
    短元素原样通过，chunk_size 永不生效，元素之间也从不合并。
    """
    splitter = splitter_for(doc_type)
    if not has_structured_elements(documents):
        return splitter.split_documents(documents)

    chunk_size = int(splitter._chunk_size or 0)
    seen: set[str] = set()
    chunks: List[Document] = []
    for block in _aggregate_blocks(documents):
        produced = (
            _split_table_block(block, chunk_size)
            if block.kind == _BLOCK_TYPE_TABLE
            else _split_text_block(block, splitter)
        )
        for chunk in produced:
            fingerprint = _fingerprint(chunk.page_content)
            if len(fingerprint) >= MIN_DEDUP_CHARS and fingerprint in seen:
                continue
            seen.add(fingerprint)
            chunks.append(chunk)
    return chunks
