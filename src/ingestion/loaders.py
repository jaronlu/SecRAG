import hashlib
from pathlib import (
    Path,
)
from typing import List

import pandas as pd
from langchain_core.documents import Document
from langchain_unstructured import UnstructuredLoader

# ⚡ 字段统一：metadata 键名和枚举值见 src/schemas/constants.py
from src.schemas.constants import (
    DOC_TYPE_FINANCIAL_DATA,
    META_DATE,
    META_DOC_ID,
    META_DOC_TYPE,
    META_PERMISSION_LEVEL,
    META_SOURCE,
    META_STOCK_CODE,
    META_TITLE,
    PERMISSION_INTERNAL,
)


def load_pdf(file_path: Path) -> List[Document]:
    """加载 PDF 文档，优先使用 UnstructuredLoader，失败时回退到 pypdf"""
    try:
        # UnstructuredLoader 是 LangChain 的文档加载器（Document Loader）
        # ObjC 类比：≈ [[NSXMLParser alloc] initWithContentsOfURL:url] + delegate 解析
        # 它接收文件路径，内部调用 unstructured 库做版面分析（layout analysis），提取纯文本
        loader = UnstructuredLoader(
            file_path=str(file_path)
        )  # 初始化加载器；file_path 是 Path 对象，str() 转成字符串路径
        # print(f'loader: {loader}')  # ← 遗留调试代码，已移除
        # loader.load() 同步执行解析，返回 List[Document]
        # ObjC 类比：[parser parse] — 阻塞直到解析完成，每个 Document ≈ 解析出的一个文本段落
        return loader.load()
    except Exception as e:
        # 如果 unstructured 缺少依赖（如 pikepdf、pdf2image），或文件损坏，不会崩溃而是返回空列表
        # ObjC 类比：@try { ... } @catch (NSException *e) { return @[]; }
        print(f"error: {e}")
        return []


def load_word(file_path: Path) -> List[Document]:
    """加载 Word 文档"""
    # Word 文档（.docx）也用 UnstructuredLoader 解析
    # 内部调用 mammoth 或 python-docx 提取 docx 中的段落和表格
    # ObjC 类比：≈ NSAttributedString initWithURL:options:documentAttributes:
    loader = UnstructuredLoader(file_path=str(file_path))
    return loader.load()


def load_directory(directory: Path) -> List[Document]:
    """批量加载目录下的文档"""
    documents = []  # ≈ NSMutableArray — 累积所有加载的 Document

    # rglob("*.pdf") 递归遍历 directory 下所有 .pdf 文件
    # ObjC 类比：NSDirectoryEnumerator *enumerator = [fileManager enumeratorAtURL:...];
    for file_path in directory.rglob("*.pdf"):
        documents.extend(load_pdf(file_path))  # extend ≈ [array addObjectsFromArray:pdfDocs]

    for file_path in directory.rglob("*.docx"):
        documents.extend(load_word(file_path))  # 递归加载所有 Word 文档

    for file_path in directory.rglob("*.doc"):
        documents.extend(load_word(file_path))  # 递归加载所有 .doc 格式

    return documents  # 返回合并后的所有文档块列表


def load_html(file_path: Path) -> List[Document]:
    """加载本地公告 HTML"""
    loader = UnstructuredLoader(file_path=str(file_path))
    return loader.load()


def load_financial_csv(file_path: Path) -> List[Document]:
    """加载财务数据 CSV，每行生成一个 Document"""
    df = pd.read_csv(file_path)
    documents = []
    source = str(file_path)
    doc_id = hashlib.sha1(source.encode("utf-8")).hexdigest()[:16]

    for _, row in df.iterrows():
        content = "\n".join([f"{col}: {val}" for col, val in row.items()])
        documents.append(
            Document(
                page_content=content,
                metadata={
                    META_DOC_TYPE: DOC_TYPE_FINANCIAL_DATA,
                    META_SOURCE: source,
                    META_DOC_ID: doc_id,
                    META_TITLE: file_path.stem,
                    META_DATE: str(row.get("year", "")),
                    META_STOCK_CODE: str(row.get("code", "")),
                    META_PERMISSION_LEVEL: PERMISSION_INTERNAL,
                },
            )
        )
    return documents


# __main__ 块：直接运行此文件时执行的测试/演示代码
# ObjC 类比：≈ int main(int argc, char *argv[]) — 脚本入口
if __name__ == "__main__":
    # 以下代码已注释，仅作为路径调试参考
    # print(f"Path(__file__): {Path(__file__)}")  # 当前文件绝对路径
    # print(f"Path(__file__).parent: {Path(__file__).parent}")  # 父目录（ingestion/）
    # print(f"Path(__file__).parent.parent: {Path(__file__).parent.parent}")  # 祖父目录（src/）

    # 加载示例财务 CSV 并打印，手动验证解析结果
    documents = load_financial_csv(
        Path(__file__).parent.parent / "data/announcements/sample-financial.csv"
    )
    print(f"共加载 {len(documents)} 条财务数据：")
    # enumerate(documents, 1) 从 1 开始编号；ObjC 类比：for (int i = 0; i < [docs count]; i++)
    for i, doc in enumerate(documents, 1):
        print(f"\n--- 文档 {i} ---")
        print(f"  内容:\n{doc.page_content}")  # 打印 page_content（纯文本）
        print(f"  元数据: {doc.metadata}")  # 打印 metadata（结构化字段）
