from pathlib import Path                               # ≈ NSString filePath — 跨平台路径操作，类似 NSSearchPathForDirectoriesInDomains 返回的路径字符串
from typing import List                               # ≈ NSArray<id> — 类型注解，声明返回值是 Document 对象的数组
import pandas as pd                                   # ≈ Core Data 的 NSFetchedResultsController + 表格解析 — 把 CSV 读成结构化表格
from langchain_unstructured import UnstructuredLoader  # ≈ NSXMLParser / NSAttributedString 文档解析器 — 把 PDF/Word/HTML 文件解析成结构化文本
from langchain_core.documents import Document         # ≈ @interface Document : NSObject — 文档块，.page_content 是 NSString，.metadata 是 NSDictionary


def load_pdf(file_path: Path) -> List[Document]:
    """加载 PDF 文档，优先使用 UnstructuredLoader，失败时回退到 pypdf"""
    try:
        # UnstructuredLoader 是 LangChain 的文档加载器（Document Loader）
        # ObjC 类比：≈ [[NSXMLParser alloc] initWithContentsOfURL:url] + delegate 解析
        # 它接收文件路径，内部调用 unstructured 库做版面分析（layout analysis），提取纯文本
        loader = UnstructuredLoader(file_path=str(file_path))  # 初始化加载器；file_path 是 Path 对象，str() 转成字符串路径
        # print(f'loader: {loader}')  # ← 遗留调试代码，已移除
        # loader.load() 同步执行解析，返回 List[Document]
        # ObjC 类比：[parser parse] — 阻塞直到解析完成，每个 Document ≈ 解析出的一个文本段落
        return loader.load()
    except Exception as e:
        # 如果 unstructured 缺少依赖（如 pikepdf、pdf2image），或文件损坏，不会崩溃而是返回空列表
        # ObjC 类比：@try { ... } @catch (NSException *e) { return @[]; }
        print(f'error: {e}')
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


def load_announcement(url: str) -> List[Document]:
    """加载公告文档 HTML"""
    # UnstructuredLoader 支持 web_url 参数，直接抓取 URL 内容并解析
    # ObjC 类比：≈ [NSData dataWithContentsOfURL:url] 然后用 NSAttributedString 解析 HTML
    # 适合加载上市公司公告网页、证监会披露页等
    loader = UnstructuredLoader(web_url=url)
    return loader.load()


def load_financial_csv(file_path: Path) -> List[Document]:
    """加载财务数据 CSV"""
    # pandas.read_csv 把 CSV 文件读成 DataFrame（行×列的表格）
    # ObjC 类比：≈ 用 NSRegularExpression 逐行解析 CSV，或使用 NSManagedObject 批量从 CSV 导入
    df = pd.read_csv(file_path)
    documents = []  # 累积转换后的 Document 列表

    # iterrows() 逐行遍历 DataFrame，每行是一个 Series（列名→值的映射）
    # ObjC 类比：≈ for (NSDictionary *row in csvRows)
    for _, row in df.iterrows():
        # 把一行数据转成纯文本：每列用 "列名: 值" 的格式，列之间用换行分隔
        # 示例输出：
        #   code: 600519
        #   year: 2024
        #   net_profit: 747.3
        #   total_assets: 3720.5
        content = "\n".join([f"{col}: {val}" for col, val in row.items()])
        # 构造 Document 对象（LangChain 的标准文档格式）
        # ObjC 类比：≈ [[Document alloc] initWithContent:content metadata:meta]
        documents.append(Document(
            page_content=content,     # 文档正文文本；≈ NSString *content — 将被 embedding 模型向量化的内容
            metadata={                # 元数据字典；≈ NSDictionary — 不参与向量化，但会被存入向量库供后续过滤/溯源
                "doc_type": "financial_data",           # 文档类型标记，用于检索时过滤（如只搜财务数据）
                "source": str(file_path),               # 数据来源文件路径，用于引用溯源
                "stock_code": row.get("code", ""),      # 股票代码，如 "600519"，用于按标的检索
                "year": row.get("year", ""),            # 年份，如 "2024"，用于按年度筛选
                "net_profit": row.get("net_profit", ""), # 净利润，结构化字段，供 Agent 做数值计算
                "total_assets": row.get("total_assets", ""), # 总资产，同上
            }
        ))
    return documents  # 返回 CSV 每一行转换成的 Document 列表


# __main__ 块：直接运行此文件时执行的测试/演示代码
# ObjC 类比：≈ int main(int argc, char *argv[]) — 脚本入口
if __name__ == "__main__":
    # 以下代码已注释，仅作为路径调试参考
    # print(f"Path(__file__): {Path(__file__)}")  # 当前文件绝对路径
    # print(f"Path(__file__).parent: {Path(__file__).parent}")  # 父目录（ingestion/）
    # print(f"Path(__file__).parent.parent: {Path(__file__).parent.parent}")  # 祖父目录（src/）

    # 加载示例财务 CSV 并打印，手动验证解析结果
    documents = load_financial_csv(Path(__file__).parent.parent / "data/announcements/sample-financial.csv")
    print(f"共加载 {len(documents)} 条财务数据：")
    # enumerate(documents, 1) 从 1 开始编号；ObjC 类比：for (int i = 0; i < [docs count]; i++)
    for i, doc in enumerate(documents, 1):
        print(f"\n--- 文档 {i} ---")
        print(f"  内容:\n{doc.page_content}")      # 打印 page_content（纯文本）
        print(f"  元数据: {doc.metadata}")          # 打印 metadata（结构化字段）
