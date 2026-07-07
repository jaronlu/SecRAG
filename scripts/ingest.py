"""完整入库流程"""

import csv
import hashlib
import json
import re
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from langchain_huggingface import HuggingFaceEmbeddings

from src.config import config
from src.ingestion.chunkers import chunk_documents
from src.ingestion.embedder import (
    delete_chunk_ids,
    get_embedding_model,
    list_chunk_ids_by_doc_id,
    upsert_chunks,
)
from src.ingestion.registry import (
    INGEST_ACTION_ARCHIVED,
    INGEST_ACTION_CREATED,
    INGEST_ACTION_FAILED,
    INGEST_ACTION_REPLACED,
    INGEST_ACTION_SKIPPED,
    INGEST_RUN_STATUS_FAILED,
    INGEST_RUN_STATUS_SUCCESS,
    REGISTRY_STATUS_ACTIVE,
    DocumentRegistryStore,
    DocumentRegistryUpdate,
    create_run_id,
    utc_now,
)
from src.schemas.constants import (
    ALL_VALID_DOC_TYPES,
    CHROMA_DEFAULT_PERSIST_DIR,
    INGEST_REGISTRY_DB_PATH,
    META_ALLOWED_ROLES,
    META_CHUNK_HASH,
    META_CHUNK_ID,
    META_CHUNK_INDEX,
    META_CHUNKER_VERSION,
    META_DATE,
    META_DOC_ID,
    META_DOC_TYPE,
    META_DOC_VERSION,
    META_EMBEDDING_MODEL,
    META_FILE_HASH,
    META_INGESTED_AT,
    META_METADATA_HASH,
    META_PARSE_HASH,
    META_PARSER_VERSION,
    META_PERMISSION_LEVEL,
    META_RETRIEVAL_SOURCE,
    META_SOURCE,
    META_STOCK_CODE,
    META_TITLE,
    PERMISSION_INTERNAL,
    SAMPLE_METADATA_FILENAME,
)

SUPPORTED_SUFFIXES = {".pdf", ".docx", ".doc", ".html", ".htm", ".csv"}
PARSER_VERSION = "secrag-loader-v1"
CHUNKER_VERSION = "secrag-chunker-v1"
URL_HASH_PREFIX_LEN = 16


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_chunk_id(source: str, index: int, content: str) -> str:
    """构造稳定 chunk id，避免重复运行时反复写入随机 UUID。"""
    digest = hashlib.sha1(f"{source}:{index}:{content}".encode("utf-8")).hexdigest()
    return digest[:24]


def _sanitize_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    """过滤 ChromaDB 不支持的复杂 metadata 类型（bbox 坐标等）"""
    SIMPLE_TYPES = (str, int, float, bool, type(None))
    return {k: v for k, v in meta.items() if isinstance(v, SIMPLE_TYPES)}


def _load_sample_metadata(file_path: Path) -> dict[str, Any]:
    for parent in (file_path.parent, *file_path.parents):
        manifest = parent / SAMPLE_METADATA_FILENAME
        if not manifest.exists():
            continue
        metadata = json.loads(manifest.read_text(encoding="utf-8"))
        try:
            key = str(file_path.relative_to(parent))
        except ValueError:
            continue
        return metadata.get(key, {})
    return {}


def _normalize_manifest_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(metadata)
    allowed_roles = normalized.get(META_ALLOWED_ROLES)
    if isinstance(allowed_roles, list):
        normalized[META_ALLOWED_ROLES] = ",".join(allowed_roles)
    return _sanitize_metadata(normalized)


def _hash_metadata(metadata: dict[str, Any]) -> str:
    normalized = json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256_text(normalized)


def _normalize_parsed_text(documents: Iterable[Any]) -> str:
    return "\n".join(str(document.page_content).strip() for document in documents)


def _relative_path(file_path: Path, root_dir: Path | None = None) -> str:
    bases = [root_dir, Path.cwd()]
    for base in bases:
        if base is None:
            continue
        try:
            return str(file_path.resolve().relative_to(base.resolve()))
        except ValueError:
            continue
    return str(file_path)


def _file_uri_in_directory(source_uri: str, directory: Path) -> bool:
    parsed = urlparse(source_uri)
    if parsed.scheme != "file":
        return False
    try:
        Path(parsed.path).resolve().relative_to(directory.resolve())
    except ValueError:
        return False
    return True


def _canonical_url(url: str) -> str:
    parsed = urlparse(url.strip())
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()
    path = parsed.path
    return f"{scheme}://{netloc}{path}"


def _url_hash_id(url: str) -> str:
    return f"url:{sha256_text(_canonical_url(url))[:URL_HASH_PREFIX_LEN]}"


def _read_csv_rows(file_path: Path) -> list[dict[str, str]]:
    with file_path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _csv_values(rows: list[dict[str, str]], columns: tuple[str, ...]) -> list[str]:
    values = set()
    for row in rows:
        for column in columns:
            value = row.get(column, "").strip()
            if value:
                values.add(value)
    return sorted(values)


def _dataset_doc_id(file_path: Path, metadata: dict[str, Any]) -> str | None:
    source = str(metadata.get(META_SOURCE, ""))
    provider = str(metadata.get("provider", "")).replace("/", ":")
    if not source or not provider:
        return None

    api_name = source.rsplit(".", maxsplit=1)[-1]
    rows = _read_csv_rows(file_path)
    stock_codes = _csv_values(rows, ("stock_code", "sample_stock_code", "股票代码", "code"))
    market_codes = _csv_values(rows, ("code",))
    dates = _csv_values(rows, ("date", "日期"))
    stock_code = str(metadata.get(META_STOCK_CODE, "")).strip()

    if api_name == "query_history_k_data_plus" and market_codes and dates:
        return f"dataset:{provider}:{api_name}:{market_codes[0]}:{dates[0]}:{dates[-1]}"
    if api_name == "get_base_info" and stock_code:
        return f"dataset:{provider}:{api_name}:{stock_code}"
    if stock_codes:
        return f"dataset:{provider}:{api_name}:stocks={','.join(stock_codes)}"

    if stock_code:
        return f"dataset:{provider}:{api_name}:{stock_code}"
    return f"dataset:{provider}:{api_name}:{sha256_file(file_path)[:URL_HASH_PREFIX_LEN]}"


def derive_doc_id(
    file_path: Path,
    doc_type: str,
    sample_metadata: dict[str, Any] | None = None,
) -> str:
    metadata = sample_metadata or {}
    manifest_doc_id = metadata.get(META_DOC_ID)
    if isinstance(manifest_doc_id, str) and manifest_doc_id.strip():
        return manifest_doc_id.strip()

    source = str(metadata.get(META_SOURCE, "")).strip()
    if source.startswith(("http://", "https://")):
        canonical_url = _canonical_url(source)
        cninfo_match = re.search(r"/(\d+)\.pdf$", canonical_url, flags=re.IGNORECASE)
        if "cninfo.com.cn" in canonical_url and cninfo_match:
            return f"cninfo:announcement:{cninfo_match.group(1)}"

        eastmoney_match = re.search(r"/H\d+_(AP\d+)_\d+\.pdf$", canonical_url, flags=re.IGNORECASE)
        if "dfcfw.com" in canonical_url and eastmoney_match:
            return f"eastmoney:research:{eastmoney_match.group(1)}"

        return _url_hash_id(source)

    if file_path.suffix.lower() == ".csv":
        dataset_doc_id = _dataset_doc_id(file_path, metadata)
        if dataset_doc_id:
            return dataset_doc_id

    relative = _relative_path(file_path)
    return f"path:{sha256_text(relative)[:URL_HASH_PREFIX_LEN]}"


def normalize_chunks(
    chunks,
    file_path: Path,
    doc_type: str,
    sample_metadata: dict[str, Any] | None = None,
    *,
    doc_id: str | None = None,
    file_hash: str = "",
    metadata_hash: str = "",
    parse_hash: str = "",
    doc_version: int = 1,
    ingested_at: str = "",
    parser_version: str = PARSER_VERSION,
    chunker_version: str = CHUNKER_VERSION,
    embedding_model: str = "",
):
    """补齐 Chroma 过滤和去重所需的基础元数据。"""
    source = str(file_path)
    doc_id = doc_id or derive_doc_id(file_path, doc_type, sample_metadata)

    # 从 loader 输出或文件路径推断 title / date
    title = next(
        (chunk.metadata.get(META_TITLE) for chunk in chunks if chunk.metadata.get(META_TITLE)),
        file_path.stem,
    )
    date = next(
        (chunk.metadata.get(META_DATE) for chunk in chunks if chunk.metadata.get(META_DATE)),
        "",
    )

    sample_metadata = _normalize_manifest_metadata(sample_metadata or {})

    for index, chunk in enumerate(chunks):
        chunk.metadata = _sanitize_metadata(chunk.metadata)
        chunk_id = build_chunk_id(doc_id, index, chunk.page_content)
        chunk.id = chunk_id
        chunk.metadata.update(sample_metadata)
        chunk.metadata[META_CHUNK_ID] = chunk_id
        chunk.metadata[META_DOC_ID] = doc_id
        chunk.metadata.setdefault(META_DOC_TYPE, doc_type)
        chunk.metadata.setdefault(META_SOURCE, source)
        chunk.metadata.setdefault(META_TITLE, title)
        chunk.metadata.setdefault(META_DATE, date)
        chunk.metadata.setdefault(META_PERMISSION_LEVEL, PERMISSION_INTERNAL)
        chunk.metadata.setdefault(META_RETRIEVAL_SOURCE, "")
        chunk.metadata[META_FILE_HASH] = file_hash
        chunk.metadata[META_METADATA_HASH] = metadata_hash
        chunk.metadata[META_PARSE_HASH] = parse_hash
        chunk.metadata[META_CHUNK_HASH] = sha256_text(chunk.page_content)
        chunk.metadata[META_CHUNK_INDEX] = index
        chunk.metadata[META_DOC_VERSION] = doc_version
        chunk.metadata[META_INGESTED_AT] = ingested_at
        chunk.metadata[META_PARSER_VERSION] = parser_version
        chunk.metadata[META_CHUNKER_VERSION] = chunker_version
        chunk.metadata[META_EMBEDDING_MODEL] = embedding_model
    return chunks


def _load_documents(file_path: Path):
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        from src.ingestion.loaders import load_pdf

        return load_pdf(file_path=file_path)
    if suffix in [".docx", ".doc"]:
        from src.ingestion.loaders import load_word

        return load_word(file_path=file_path)
    if suffix in [".html", ".htm"]:
        from src.ingestion.loaders import load_html

        return load_html(file_path)
    if suffix == ".csv":
        from src.ingestion.loaders import load_financial_csv

        return load_financial_csv(file_path)
    raise ValueError(f"不支持的文件格式: {suffix}")


def _should_skip(existing, file_hash: str, metadata_hash: str, embedding_model_name: str) -> bool:
    if existing is None or existing.status != REGISTRY_STATUS_ACTIVE:
        return False
    return (
        existing.file_hash == file_hash
        and existing.metadata_hash == metadata_hash
        and existing.parser_version == PARSER_VERSION
        and existing.chunker_version == CHUNKER_VERSION
        and existing.embedding_model == embedding_model_name
    )


def ingest_document(
    file_path: Path,
    doc_type: str,
    *,
    registry_store: DocumentRegistryStore | None = None,
    run_id: str | None = None,
    root_dir: Path | None = None,
    embedding_model: HuggingFaceEmbeddings | None = None,
    persist_directory: str = CHROMA_DEFAULT_PERSIST_DIR,
) -> str:
    """单文档入库流程"""
    print(f"处理: {file_path}")
    registry_store = registry_store or DocumentRegistryStore(INGEST_REGISTRY_DB_PATH)
    run_id = run_id or create_run_id()
    resolved_embedding_model = embedding_model or get_embedding_model(config.embedding.model)
    embedding_model_name = config.embedding.model
    sample_metadata = _load_sample_metadata(file_path)
    effective_doc_type = sample_metadata.get(META_DOC_TYPE, doc_type)
    doc_id = derive_doc_id(file_path, effective_doc_type, sample_metadata)
    source_uri = file_path.resolve().as_uri()
    relative_path = _relative_path(file_path, root_dir)
    file_hash = sha256_file(file_path)
    metadata_hash = _hash_metadata(sample_metadata)
    seen_at = utc_now()
    title = str(sample_metadata.get(META_TITLE, file_path.stem))
    stock_code = str(sample_metadata.get(META_STOCK_CODE, ""))
    publish_date = str(sample_metadata.get(META_DATE, ""))
    existing = registry_store.get_document(doc_id)

    if _should_skip(existing, file_hash, metadata_hash, embedding_model_name):
        registry_store.mark_seen(doc_id, seen_at)
        registry_store.record_run_item(
            run_id=run_id,
            doc_id=doc_id,
            source_uri=source_uri,
            action=INGEST_ACTION_SKIPPED,
            previous_hash=existing.file_hash if existing else "",
            new_hash=file_hash,
            chunk_count=existing.chunk_count if existing else 0,
        )
        print(f"  跳过未变化文档: {doc_id}")
        return INGEST_ACTION_SKIPPED

    previous_hash = existing.file_hash if existing else ""
    action = INGEST_ACTION_CREATED if existing is None else INGEST_ACTION_REPLACED
    doc_version = 1 if existing is None else existing.doc_version + 1

    try:
        documents = _load_documents(file_path)
        if not documents:
            raise RuntimeError("文档解析结果为空")
        parse_hash = sha256_text(_normalize_parsed_text(documents))
        chunks = chunk_documents(documents=documents, doc_type=effective_doc_type)
        if not chunks:
            raise RuntimeError("文档分块结果为空")
        ingested_at = utc_now()
        chunks = normalize_chunks(
            chunks,
            file_path,
            effective_doc_type,
            sample_metadata,
            doc_id=doc_id,
            file_hash=file_hash,
            metadata_hash=metadata_hash,
            parse_hash=parse_hash,
            doc_version=doc_version,
            ingested_at=ingested_at,
            parser_version=PARSER_VERSION,
            chunker_version=CHUNKER_VERSION,
            embedding_model=embedding_model_name,
        )
        print(f"  分块数: {len(chunks)}")

        existing_ids = set(
            list_chunk_ids_by_doc_id(
                doc_id=doc_id,
                persist_directory=persist_directory,
                embedding_model=resolved_embedding_model,
            )
        )
        new_ids = {str(chunk.id) for chunk in chunks}
        upsert_chunks(
            chunks=chunks,
            persist_directory=persist_directory,
            embedding_model=resolved_embedding_model,
        )
        delete_chunk_ids(
            chunk_ids=existing_ids - new_ids,
            persist_directory=persist_directory,
            embedding_model=resolved_embedding_model,
        )
        registry_store.upsert_success(
            DocumentRegistryUpdate(
                doc_id=doc_id,
                source_uri=source_uri,
                relative_path=relative_path,
                doc_type=effective_doc_type,
                title=title,
                stock_code=stock_code,
                publish_date=publish_date,
                file_hash=file_hash,
                metadata_hash=metadata_hash,
                parse_hash=parse_hash,
                parser_version=PARSER_VERSION,
                chunker_version=CHUNKER_VERSION,
                embedding_model=embedding_model_name,
                chunk_count=len(chunks),
                doc_version=doc_version,
                last_seen_at=seen_at,
                last_ingested_at=ingested_at,
            )
        )
        registry_store.record_run_item(
            run_id=run_id,
            doc_id=doc_id,
            source_uri=source_uri,
            action=action,
            previous_hash=previous_hash,
            new_hash=file_hash,
            chunk_count=len(chunks),
        )
        print(f"  入库完成: {len(chunks)} chunks, action={action}, doc_id={doc_id}")
        return action
    except Exception as exc:
        error = str(exc)
        registry_store.mark_failed(
            doc_id=doc_id,
            source_uri=source_uri,
            relative_path=relative_path,
            doc_type=effective_doc_type,
            title=title,
            stock_code=stock_code,
            publish_date=publish_date,
            file_hash=file_hash,
            metadata_hash=metadata_hash,
            parser_version=PARSER_VERSION,
            chunker_version=CHUNKER_VERSION,
            embedding_model=embedding_model_name,
            seen_at=seen_at,
            error=error,
        )
        registry_store.record_run_item(
            run_id=run_id,
            doc_id=doc_id,
            source_uri=source_uri,
            action=INGEST_ACTION_FAILED,
            previous_hash=previous_hash,
            new_hash=file_hash,
            error=error,
        )
        print(f"失败: {file_path}, 错误: {error}")
        return INGEST_ACTION_FAILED


def _iter_supported_files(directory: Path):
    for file_path in sorted(directory.rglob("*")):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_SUFFIXES:
            yield file_path


def ingest_directory(directory: Path, doc_type: str, *, full_scan: bool = False):
    """批量入库"""
    registry_store = DocumentRegistryStore(INGEST_REGISTRY_DB_PATH)
    run_id = create_run_id()
    started_at = utc_now()
    registry_store.start_run(
        run_id=run_id,
        root_uri=directory.resolve().as_uri(),
        full_scan=full_scan,
        started_at=started_at,
    )
    embedding_model = get_embedding_model(config.embedding.model)
    seen_doc_ids = set()
    failed = False

    for file_path in _iter_supported_files(directory):
        action = ingest_document(
            file_path=file_path,
            doc_type=doc_type,
            registry_store=registry_store,
            run_id=run_id,
            root_dir=directory,
            embedding_model=embedding_model,
        )
        sample_metadata = _load_sample_metadata(file_path)
        effective_doc_type = sample_metadata.get(META_DOC_TYPE, doc_type)
        seen_doc_ids.add(derive_doc_id(file_path, effective_doc_type, sample_metadata))
        failed = failed or action == INGEST_ACTION_FAILED

    if full_scan and not failed:
        seen_at = utc_now()
        for record in registry_store.active_documents():
            if record.doc_id in seen_doc_ids:
                continue
            if not _file_uri_in_directory(record.source_uri, directory):
                continue
            old_ids = list_chunk_ids_by_doc_id(
                doc_id=record.doc_id,
                persist_directory=CHROMA_DEFAULT_PERSIST_DIR,
                embedding_model=embedding_model,
            )
            delete_chunk_ids(
                chunk_ids=old_ids,
                persist_directory=CHROMA_DEFAULT_PERSIST_DIR,
                embedding_model=embedding_model,
            )
            registry_store.mark_archived(record.doc_id, seen_at)
            registry_store.record_run_item(
                run_id=run_id,
                doc_id=record.doc_id,
                source_uri=record.source_uri,
                action=INGEST_ACTION_ARCHIVED,
                previous_hash=record.file_hash,
                chunk_count=0,
            )

    status = INGEST_RUN_STATUS_FAILED if failed else INGEST_RUN_STATUS_SUCCESS
    registry_store.finish_run(run_id=run_id, status=status, finished_at=utc_now())
    print(f"入库任务完成: run_id={run_id}, status={status}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python scripts/ingest.py <目录路径> <文档类型> [--full-scan]")
        print(
            "文档类型: research_report / announcement / regulation / financial_data / meeting_minutes / product / faq"
        )
        sys.exit(1)

    directory = Path(sys.argv[1])
    doc_type = sys.argv[2]
    if doc_type not in ALL_VALID_DOC_TYPES:
        print(f"不支持的文档类型: {doc_type}")
        sys.exit(1)
    ingest_directory(
        directory=directory, doc_type=doc_type, full_scan="--full-scan" in sys.argv[3:]
    )
