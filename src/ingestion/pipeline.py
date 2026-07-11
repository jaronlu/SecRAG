"""Incremental ingestion pipeline."""

from __future__ import annotations

from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings

from src.config import config
from src.ingestion.chunkers import chunk_documents
from src.ingestion.embedder import (
    delete_chunk_ids,
    get_embedding_model,
    list_chunk_ids_by_doc_id,
    upsert_chunks,
)
from src.ingestion.identity import (
    CHUNKER_VERSION,
    PARSER_VERSION,
    SUPPORTED_SUFFIXES,
    derive_doc_id,
    file_uri_in_directory,
    hash_metadata,
    load_documents,
    load_sample_metadata,
    normalize_manifest_metadata,
    normalize_chunks,
    normalize_parsed_text,
    relative_path,
    sha256_file,
    sha256_text,
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
    CHROMA_DEFAULT_PERSIST_DIR,
    INGEST_REGISTRY_DB_PATH,
    META_DATE,
    META_DOC_TYPE,
    META_STOCK_CODE,
    META_TITLE,
)


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
    print(f"处理: {file_path}")
    registry_store = registry_store or DocumentRegistryStore(INGEST_REGISTRY_DB_PATH)
    run_id = run_id or create_run_id()
    resolved_embedding_model = embedding_model or get_embedding_model(config.embedding.model)
    embedding_model_name = config.embedding.model
    sample_metadata = load_sample_metadata(file_path)
    effective_doc_type = sample_metadata.get(META_DOC_TYPE, doc_type)
    doc_id = derive_doc_id(file_path, effective_doc_type, sample_metadata)
    source_uri = file_path.resolve().as_uri()
    relative = relative_path(file_path, root_dir)
    file_hash = sha256_file(file_path)
    metadata_hash = hash_metadata(sample_metadata)
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
        documents = load_documents(file_path)
        if not documents:
            raise RuntimeError("文档解析结果为空")
        manifest_metadata = normalize_manifest_metadata(sample_metadata)
        for document in documents:
            document.metadata.update(manifest_metadata)
        parse_hash = sha256_text(normalize_parsed_text(documents))
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
                relative_path=relative,
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
            relative_path=relative,
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


def iter_supported_files(directory: Path):
    for file_path in sorted(directory.rglob("*")):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_SUFFIXES:
            yield file_path


def ingest_directory(directory: Path, doc_type: str, *, full_scan: bool = False) -> None:
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

    for file_path in iter_supported_files(directory):
        action = ingest_document(
            file_path=file_path,
            doc_type=doc_type,
            registry_store=registry_store,
            run_id=run_id,
            root_dir=directory,
            embedding_model=embedding_model,
        )
        sample_metadata = load_sample_metadata(file_path)
        effective_doc_type = sample_metadata.get(META_DOC_TYPE, doc_type)
        seen_doc_ids.add(derive_doc_id(file_path, effective_doc_type, sample_metadata))
        failed = failed or action == INGEST_ACTION_FAILED

    if full_scan and not failed:
        seen_at = utc_now()
        for record in registry_store.active_documents():
            if record.doc_id in seen_doc_ids:
                continue
            if not file_uri_in_directory(record.source_uri, directory):
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
