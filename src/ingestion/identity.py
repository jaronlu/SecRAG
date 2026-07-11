"""Document identity, hashing, loading, and chunk metadata normalization."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from src.schemas.constants import (
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
    PERMISSION_CONFIDENTIAL,
    PERMISSION_INTERNAL,
    PERMISSION_PUBLIC,
    ROLE_ADVISOR,
    ROLE_COMPLIANCE,
    ROLE_INSTITUTIONAL_SALES,
    ROLE_OPERATIONS,
    ROLE_TECHNICAL,
)

SUPPORTED_SUFFIXES = {".pdf", ".docx", ".doc", ".html", ".htm", ".csv", ".xlsx", ".xls"}
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
    digest = hashlib.sha1(f"{source}:{index}:{content}".encode("utf-8")).hexdigest()
    return digest[:24]


def sanitize_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    simple_types = (str, int, float, bool, type(None))
    return {key: value for key, value in meta.items() if isinstance(value, simple_types)}


def load_sample_metadata(file_path: Path) -> dict[str, Any]:
    """Load and validate the required sibling `<filename>.meta.json` manifest."""
    manifest = file_path.with_name(file_path.name + ".meta.json")
    if not manifest.exists():
        raise ValueError(f"缺少权限清单: {manifest}")
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError(f"权限清单必须是 JSON 对象: {manifest}")

    permission = metadata.get(META_PERMISSION_LEVEL)
    valid_permissions = {PERMISSION_PUBLIC, PERMISSION_INTERNAL, PERMISSION_CONFIDENTIAL}
    if permission not in valid_permissions:
        raise ValueError(f"非法 permission_level: {permission}")

    allowed_roles = metadata.get(META_ALLOWED_ROLES, [])
    if not isinstance(allowed_roles, list) or not all(
        isinstance(role, str) for role in allowed_roles
    ):
        raise ValueError("allowed_roles 必须是角色字符串列表")
    valid_roles = {
        ROLE_ADVISOR,
        ROLE_INSTITUTIONAL_SALES,
        ROLE_COMPLIANCE,
        ROLE_OPERATIONS,
        ROLE_TECHNICAL,
    }
    unknown_roles = set(allowed_roles) - valid_roles
    if unknown_roles:
        raise ValueError(f"非法 allowed_roles: {sorted(unknown_roles)}")
    if permission != PERMISSION_PUBLIC and not allowed_roles:
        raise ValueError("internal/confidential 文档必须声明 allowed_roles")
    return metadata


def normalize_manifest_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(metadata)
    allowed_roles = normalized.get(META_ALLOWED_ROLES)
    if isinstance(allowed_roles, list):
        normalized[META_ALLOWED_ROLES] = ",".join(allowed_roles)
    return sanitize_metadata(normalized)


def hash_metadata(metadata: dict[str, Any]) -> str:
    normalized = json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256_text(normalized)


def normalize_parsed_text(documents: Iterable[Any]) -> str:
    return "\n".join(str(document.page_content).strip() for document in documents)


def relative_path(file_path: Path, root_dir: Path | None = None) -> str:
    bases = [root_dir, Path.cwd()]
    for base in bases:
        if base is None:
            continue
        try:
            return str(file_path.resolve().relative_to(base.resolve()))
        except ValueError:
            continue
    return str(file_path)


def file_uri_in_directory(source_uri: str, directory: Path) -> bool:
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

    if file_path.suffix.lower() in {".csv", ".xlsx", ".xls"}:
        dataset_doc_id = _dataset_doc_id(file_path, metadata)
        if dataset_doc_id:
            return dataset_doc_id

    relative = relative_path(file_path)
    return f"path:{sha256_text(relative)[:URL_HASH_PREFIX_LEN]}"


def load_documents(file_path: Path):
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
    if suffix in {".xlsx", ".xls"}:
        from src.ingestion.loaders import load_financial_excel

        return load_financial_excel(file_path)
    raise ValueError(f"不支持的文件格式: {suffix}")


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
    source = str(file_path)
    doc_id = doc_id or derive_doc_id(file_path, doc_type, sample_metadata)
    title = next(
        (chunk.metadata.get(META_TITLE) for chunk in chunks if chunk.metadata.get(META_TITLE)),
        file_path.stem,
    )
    date = next(
        (chunk.metadata.get(META_DATE) for chunk in chunks if chunk.metadata.get(META_DATE)),
        "",
    )
    normalized_sample_metadata = normalize_manifest_metadata(sample_metadata or {})

    for index, chunk in enumerate(chunks):
        chunk.metadata = sanitize_metadata(chunk.metadata)
        chunk_id = build_chunk_id(doc_id, index, chunk.page_content)
        chunk.id = chunk_id
        chunk.metadata.update(normalized_sample_metadata)
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
