"""Server-controlled ingestion catalog and filesystem preflight."""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.ingestion.identity import (
    SUPPORTED_SUFFIXES,
    hash_metadata,
    load_sample_metadata,
    sha256_file,
)
from src.schemas.constants import (
    ALL_VALID_DOC_TYPES,
    DOC_TYPE_ANNOUNCEMENT,
    DOC_TYPE_FAQ,
    DOC_TYPE_FINANCIAL_DATA,
    DOC_TYPE_PRODUCT,
    DOC_TYPE_REGULATION,
    DOC_TYPE_RESEARCH_REPORT,
    INGEST_ERROR_UNSAFE_SOURCE_PATH,
    META_ALLOWED_ROLES,
    META_DOC_TYPE,
    META_PERMISSION_LEVEL,
)
from src.schemas.typed_dicts import (
    IngestionCategoryConfig,
    IngestionCategorySummary,
    IngestionFile,
)


def get_ingestion_catalog() -> tuple[IngestionCategoryConfig, ...]:
    return (
        {
            "category_id": "demo_product",
            "label": "Demo / 产品文档",
            "group": "Demo 知识库",
            "relative_path": "data/raw/demo_knowledge_base/samples/product",
            "default_doc_type": DOC_TYPE_PRODUCT,
            "allowed_doc_types": [DOC_TYPE_PRODUCT],
        },
        {
            "category_id": "demo_regulation",
            "label": "Demo / 法规文档",
            "group": "Demo 知识库",
            "relative_path": "data/raw/demo_knowledge_base/samples/regulation",
            "default_doc_type": DOC_TYPE_REGULATION,
            "allowed_doc_types": [DOC_TYPE_REGULATION],
        },
        {
            "category_id": "demo_faq",
            "label": "Demo / FAQ 与操作流程",
            "group": "Demo 知识库",
            "relative_path": "data/raw/demo_knowledge_base/samples/faq",
            "default_doc_type": DOC_TYPE_FAQ,
            "allowed_doc_types": [DOC_TYPE_FAQ],
        },
        {
            "category_id": "demo_report",
            "label": "Demo / 研报摘要",
            "group": "Demo 知识库",
            "relative_path": "data/raw/demo_knowledge_base/samples/report",
            "default_doc_type": DOC_TYPE_RESEARCH_REPORT,
            "allowed_doc_types": [DOC_TYPE_RESEARCH_REPORT],
        },
        {
            "category_id": "demo_mixed",
            "label": "Demo / 混合兼容样本",
            "group": "Demo 知识库",
            "relative_path": "data/raw/demo_knowledge_base/announcements",
            "default_doc_type": DOC_TYPE_FAQ,
            "allowed_doc_types": [DOC_TYPE_FAQ, DOC_TYPE_FINANCIAL_DATA],
        },
        {
            "category_id": "real_announcements",
            "label": "真实证券数据 / 公告",
            "group": "真实证券数据",
            "relative_path": "data/raw/real_securities_data/announcements",
            "default_doc_type": DOC_TYPE_ANNOUNCEMENT,
            "allowed_doc_types": [DOC_TYPE_ANNOUNCEMENT],
        },
        {
            "category_id": "real_reports",
            "label": "真实证券数据 / 研报",
            "group": "真实证券数据",
            "relative_path": "data/raw/real_securities_data/reports",
            "default_doc_type": DOC_TYPE_RESEARCH_REPORT,
            "allowed_doc_types": [DOC_TYPE_RESEARCH_REPORT],
        },
        {
            "category_id": "real_financials",
            "label": "真实证券数据 / 财务数据",
            "group": "真实证券数据",
            "relative_path": "data/raw/real_securities_data/financials",
            "default_doc_type": DOC_TYPE_FINANCIAL_DATA,
            "allowed_doc_types": [DOC_TYPE_FINANCIAL_DATA],
        },
    )


class UnknownIngestionCategoryError(LookupError):
    pass


class UnsafeIngestionPathError(ValueError):
    pass


@dataclass(frozen=True)
class SnapshotCandidate:
    relative_path: str
    file_hash: str
    metadata_hash: str
    doc_type: str


@dataclass(frozen=True)
class CategoryPreflight:
    config: IngestionCategoryConfig
    summary: IngestionCategorySummary
    files: list[IngestionFile]
    snapshots: list[SnapshotCandidate]
    category_root: Path


def get_category_config(
    category_id: str,
    catalog: tuple[IngestionCategoryConfig, ...] | None = None,
) -> IngestionCategoryConfig:
    for category in catalog or get_ingestion_catalog():
        if category["category_id"] == category_id:
            return category
    raise UnknownIngestionCategoryError(category_id)


def _absolute_lexical(path: Path) -> Path:
    return Path(os.path.abspath(path))


def _assert_no_symlink_components(base: Path, target: Path) -> None:
    try:
        relative = target.relative_to(base)
    except ValueError as exc:
        raise UnsafeIngestionPathError("路径不属于允许根目录") from exc

    current = base
    for part in (".", *relative.parts):
        if part != ".":
            current = current / part
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError:
            raise
        if stat.S_ISLNK(mode):
            raise UnsafeIngestionPathError("路径包含符号链接")


def ensure_safe_source_path(path: Path, category_root: Path, data_raw_root: Path) -> Path:
    raw_lexical = _absolute_lexical(data_raw_root)
    category_lexical = _absolute_lexical(category_root)
    path_lexical = _absolute_lexical(path)

    _assert_no_symlink_components(raw_lexical, category_lexical)
    _assert_no_symlink_components(category_lexical, path_lexical)

    resolved_raw = raw_lexical.resolve(strict=True)
    resolved_category = category_lexical.resolve(strict=True)
    resolved_path = path_lexical.resolve(strict=True)
    if not resolved_category.is_relative_to(resolved_raw):
        raise UnsafeIngestionPathError("分类目录越过 DATA_RAW_ROOT")
    if not resolved_path.is_relative_to(resolved_category):
        raise UnsafeIngestionPathError("入库路径越过分类目录")
    if not resolved_path.is_relative_to(resolved_raw):
        raise UnsafeIngestionPathError("入库路径越过 DATA_RAW_ROOT")
    return resolved_path


def _project_relative(path: Path, project_root: Path) -> str:
    return str(_absolute_lexical(path).relative_to(_absolute_lexical(project_root)))


def _unsafe_file(path: Path, project_root: Path, extension: str) -> IngestionFile:
    return {
        "relative_path": _project_relative(path, project_root),
        "extension": extension,
        "doc_type": "",
        "permission_level": "",
        "allowed_roles": [],
        "manifest_status": "unsafe_path",
        "error": "文件或权限清单路径不安全",
    }


def _invalid_file(
    path: Path,
    project_root: Path,
    extension: str,
    status: str,
    error: str,
) -> IngestionFile:
    return {
        "relative_path": _project_relative(path, project_root),
        "extension": extension,
        "doc_type": "",
        "permission_level": "",
        "allowed_roles": [],
        "manifest_status": status,
        "error": error,
    }


def preflight_category(
    category: IngestionCategoryConfig,
    *,
    project_root: Path,
    data_raw_root: Path,
) -> CategoryPreflight:
    category_root = project_root / category["relative_path"]
    ensure_safe_source_path(category_root, category_root, data_raw_root)

    candidates: list[Path] = []
    for candidate in sorted(category_root.rglob("*")):
        if candidate.is_symlink():
            if candidate.suffix.lower() in SUPPORTED_SUFFIXES:
                candidates.append(candidate)
                continue
            if candidate.name.endswith(".meta.json"):
                continue
            raise UnsafeIngestionPathError("分类目录包含符号链接")
        if candidate.is_dir():
            ensure_safe_source_path(candidate, category_root, data_raw_root)
            continue
        if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_SUFFIXES:
            candidates.append(candidate)

    files: list[IngestionFile] = []
    snapshots: list[SnapshotCandidate] = []
    manifest_count = 0
    unsafe_error = False
    for file_path in candidates:
        extension = file_path.suffix.lower()
        manifest_path = file_path.with_name(f"{file_path.name}.meta.json")
        if file_path.is_symlink() or manifest_path.is_symlink():
            files.append(_unsafe_file(file_path, project_root, extension))
            unsafe_error = True
            if manifest_path.exists():
                manifest_count += 1
            continue

        try:
            ensure_safe_source_path(file_path, category_root, data_raw_root)
        except (FileNotFoundError, UnsafeIngestionPathError):
            files.append(_unsafe_file(file_path, project_root, extension))
            unsafe_error = True
            continue

        if not manifest_path.exists():
            files.append(
                _invalid_file(
                    file_path,
                    project_root,
                    extension,
                    "missing",
                    "缺少 sibling .meta.json",
                )
            )
            continue

        manifest_count += 1
        try:
            ensure_safe_source_path(manifest_path, category_root, data_raw_root)
            metadata: dict[str, Any] = load_sample_metadata(file_path)
            doc_type = metadata.get(META_DOC_TYPE)
            if doc_type not in ALL_VALID_DOC_TYPES:
                raise ValueError("manifest 缺少或包含非法 doc_type")
            if doc_type not in category["allowed_doc_types"]:
                files.append(
                    _invalid_file(
                        file_path,
                        project_root,
                        extension,
                        "type_mismatch",
                        "manifest doc_type 不属于所选分类",
                    )
                )
                continue
        except UnsafeIngestionPathError:
            files.append(_unsafe_file(file_path, project_root, extension))
            unsafe_error = True
            continue
        except (OSError, ValueError, TypeError):
            files.append(
                _invalid_file(
                    file_path,
                    project_root,
                    extension,
                    "invalid",
                    "manifest 内容无效",
                )
            )
            continue

        relative = _project_relative(file_path, project_root)
        files.append({
            "relative_path": relative,
            "extension": extension,
            "doc_type": str(doc_type),
            "permission_level": str(metadata.get(META_PERMISSION_LEVEL, "")),
            "allowed_roles": list(metadata.get(META_ALLOWED_ROLES, [])),
            "manifest_status": "valid",
            "error": "",
        })
        snapshots.append(
            SnapshotCandidate(
                relative_path=relative,
                file_hash=sha256_file(file_path),
                metadata_hash=hash_metadata(metadata),
                doc_type=str(doc_type),
            )
        )

    invalid_count = sum(file["manifest_status"] != "valid" for file in files)
    ready = bool(files) and invalid_count == 0
    summary: IngestionCategorySummary = {
        **category,
        "file_count": len(files),
        "manifest_count": manifest_count,
        "invalid_manifest_count": invalid_count,
        "ready": ready,
        "error_code": INGEST_ERROR_UNSAFE_SOURCE_PATH if unsafe_error else "",
        "error": "存在不安全路径" if unsafe_error else "",
    }
    return CategoryPreflight(
        config=category,
        summary=summary,
        files=files,
        snapshots=snapshots,
        category_root=category_root,
    )
