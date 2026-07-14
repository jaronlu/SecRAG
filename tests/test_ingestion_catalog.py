import json
from pathlib import Path

import pytest

from src.ingestion.catalog import (
    UnsafeIngestionPathError,
    get_ingestion_catalog,
    preflight_category,
)
from src.schemas import constants
from src.schemas.constants import ALL_VALID_DOC_TYPES
from src.schemas.typed_dicts import IngestionCategoryConfig


def _write_document(root: Path, name: str = "sample.csv", doc_type: str = "financial_data") -> Path:
    file_path = root / name
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text("code,year\n600519,2026\n", encoding="utf-8")
    file_path.with_name(f"{file_path.name}.meta.json").write_text(
        json.dumps({
            "doc_type": doc_type,
            "permission_level": "internal",
            "allowed_roles": ["technical"],
        }),
        encoding="utf-8",
    )
    return file_path


def _category(relative_path: str) -> IngestionCategoryConfig:
    return {
        "category_id": "financials",
        "label": "财务数据",
        "group": "证券数据",
        "relative_path": relative_path,
        "default_doc_type": "financial_data",
        "allowed_doc_types": ["financial_data"],
    }


def test_global_constants_do_not_encode_demo_categories():
    for name, value in vars(constants).items():
        if name.isupper():
            assert "demo" not in name.lower()
            if isinstance(value, str):
                assert "demo" not in value.lower()


def test_catalog_ids_are_unique_and_doc_types_are_valid():
    catalog = get_ingestion_catalog()

    assert len({item["category_id"] for item in catalog}) == len(catalog)
    for item in catalog:
        assert item["default_doc_type"] in ALL_VALID_DOC_TYPES
        assert set(item["allowed_doc_types"]) <= ALL_VALID_DOC_TYPES


def test_preflight_builds_ready_summary_and_snapshot(tmp_path):
    category_root = tmp_path / "data/raw/financials"
    file_path = _write_document(category_root)

    result = preflight_category(
        _category("data/raw/financials"),
        project_root=tmp_path,
        data_raw_root=tmp_path / "data/raw",
    )

    assert result.summary["ready"] is True
    assert result.summary["file_count"] == 1
    assert result.summary["manifest_count"] == 1
    assert result.files[0]["relative_path"] == "data/raw/financials/sample.csv"
    assert result.snapshots[0].file_hash
    assert result.snapshots[0].metadata_hash
    assert result.snapshots[0].relative_path == str(file_path.relative_to(tmp_path))


def test_business_file_symlink_is_reported_without_reading_target(tmp_path):
    category_root = tmp_path / "data/raw/financials"
    category_root.mkdir(parents=True)
    outside = tmp_path / "outside.csv"
    outside.write_text("secret", encoding="utf-8")
    linked = category_root / "outside.csv"
    linked.symlink_to(outside)

    result = preflight_category(
        _category("data/raw/financials"),
        project_root=tmp_path,
        data_raw_root=tmp_path / "data/raw",
    )

    assert result.summary["ready"] is False
    assert result.files[0]["manifest_status"] == "unsafe_path"
    assert result.snapshots == []


def test_descendant_directory_symlink_rejects_category(tmp_path):
    category_root = tmp_path / "data/raw/financials"
    category_root.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (category_root / "linked-dir").symlink_to(outside, target_is_directory=True)

    with pytest.raises(UnsafeIngestionPathError):
        preflight_category(
            _category("data/raw/financials"),
            project_root=tmp_path,
            data_raw_root=tmp_path / "data/raw",
        )


def test_manifest_symlink_is_reported_as_unsafe(tmp_path):
    category_root = tmp_path / "data/raw/financials"
    file_path = _write_document(category_root)
    manifest = file_path.with_name(f"{file_path.name}.meta.json")
    manifest.unlink()
    outside_manifest = tmp_path / "outside.json"
    outside_manifest.write_text("{}", encoding="utf-8")
    manifest.symlink_to(outside_manifest)

    result = preflight_category(
        _category("data/raw/financials"),
        project_root=tmp_path,
        data_raw_root=tmp_path / "data/raw",
    )

    assert result.files[0]["manifest_status"] == "unsafe_path"
