from pathlib import Path

import pytest

from scripts.clear_databases import (
    DatabaseTarget,
    UnsafeCleanupTargetError,
    clear_databases,
)


def _targets(root: Path) -> list[DatabaseTarget]:
    return [
        DatabaseTarget("chroma", root / "data/chroma", "directory"),
        DatabaseTarget("financial", root / "data/financial.db", "sqlite"),
        DatabaseTarget("ingest_registry", root / "data/ingest_registry.db", "sqlite"),
        DatabaseTarget("audit", root / "data/audit.db", "sqlite"),
        DatabaseTarget("conversations", root / "data/conversations.db", "sqlite"),
    ]


def _create_databases(root: Path) -> None:
    chroma = root / "data/chroma"
    chroma.mkdir(parents=True)
    (chroma / "chroma.sqlite3").write_text("vector", encoding="utf-8")
    for name in ("financial.db", "ingest_registry.db", "audit.db", "conversations.db"):
        path = root / "data" / name
        path.write_text("sqlite", encoding="utf-8")
        Path(f"{path}-wal").write_text("wal", encoding="utf-8")


def test_dry_run_does_not_remove_databases(tmp_path):
    _create_databases(tmp_path)

    results = clear_databases(_targets(tmp_path), project_root=tmp_path)

    assert {result.status for result in results} == {"would_remove"}
    assert (tmp_path / "data/chroma/chroma.sqlite3").exists()
    assert (tmp_path / "data/financial.db-wal").exists()


def test_confirm_removes_all_databases_and_sidecars_but_preserves_raw(tmp_path):
    _create_databases(tmp_path)
    raw_file = tmp_path / "data/raw/source.csv"
    raw_file.parent.mkdir(parents=True)
    raw_file.write_text("evidence", encoding="utf-8")
    unrelated = tmp_path / "data/keep.txt"
    unrelated.write_text("keep", encoding="utf-8")

    results = clear_databases(_targets(tmp_path), project_root=tmp_path, confirm=True)

    assert {result.status for result in results} == {"removed"}
    assert not (tmp_path / "data/chroma").exists()
    assert not (tmp_path / "data/financial.db").exists()
    assert not (tmp_path / "data/financial.db-wal").exists()
    assert raw_file.read_text(encoding="utf-8") == "evidence"
    assert unrelated.read_text(encoding="utf-8") == "keep"


def test_confirm_is_idempotent_when_targets_are_missing(tmp_path):
    results = clear_databases(_targets(tmp_path), project_root=tmp_path, confirm=True)

    assert {result.status for result in results} == {"missing"}


def test_refuses_outside_project_before_removing_any_target(tmp_path):
    inside = tmp_path / "data/financial.db"
    inside.parent.mkdir(parents=True)
    inside.write_text("keep", encoding="utf-8")
    targets = [
        DatabaseTarget("financial", inside, "sqlite"),
        DatabaseTarget("outside", tmp_path.parent / "outside.db", "sqlite"),
    ]

    with pytest.raises(UnsafeCleanupTargetError, match="项目目录外"):
        clear_databases(targets, project_root=tmp_path, confirm=True)

    assert inside.exists()


def test_refuses_raw_directory_and_symlink_targets(tmp_path):
    raw_dir = tmp_path / "data/raw"
    raw_dir.mkdir(parents=True)
    with pytest.raises(UnsafeCleanupTargetError, match="受保护路径"):
        clear_databases(
            [DatabaseTarget("chroma", raw_dir, "directory")],
            project_root=tmp_path,
            confirm=True,
        )

    real_dir = tmp_path / "data/real-chroma"
    real_dir.mkdir()
    symlink = tmp_path / "data/chroma"
    symlink.symlink_to(real_dir, target_is_directory=True)
    with pytest.raises(UnsafeCleanupTargetError, match="符号链接"):
        clear_databases(
            [DatabaseTarget("chroma", symlink, "directory")],
            project_root=tmp_path,
            confirm=True,
        )


def test_validates_all_target_types_before_removing_any_database(tmp_path):
    financial = tmp_path / "data/financial.db"
    financial.parent.mkdir(parents=True)
    financial.write_text("keep", encoding="utf-8")
    invalid_audit = tmp_path / "data/audit.db"
    invalid_audit.mkdir()

    with pytest.raises(UnsafeCleanupTargetError, match="SQLite 目标不能是目录"):
        clear_databases(
            [
                DatabaseTarget("financial", financial, "sqlite"),
                DatabaseTarget("audit", invalid_audit, "sqlite"),
            ],
            project_root=tmp_path,
            confirm=True,
        )

    assert financial.exists()
