"""Safely remove all local SecRAG database storage."""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

from pydantic_settings import BaseSettings, SettingsConfigDict

from src.schemas.constants import (
    AUDIT_DB_PATH,
    CHROMA_DEFAULT_PERSIST_DIR,
    CONVERSATION_DB_PATH,
    FINANCIAL_DB_PATH,
    INGEST_REGISTRY_DB_PATH,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SQLITE_SIDECAR_SUFFIXES = ("-wal", "-shm", "-journal")
TargetKind = Literal["directory", "sqlite"]


class CleanupSettings(BaseSettings):
    """Load cleanup-relevant paths without requiring LLM credentials."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    chroma_persist_directory: str = CHROMA_DEFAULT_PERSIST_DIR
    audit_db_path: str = AUDIT_DB_PATH
    conversation_db_path: str = CONVERSATION_DB_PATH


@dataclass(frozen=True)
class DatabaseTarget:
    name: str
    path: Path
    kind: TargetKind


@dataclass(frozen=True)
class CleanupResult:
    name: str
    path: Path
    status: Literal["would_remove", "removed", "missing"]


class UnsafeCleanupTargetError(ValueError):
    """Raised before deletion when any target violates the cleanup boundary."""


def default_targets() -> list[DatabaseTarget]:
    settings = CleanupSettings()
    return [
        DatabaseTarget("chroma", Path(settings.chroma_persist_directory), "directory"),
        DatabaseTarget("financial", Path(FINANCIAL_DB_PATH), "sqlite"),
        DatabaseTarget("ingest_registry", Path(INGEST_REGISTRY_DB_PATH), "sqlite"),
        DatabaseTarget("audit", Path(settings.audit_db_path), "sqlite"),
        DatabaseTarget("conversations", Path(settings.conversation_db_path), "sqlite"),
    ]


def clear_databases(
    targets: Sequence[DatabaseTarget],
    *,
    project_root: Path = PROJECT_ROOT,
    confirm: bool = False,
    allow_outside_project: bool = False,
) -> list[CleanupResult]:
    """Validate every target first, then remove databases when confirmed."""
    root = project_root.resolve()
    validated = [
        (target, _validate_target(target, root, allow_outside_project))
        for target in targets
    ]
    for target, path in validated:
        _validate_target_type(target, path)

    results: list[CleanupResult] = []
    for target, path in validated:
        candidates = _target_candidates(target.kind, path)
        existing = [candidate for candidate in candidates if candidate.exists()]
        if not existing:
            results.append(CleanupResult(target.name, path, "missing"))
            continue
        if not confirm:
            results.append(CleanupResult(target.name, path, "would_remove"))
            continue

        if target.kind == "directory":
            shutil.rmtree(path)
        else:
            for candidate in existing:
                candidate.unlink()
        results.append(CleanupResult(target.name, path, "removed"))
    return results


def _validate_target(
    target: DatabaseTarget,
    project_root: Path,
    allow_outside_project: bool,
) -> Path:
    raw_path = target.path if target.path.is_absolute() else project_root / target.path
    if raw_path.is_symlink():
        raise UnsafeCleanupTargetError(f"拒绝清理符号链接: {raw_path}")

    path = raw_path.resolve(strict=False)
    data_root = project_root / "data"
    protected_roots = (data_root / "raw", project_root / "artifacts", project_root / ".git")
    if path in (project_root, data_root):
        raise UnsafeCleanupTargetError(f"拒绝清理高风险目录: {path}")
    if any(path == protected or path.is_relative_to(protected) for protected in protected_roots):
        raise UnsafeCleanupTargetError(f"拒绝清理受保护路径: {path}")
    if not allow_outside_project and not path.is_relative_to(project_root):
        raise UnsafeCleanupTargetError(f"数据库路径位于项目目录外: {path}")
    return path


def _target_candidates(kind: TargetKind, path: Path) -> list[Path]:
    if kind == "directory":
        return [path]
    return [path, *(Path(f"{path}{suffix}") for suffix in SQLITE_SIDECAR_SUFFIXES)]


def _validate_target_type(target: DatabaseTarget, path: Path) -> None:
    candidates = _target_candidates(target.kind, path)
    for candidate in candidates:
        if candidate.is_symlink():
            raise UnsafeCleanupTargetError(f"拒绝清理符号链接: {candidate}")
    if target.kind == "directory" and path.exists() and not path.is_dir():
        raise UnsafeCleanupTargetError(f"目录型数据库目标不是目录: {path}")
    if target.kind == "sqlite":
        directory = next((candidate for candidate in candidates if candidate.is_dir()), None)
        if directory is not None:
            raise UnsafeCleanupTargetError(f"SQLite 目标不能是目录: {directory}")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview or remove all local SecRAG databases. Stop the service first."
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Actually delete the validated database targets. The default is dry-run.",
    )
    parser.add_argument(
        "--allow-outside-project",
        action="store_true",
        help="Allow configured database paths outside the SecRAG project directory.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    mode = "confirm" if args.confirm else "dry-run"
    print(f"mode: {mode}")
    print("warning: stop the SecRAG service before confirmed cleanup")
    try:
        results = clear_databases(
            default_targets(),
            confirm=args.confirm,
            allow_outside_project=args.allow_outside_project,
        )
    except UnsafeCleanupTargetError as exc:
        print(f"refused: {exc}")
        return 2

    for result in results:
        print(f"{result.status}: {result.name}: {result.path}")
    if not args.confirm:
        print("no files removed; rerun with --confirm to execute")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
