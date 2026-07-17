"""Shared helpers for reproducible evaluation artifacts and admission checks."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


def load_dataset(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("评估集必须是 JSON 数组")
    return payload


def current_commit_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "uncommitted"


def portable_dataset_path(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return str(resolved)


def write_artifact(
    *,
    name: str,
    dataset_path: str | Path,
    summary: dict[str, Any],
    output_root: str | Path = "artifacts/evaluation",
) -> Path:
    output_dir = Path(output_root) / current_commit_sha()
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / f"{name}.json"
    artifact_path.write_text(
        json.dumps(
            {
                "commit_sha": current_commit_sha(),
                "dataset": portable_dataset_path(dataset_path),
                "summary": summary,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return artifact_path
