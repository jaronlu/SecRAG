"""一次性迁移：为存量 Chroma collection 的元数据补写数值 date_day（yyyymmdd）。

背景（todo/issues.md ISSUE-1）：date_day 数值化改造（src/utils/dates.py）只对
增量入库生效，存量 collection 只有字符串型 date 元数据；Chroma 数值范围过滤
对旧数据必然 0 召回（fail-closed，非泄漏）。本脚本读取每个 chunk 的现有
metadata，用 parse_date_day 从 date 解析出 date_day 后连完整 metadata 回写，
不改动 embedding 与文本。

- 幂等：已有 date_day 的 chunk 跳过，可重复执行。
- 无法解析的 date（空串、"最近三个月"等）保持缺失，按 doc_type 汇总为豁免清单。
- 默认 dry-run 只统计；--apply 才写库。

用法：
  uv run python scripts/migrate_date_day.py             # dry-run
  uv run python scripts/migrate_date_day.py --apply     # 实际写库
"""

from __future__ import annotations

import argparse
from collections import Counter
from typing import Any, Mapping, Protocol

import chromadb

from src.schemas.constants import (
    CHROMA_COLLECTION_NAME,
    CHROMA_DEFAULT_PERSIST_DIR,
    META_DATE,
    META_DATE_DAY,
    META_DOC_TYPE,
)
from src.utils.dates import parse_date_day


class MigratableCollection(Protocol):
    """迁移所需的最小 Chroma collection 接口，便于测试替身。"""

    def count(self) -> int: ...

    def get(self, *args: Any, **kwargs: Any) -> Mapping[str, Any]: ...

    def update(self, *args: Any, **kwargs: Any) -> object: ...


def _needs_backfill(metadata: dict[str, Any]) -> bool:
    return metadata.get(META_DATE_DAY) is None


def migrate_collection(
    collection: MigratableCollection,
    *,
    apply: bool,
    batch_size: int = 500,
) -> Counter[str]:
    """分批扫描 collection 并按需补写 date_day，返回计数器汇总。

    计数键：scanned / backfilled / already_had_date_day / unparseable，
    以及 unparseable|<doc_type>|<date 原值> 的豁免明细。
    """
    summary: Counter[str] = Counter()
    total = collection.count()
    offset = 0
    while offset < total:
        batch = collection.get(include=["metadatas"], limit=batch_size, offset=offset)
        metadatas = batch.get("metadatas") or []
        ids = batch.get("ids") or []
        if not metadatas:
            break
        update_ids: list[str] = []
        update_metadatas: list[dict[str, Any]] = []
        for chunk_id, metadata in zip(ids, metadatas):
            summary["scanned"] += 1
            if not _needs_backfill(metadata):
                summary["already_had_date_day"] += 1
                continue
            date_day = parse_date_day(metadata.get(META_DATE))
            if date_day is None:
                summary["unparseable"] += 1
                doc_type = metadata.get(META_DOC_TYPE, "unknown")
                summary[f"unparseable|{doc_type}|{metadata.get(META_DATE)!r}"] += 1
                continue
            update_ids.append(str(chunk_id))
            update_metadatas.append({**metadata, META_DATE_DAY: date_day})
        if apply and update_ids:
            collection.update(ids=update_ids, metadatas=update_metadatas)
            summary["backfilled"] += len(update_ids)
        elif not apply:
            summary["backfilled"] += len(update_ids)
        offset += len(metadatas)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "migrate").splitlines()[0])
    parser.add_argument("--apply", action="store_true", help="实际写库（默认 dry-run）")
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--persist-dir", default=CHROMA_DEFAULT_PERSIST_DIR)
    parser.add_argument("--collection", default=CHROMA_COLLECTION_NAME)
    args = parser.parse_args()

    client = chromadb.PersistentClient(path=args.persist_dir)
    collection = client.get_collection(args.collection)
    summary = migrate_collection(collection, apply=args.apply, batch_size=args.batch_size)

    print(f"collection={args.collection} apply={args.apply}")
    print(
        f"scanned={summary['scanned']} backfilled={summary['backfilled']} "
        f"already_had_date_day={summary['already_had_date_day']} "
        f"unparseable={summary['unparseable']}"
    )
    for key, count in sorted(summary.items()):
        if key.startswith("unparseable|"):
            print(f"  豁免 {key.removeprefix('unparseable|')}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
