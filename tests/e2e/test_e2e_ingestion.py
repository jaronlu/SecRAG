"""全链路测试案例：环节 B 文档上传 → 解析/分块 → 向量化入库（TC-003~TC-010）。

入库链路使用注入式 IngestionService：registry/ChromaDB/分类目录全部落到
tmp_path，embedding 用确定性假模型，不下载真实模型、不碰生产数据。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.ingestion.service import (
    CategoryPreflightError,
    IngestionService,
)
from src.schemas.constants import (
    ALL_VALID_DOC_TYPES,
    DOC_TYPE_RESEARCH_REPORT,
    META_ALLOWED_ROLES,
    META_PERMISSION_LEVEL,
    ROLE_ADVISOR,
    ROLE_COMPLIANCE,
    ROLE_OPERATIONS,
    ROLE_TECHNICAL,
)

FAKE_EMBED_MODEL = "fake-embed-e2e"

FUND_REPORT_HTML = """<html><body>
<h1>XX货币市场基金2024年年度报告（摘要）</h1>
<p>根据基金合同与2024年年度报告，本基金风险等级为R1（低风险），适合保守型投资者。</p>
<p>基金主要投资于货币市场工具，不投资股票或可转换债券。</p>
<p>报告期内基金收益率为1.8%，规模为120亿元。</p>
</body></html>"""

VALID_META = {
    "doc_type": DOC_TYPE_RESEARCH_REPORT,
    META_PERMISSION_LEVEL: "public",
    META_ALLOWED_ROLES: [ROLE_ADVISOR, ROLE_COMPLIANCE, ROLE_OPERATIONS, ROLE_TECHNICAL],
}


class FakeEmbeddings:
    """确定性假 embedding：只满足 Chroma 接口，不做真实语义。"""

    model_name = FAKE_EMBED_MODEL

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(min(len(t), 64)), 1.0, 0.0] for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(min(len(text), 64)), 1.0, 0.0]


def write_document(
    category_dir: Path,
    name: str,
    content: bytes | str,
    meta: dict[str, Any] | None,
) -> Path:
    file_path = category_dir / name
    if isinstance(content, str):
        content = content.encode("utf-8")
    file_path.write_bytes(content)
    if meta is not None:
        file_path.with_name(f"{file_path.name}.meta.json").write_text(
            json.dumps(meta, ensure_ascii=False), encoding="utf-8"
        )
    return file_path


@pytest.fixture()
def ingestion_env(tmp_path):
    """隔离的入库环境：返回 (service, category_dir, helper)。"""
    category_dir = tmp_path / "data" / "raw" / "reports"
    category_dir.mkdir(parents=True)
    category = {
        "category_id": "reports",
        "label": "财报公告",
        "group": "reports",
        "relative_path": "data/raw/reports",
        "default_doc_type": DOC_TYPE_RESEARCH_REPORT,
        "allowed_doc_types": sorted(ALL_VALID_DOC_TYPES),
    }
    service = IngestionService(
        project_root=tmp_path,
        registry_path=tmp_path / "registry.db",
        persist_directory=str(tmp_path / "chroma"),
        catalog=(category,),
        embedding_model_factory=lambda model: FakeEmbeddings(),
    )

    def run(category_id: str = "reports"):
        queued = service.create_run(category_id, requested_by="tester")
        return service.execute_run(queued["run_id"]), queued["run_id"]

    return type(
        "Env",
        (),
        {
            "service": service,
            "category_dir": category_dir,
            "tmp_path": tmp_path,
            "run": staticmethod(run),
        },
    )()


def chunk_ids_for(env, doc_id: str) -> list[str]:
    from src.ingestion.embedder import list_chunk_ids_by_doc_id

    return list_chunk_ids_by_doc_id(
        doc_id=doc_id,
        persist_directory=env.service.persist_directory,
        embedding_model=FakeEmbeddings(),
    )


# ══════════════════════════════════════════════════════════════════════
# TC-003 正常入库主流程
# ══════════════════════════════════════════════════════════════════════


def test_tc003_ingest_financial_report_end_to_end(ingestion_env):
    """TC-003：财报 HTML + public 清单 → created，chunk 落向量库，registry active。"""
    env = ingestion_env
    write_document(env.category_dir, "xx_money_fund_2024.html", FUND_REPORT_HTML, VALID_META)

    summary, run_id = env.run()

    assert summary["status"] == "success"
    items = env.service.list_run_items(run_id)
    assert len(items) == 1
    assert items[0]["action"] == "created"
    assert items[0]["chunk_count"] >= 1

    doc = env.service.registry.get_document(items[0]["doc_id"])
    assert doc is not None and doc.status == "active"
    assert doc.chunk_count == items[0]["chunk_count"]

    stored_ids = chunk_ids_for(env, items[0]["doc_id"])
    assert len(stored_ids) == items[0]["chunk_count"]
