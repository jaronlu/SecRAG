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


# ══════════════════════════════════════════════════════════════════════
# TC-004 重复入库幂等跳过
# ══════════════════════════════════════════════════════════════════════


def test_tc004_reingest_unchanged_document_is_skipped(ingestion_env):
    """TC-004：内容/清单/解析器/分块器/模型均未变时重跑入库 → skipped，无重复 chunk。"""
    env = ingestion_env
    write_document(env.category_dir, "xx_money_fund_2024.html", FUND_REPORT_HTML, VALID_META)

    first_summary, first_run_id = env.run()
    assert first_summary["status"] == "success"
    doc_id = env.service.list_run_items(first_run_id)[0]["doc_id"]
    chunk_count = len(chunk_ids_for(env, doc_id))

    second_summary, second_run_id = env.run()

    assert second_summary["status"] == "success"
    second_items = env.service.list_run_items(second_run_id)
    assert len(second_items) == 1
    assert second_items[0]["action"] == "skipped"
    assert len(chunk_ids_for(env, doc_id)) == chunk_count


# ══════════════════════════════════════════════════════════════════════
# TC-005 文档更新替换
# ══════════════════════════════════════════════════════════════════════


def test_tc005_updated_document_replaces_old_chunks(ingestion_env):
    """TC-005：正文变化 → replaced，doc_version+1，旧 chunk 被清理不残留。"""
    env = ingestion_env
    file_path = write_document(
        env.category_dir, "xx_money_fund_2024.html", FUND_REPORT_HTML, VALID_META
    )
    _, first_run_id = env.run()
    doc_id = env.service.list_run_items(first_run_id)[0]["doc_id"]
    old_version = env.service.registry.get_document(doc_id).doc_version
    old_ids = set(chunk_ids_for(env, doc_id))

    file_path.write_text(
        FUND_REPORT_HTML.replace(
            "</body>", "<p>新增：基金分红条款为每日分红，月末集中支付。</p></body>"
        ),
        encoding="utf-8",
    )
    second_summary, second_run_id = env.run()

    assert second_summary["status"] == "success"
    second_items = env.service.list_run_items(second_run_id)
    assert second_items[0]["action"] == "replaced"
    doc = env.service.registry.get_document(doc_id)
    assert doc.doc_version == old_version + 1

    new_ids = set(chunk_ids_for(env, doc_id))
    assert new_ids and new_ids != old_ids
    removed = old_ids - new_ids
    if removed:
        from src.ingestion.embedder import get_vectorstore

        vs = get_vectorstore(
            persist_directory=env.service.persist_directory, embedding_model=FakeEmbeddings()
        )
        remaining = set(vs.get(ids=sorted(removed))["ids"])
        assert remaining.isdisjoint(removed), "旧 chunk 必须被清理"


# ══════════════════════════════════════════════════════════════════════
# TC-006 空文档
# ══════════════════════════════════════════════════════════════════════


def test_tc006_empty_document_fails_without_vector_writes(ingestion_env):
    """TC-006：解析结果为空的 HTML → failed + document_processing_failed，不写向量库。"""
    env = ingestion_env
    write_document(
        env.category_dir, "empty_report.html", "<html><body></body></html>", VALID_META
    )

    summary, run_id = env.run()

    assert summary["status"] == "failed"
    items = env.service.list_run_items(run_id)
    assert len(items) == 1
    assert items[0]["action"] == "failed"
    assert items[0]["error_code"] == "document_processing_failed"
    assert items[0]["error"] == "文档处理失败"

    doc = env.service.registry.get_document(items[0]["doc_id"])
    assert doc is not None and doc.status != "active"
    assert chunk_ids_for(env, items[0]["doc_id"]) == []


# ══════════════════════════════════════════════════════════════════════
# TC-007 损坏文件
# ══════════════════════════════════════════════════════════════════════


def test_tc007_corrupt_pdf_fails_while_batch_continues(ingestion_env):
    """TC-007：伪 PDF failed，同批合法 HTML 仍 created——逐文件容错互不影响。"""
    env = ingestion_env
    write_document(
        env.category_dir, "broken_report.pdf", b"%PDF-1.4 \x00 garbage-bytes", VALID_META
    )
    write_document(env.category_dir, "good_report.html", FUND_REPORT_HTML, VALID_META)

    summary, run_id = env.run()

    assert summary["status"] == "failed"
    items = {item["relative_path"].split("/")[-1]: item for item in env.service.list_run_items(run_id)}
    assert items["broken_report.pdf"]["action"] == "failed"
    assert items["broken_report.pdf"]["error_code"] == "document_processing_failed"
    assert items["good_report.html"]["action"] == "created"

    broken_doc_id = items["broken_report.pdf"]["doc_id"]
    assert chunk_ids_for(env, broken_doc_id) == []


# ══════════════════════════════════════════════════════════════════════
# TC-008 不支持的文件格式
# ══════════════════════════════════════════════════════════════════════


def test_tc008_unsupported_suffixes_never_enter_run(ingestion_env):
    """TC-008：.txt/.zip 不被收集为业务文件，run 中不产生对应条目。"""
    env = ingestion_env
    write_document(env.category_dir, "notes.txt", "纯文本研报", VALID_META)
    write_document(env.category_dir, "bundle.zip", b"PK\x03\x04", VALID_META)

    files = env.service.list_category_files("reports")
    collected = {f["relative_path"].split("/")[-1] for f in files}
    assert "notes.txt" not in collected
    assert "bundle.zip" not in collected

    with pytest.raises(CategoryPreflightError):
        env.run()  # 分类中没有可入库业务文件


# ══════════════════════════════════════════════════════════════════════
# TC-009 缺少/非法权限清单（fail closed：预检拒绝建 run）
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("name,content,meta,desc", [
    pytest.param(
        "no_manifest.html", FUND_REPORT_HTML, None, "缺少 .meta.json", id="missing-manifest"
    ),
    pytest.param(
        "bad_permission.html",
        FUND_REPORT_HTML,
        {**VALID_META, META_PERMISSION_LEVEL: "topsecret"},
        "非法 permission_level",
        id="bad-permission",
    ),
    pytest.param(
        "bad_roles.html",
        FUND_REPORT_HTML,
        {**VALID_META, META_ALLOWED_ROLES: ["superuser"]},
        "非法 allowed_roles",
        id="bad-roles",
    ),
    pytest.param(
        "internal_no_roles.html",
        FUND_REPORT_HTML,
        {**VALID_META, META_PERMISSION_LEVEL: "internal", META_ALLOWED_ROLES: []},
        "internal 缺 allowed_roles",
        id="internal-without-roles",
    ),
])
def test_tc009_invalid_manifest_blocks_run_creation(ingestion_env, name, content, meta, desc):
    """TC-009：清单缺失或非法时 create_run 直接拒绝（fail closed），不产生 run、不写向量库。"""
    env = ingestion_env
    write_document(env.category_dir, name, content, meta)

    with pytest.raises(CategoryPreflightError) as excinfo:
        env.service.create_run("reports", requested_by="tester")

    files = {f["relative_path"].split("/")[-1]: f for f in excinfo.value.files}
    assert files[name]["manifest_status"] != "valid", f"{desc} 应被预检标记"
    assert env.service.list_recent_runs(1) == []


# ══════════════════════════════════════════════════════════════════════
# TC-010 入库管理接口权限与参数
# ══════════════════════════════════════════════════════════════════════


@pytest.fixture()
def ingestion_api_client(monkeypatch, tmp_path):
    """真实路由 + 隔离 IngestionService + 可切换角色。"""
    from fastapi.testclient import TestClient

    from src.api.auth import AuthenticatedUser, authenticate_user
    from src.api.ingestion import _get_ingestion_service
    from src.api.main import app

    category_dir = tmp_path / "data" / "raw" / "reports"
    category_dir.mkdir(parents=True)
    write_document(category_dir, "xx_money_fund_2024.html", FUND_REPORT_HTML, VALID_META)
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
    monkeypatch.setattr("src.api.ingestion._get_ingestion_service", lambda: service)

    current_user = {"role": ROLE_ADVISOR}

    def _auth_user():
        return AuthenticatedUser(f"user_{current_user['role']}", current_user["role"], "dept")

    app.dependency_overrides[authenticate_user] = _auth_user
    app.dependency_overrides[_get_ingestion_service] = lambda: service
    yield type("ClientCtx", (), {"client": TestClient(app), "role": current_user})()
    app.dependency_overrides.clear()


def test_tc010_ingestion_api_role_and_params(ingestion_api_client):
    """TC-010：非 technical 403；未知分类 404；technical 正常 200。"""
    ctx = ingestion_api_client
    API = "/v1/admin/ingestion"

    ctx.role["role"] = ROLE_ADVISOR
    res = ctx.client.get(f"{API}/categories")
    assert res.status_code == 403
    assert res.json()["detail"] == "technical role required"

    ctx.role["role"] = "technical"
    res = ctx.client.get(f"{API}/categories")
    assert res.status_code == 200
    body = res.json()
    assert [c["category_id"] for c in body["categories"]] == ["reports"]

    res = ctx.client.get(f"{API}/categories/not-a-category/files")
    assert res.status_code == 404
    assert res.json()["detail"] == "文档分类不存在"
