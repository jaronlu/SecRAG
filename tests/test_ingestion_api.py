import pytest
from httpx import ASGITransport, AsyncClient

from src.api.ingestion import _get_ingestion_service
from src.api.main import app


class FakeIngestionService:
    def __init__(self) -> None:
        self.executed: list[str] = []

    def list_categories(self):
        return (
            [
                {
                    "category_id": "financials",
                    "label": "财务数据",
                    "group": "证券数据",
                    "relative_path": "data/raw/financials",
                    "default_doc_type": "financial_data",
                    "allowed_doc_types": ["financial_data"],
                    "file_count": 1,
                    "manifest_count": 1,
                    "invalid_manifest_count": 0,
                    "ready": True,
                    "error_code": "",
                    "error": "",
                }
            ],
            None,
        )

    def list_category_files(self, category_id: str):
        assert category_id == "financials"
        return [
            {
                "relative_path": "data/raw/financials/sample.csv",
                "extension": ".csv",
                "doc_type": "financial_data",
                "permission_level": "internal",
                "allowed_roles": ["technical"],
                "manifest_status": "valid",
                "error": "",
            }
        ]

    def create_run(self, category_id: str, *, requested_by: str, executor: str = "api"):
        assert category_id == "financials"
        assert requested_by == "user_tech"
        assert executor == "api"
        return self._summary("queued")

    def execute_run(self, run_id: str):
        self.executed.append(run_id)
        return self._summary("success")

    def get_run(self, run_id: str):
        assert run_id == "run-1"
        return self._summary("running")

    def list_run_items(self, run_id: str):
        assert run_id == "run-1"
        return [
            {
                "doc_id": "doc-1",
                "sequence": 1,
                "relative_path": "data/raw/financials/sample.csv",
                "action": "created",
                "chunk_count": 2,
                "processed_at": "2026-07-14T08:00:01+00:00",
                "error_code": "",
                "error": "",
            }
        ]

    def list_document_chunks(self, doc_id: str, *, offset: int, limit: int):
        assert doc_id == "doc-1"
        assert offset == 0
        assert limit == 50
        return (
            [
                {
                    "chunk_id": "chunk-1",
                    "chunk_index": 0,
                    "chunk_hash": "hash-1",
                    "doc_type": "financial_data",
                    "title": "sample",
                    "stock_code": "600519",
                    "date": "2026",
                    "page_number": "",
                    "content_length": 6,
                    "content": "stored",
                    "permission_level": "internal",
                    "allowed_roles": ["technical"],
                    "parser_version": "parser-v1",
                    "chunker_version": "chunker-v1",
                    "embedding_model": "embedding-v1",
                }
            ],
            1,
        )

    def list_recent_runs(self, limit: int):
        assert limit == 10
        return [self._summary("success")]

    @staticmethod
    def _summary(status: str):
        return {
            "run_id": "run-1",
            "category_id": "financials",
            "status": status,
            "queued_at": "2026-07-14T08:00:00+00:00",
            "started_at": None if status == "queued" else "2026-07-14T08:00:01+00:00",
            "finished_at": "2026-07-14T08:00:02+00:00" if status == "success" else None,
            "total_files": 1,
            "processed_files": 1 if status == "success" else 0,
            "created": 1 if status == "success" else 0,
            "replaced": 0,
            "skipped": 0,
            "archived": 0,
            "failed": 0,
            "error_code": "",
            "error": "",
        }


@pytest.mark.asyncio
async def test_ingestion_routes_require_technical_role():
    fake = FakeIngestionService()
    app.dependency_overrides[_get_ingestion_service] = lambda: fake
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get(
                "/v1/admin/ingestion/categories",
                headers={"Authorization": "Bearer demo-advisor"},
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 403


@pytest.mark.asyncio
async def test_ingestion_catalog_files_and_runs_api():
    fake = FakeIngestionService()
    app.dependency_overrides[_get_ingestion_service] = lambda: fake
    headers = {"Authorization": "Bearer demo-tech"}
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            categories = await client.get("/v1/admin/ingestion/categories", headers=headers)
            files = await client.get(
                "/v1/admin/ingestion/categories/financials/files", headers=headers
            )
            created = await client.post(
                "/v1/admin/ingestion/runs",
                headers=headers,
                json={"category_id": "financials"},
            )
            run = await client.get("/v1/admin/ingestion/runs/run-1", headers=headers)
            items = await client.get("/v1/admin/ingestion/runs/run-1/items", headers=headers)
            chunks = await client.get(
                "/v1/admin/ingestion/chunks?doc_id=doc-1", headers=headers
            )
            recent = await client.get("/v1/admin/ingestion/runs?limit=10", headers=headers)
    finally:
        app.dependency_overrides.clear()

    assert categories.status_code == 200
    assert categories.json()["categories"][0]["category_id"] == "financials"
    assert files.status_code == 200
    assert files.json()["files"][0]["manifest_status"] == "valid"
    assert created.status_code == 202
    assert created.json()["status"] == "queued"
    assert fake.executed == ["run-1"]
    assert run.status_code == 200
    assert items.json()["items"][0]["relative_path"].startswith("data/raw/")
    assert chunks.status_code == 200
    assert chunks.json()["chunks"][0]["content"] == "stored"
    assert "source" not in chunks.json()["chunks"][0]
    assert "embedding" not in chunks.json()["chunks"][0]
    assert recent.status_code == 200


@pytest.mark.asyncio
async def test_create_ingestion_run_rejects_extra_path_field():
    fake = FakeIngestionService()
    app.dependency_overrides[_get_ingestion_service] = lambda: fake
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/admin/ingestion/runs",
                headers={"Authorization": "Bearer demo-tech"},
                json={"category_id": "financials", "path": "/tmp"},
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 422
