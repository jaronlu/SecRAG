from src.api.ui import render_ui_html


def test_ui_contains_ingestion_tab_and_server_paths():
    html = render_ui_html()

    assert 'id="ingestionTab"' in html
    assert 'id="ingestionCategory"' in html
    assert 'id="startIngestion"' in html
    assert "/v1/admin/ingestion/categories" in html
    assert "/v1/admin/ingestion/runs" in html
    assert "/v1/admin/ingestion/chunks" in html
    assert 'id="chunkList"' in html
    assert 'id="loadMoreChunks"' in html
    assert "Embedding Chunks" in html
    assert "__INGESTION_" not in html


def test_ui_uses_category_dropdown_without_arbitrary_path_input():
    html = render_ui_html()

    assert '<select id="ingestionCategory">' in html
    assert '<input id="ingestionPath" readonly' in html
    assert 'name="path"' not in html
    assert "full_scan" not in html


def test_ui_renders_chunk_results_as_a_list_without_vectors():
    html = render_ui_html()

    assert '<ol id="chunkList" class="chunk-list">' in html
    assert "loadDocumentChunks" in html
    assert "embedding_model" in html
    assert "embedding_vector" not in html
    assert "无法连接服务，请确认 SecRAG 后端已启动" in html
