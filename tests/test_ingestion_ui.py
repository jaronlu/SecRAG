from src.api.ui import render_ui_html


def test_ui_contains_ingestion_tab_and_server_paths():
    html = render_ui_html()

    assert 'id="ingestionTab"' in html
    assert 'id="ingestionCategory"' in html
    assert 'id="startIngestion"' in html
    assert "/v1/admin/ingestion/categories" in html
    assert "/v1/admin/ingestion/runs" in html
    assert "__INGESTION_" not in html


def test_ui_uses_category_dropdown_without_arbitrary_path_input():
    html = render_ui_html()

    assert '<select id="ingestionCategory">' in html
    assert '<input id="ingestionPath" readonly' in html
    assert 'name="path"' not in html
    assert "full_scan" not in html
