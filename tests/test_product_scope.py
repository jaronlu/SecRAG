"""产品 profile 行为：默认范围、允许的检索源/工具、非法值 fail closed。"""

from src.product_scope import (
    PROFILE_ALLOWED_SOURCES,
    PROFILE_ALLOWED_TOOLS,
    MVP_PRIMARY_ROLE,
    get_product_profile,
)
from src.schemas.constants import (
    ROLE_OPERATIONS,
    SOURCE_FAQ,
    SOURCE_PRODUCT,
)


def test_knowledge_qa_profile_allows_only_curated_sources():
    from src.product_scope import allowed_sources_for_profile

    assert allowed_sources_for_profile("knowledge_qa") == {SOURCE_FAQ, SOURCE_PRODUCT}


def test_full_profile_has_no_product_source_restriction():
    from src.product_scope import allowed_sources_for_profile

    assert allowed_sources_for_profile("full") is None


def test_unknown_profile_fails_closed():
    import pytest
    from src.product_scope import get_product_profile

    with pytest.raises(ValueError):
        get_product_profile("unknown")


def test_mvp_primary_role_is_operations():
    assert MVP_PRIMARY_ROLE == ROLE_OPERATIONS


def test_knowledge_qa_profile_whitelists_only_knowledge_tools():
    from src.product_scope import allowed_tools_for_profile

    assert allowed_tools_for_profile("knowledge_qa") == {"product_search", "faq_search"}


def test_full_profile_has_no_tool_restriction():
    from src.product_scope import allowed_tools_for_profile

    assert allowed_tools_for_profile("full") is None


def test_profile_maps_cover_all_declared_profiles():
    from src.product_scope import ProductProfile

    assert set(PROFILE_ALLOWED_SOURCES) == set(PROFILE_ALLOWED_TOOLS)
    for profile in ProductProfile.__args__:
        assert profile in PROFILE_ALLOWED_SOURCES
        assert profile in PROFILE_ALLOWED_TOOLS


def test_get_product_profile_without_argument_reads_config():
    from src.config import config

    assert get_product_profile() == config.product_profile
