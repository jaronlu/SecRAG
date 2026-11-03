"""产品 profile 范围：默认角色、知识源白名单与工具白名单。

knowledge_qa 是默认产品形态：只向模型暴露产品文档与 FAQ 检索。
full profile 保留完整 Agent/工具矩阵，用于开发、评估与回退。

本模块保持零业务依赖（不导入 Agent、Retriever 或前端模块），
服务端代码与测试都可以直接解析 profile 范围而不会形成循环依赖。
"""

from typing import Final, Literal, cast

from src.schemas.constants import (
    ROLE_OPERATIONS,
    SOURCE_FAQ,
    SOURCE_PRODUCT,
)

ProductProfile = Literal["knowledge_qa", "full"]

# 收敛后产品的试点受众：运营/支持人员
MVP_PRIMARY_ROLE: Final[str] = ROLE_OPERATIONS

# profile → 允许的检索源。None 表示"无产品层限制"；
# 角色层权限（ROLE_ALLOWED_SOURCES）仍独立执行。
PROFILE_ALLOWED_SOURCES: dict[ProductProfile, frozenset[str] | None] = {
    "knowledge_qa": frozenset({SOURCE_PRODUCT, SOURCE_FAQ}),
    "full": None,
}

# profile → 可执行的工具名（与 src/agents/tools.py 的 @tool 函数名一致）。
# None 表示 full profile 保留完整工具注册表。
PROFILE_ALLOWED_TOOLS: dict[ProductProfile, frozenset[str] | None] = {
    "knowledge_qa": frozenset({"product_search", "faq_search"}),
    "full": None,
}


def get_product_profile(profile: str | None = None) -> ProductProfile:
    """解析当前生效的产品 profile，非法值 fail closed。

    无参数时读取 ``config.product_profile``（pydantic 启动时已校验）。
    显式传入未知值时抛 ValueError，而不是静默放宽到 ``full``。
    """
    if profile is None:
        from src.config import config

        profile = config.product_profile
    if profile not in PROFILE_ALLOWED_SOURCES:
        raise ValueError(
            f"Unknown product profile: {profile!r}. "
            f"Expected one of: {sorted(PROFILE_ALLOWED_SOURCES)}"
        )
    return cast(ProductProfile, profile)


def allowed_sources_for_profile(profile: ProductProfile) -> frozenset[str] | None:
    """返回该 profile 可规划的检索源（None = 不限制）。"""
    return PROFILE_ALLOWED_SOURCES[get_product_profile(profile)]


def allowed_tools_for_profile(profile: ProductProfile) -> frozenset[str] | None:
    """返回该 profile 可执行的工具（None = 不限制）。"""
    return PROFILE_ALLOWED_TOOLS[get_product_profile(profile)]
