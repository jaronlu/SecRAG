from html import escape
from pathlib import Path

from src.api.auth import TOKEN_USER_BINDINGS
from src.schemas.constants import (
    API_ROUTE_ASSISTANT_QA,
    API_ROUTE_ASSISTANT_THREADS,
    ROLE_ADVISOR,
    ROLE_COMPLIANCE,
    ROLE_INSTITUTIONAL_SALES,
    ROLE_OPERATIONS,
    ROLE_TECHNICAL,
)


ROLE_UI_OPTIONS = {
    ROLE_TECHNICAL: "技术支持",
    ROLE_ADVISOR: "投顾",
    ROLE_INSTITUTIONAL_SALES: "机构销售",
    ROLE_COMPLIANCE: "合规",
    ROLE_OPERATIONS: "运营",
}

UI_TEMPLATE_PATH = Path(__file__).with_name("ui.html")


def _render_identity_options() -> str:
    options = []
    token_by_role = {}
    duplicate_roles = set()
    for token, user in TOKEN_USER_BINDINGS.items():
        if user.role in token_by_role:
            duplicate_roles.add(user.role)
        token_by_role[user.role] = token

    missing_roles = [role for role in ROLE_UI_OPTIONS if role not in token_by_role]
    extra_roles = [role for role in token_by_role if role not in ROLE_UI_OPTIONS]
    if duplicate_roles or missing_roles or extra_roles:
        raise RuntimeError(
            "UI role options and demo token roles are inconsistent: "
            f"duplicates={sorted(duplicate_roles)}, missing={missing_roles}, extra={extra_roles}"
        )

    for role, label in ROLE_UI_OPTIONS.items():
        token = token_by_role[role]
        options.append(
            '<option value="{token}" data-role="{role}">{label}</option>'.format(
                token=escape(token),
                role=escape(role),
                label=escape(label),
            )
        )
    return "\n".join(options)


def render_ui_html() -> str:
    return UI_TEMPLATE_PATH.read_text(encoding="utf-8").replace(
        "__IDENTITY_OPTIONS__",
        _render_identity_options(),
    ).replace(
        "__ASSISTANT_QA_PATH__",
        API_ROUTE_ASSISTANT_QA,
    ).replace(
        "__THREADS_PATH__",
        API_ROUTE_ASSISTANT_THREADS,
    )
