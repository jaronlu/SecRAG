"""当前认证上下文接口：向问答前端暴露服务端派生的最小身份信息。"""

from fastapi import APIRouter, Depends

from src.api.auth import AuthenticatedUser, authenticate_user
from src.config import config
from src.schemas.constants import API_ROUTE_ASSISTANT_ME
from src.schemas.request_response import CurrentUserResponse

router = APIRouter()


@router.get(API_ROUTE_ASSISTANT_ME, response_model=CurrentUserResponse)
async def current_user(user: AuthenticatedUser = Depends(authenticate_user)) -> CurrentUserResponse:
    """角色/部门由 Bearer token 服务端派生，不信任请求体中的任何身份字段。"""
    return CurrentUserResponse(
        role=user.role,
        department=user.department,
        product_profile=config.product_profile,
    )
