"""Portfolio domain types shared across ingestion, jobs and API layers."""

from __future__ import annotations

from typing_extensions import TypedDict

POSITION_SIDE_LONG = "long"
POSITION_SIDE_WATCH = "watch"
VALID_POSITION_SIDES = (POSITION_SIDE_LONG, POSITION_SIDE_WATCH)

POSITION_STATUS_ACTIVE = "active"
POSITION_STATUS_DELETED = "deleted"

POSITION_SOURCE_MANUAL = "manual"
POSITION_SOURCE_IMPORT = "import"
VALID_POSITION_SOURCES = (POSITION_SOURCE_MANUAL, POSITION_SOURCE_IMPORT)


class PortfolioPositionDict(TypedDict, total=False):
    """A single holding or watchlist entry owned by exactly one user.

    `position_side` separates real holdings from watchlist entries so the two
    share one storage table instead of drifting into duplicated schemas.
    """

    position_id: str
    user_id: str
    stock_code: str
    stock_name: str
    position_side: str
    weight: float
    source: str
    note: str
    status: str
    created_at: str
    updated_at: str
    deleted_at: str | None


class PortfolioPositionUpdateDict(TypedDict, total=False):
    """Explicit partial update payload; `None` means "leave unchanged"."""

    stock_name: str | None
    weight: float | None
    note: str | None
    position_side: str | None
