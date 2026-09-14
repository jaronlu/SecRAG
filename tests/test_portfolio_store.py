"""Portfolio store tests covering CRUD and per-user isolation.

Every test uses a throwaway SQLite file so no shared state leaks between cases.
"""

from __future__ import annotations

import sqlite3

import pytest

from src.portfolio.models import (
    POSITION_SIDE_LONG,
    POSITION_SIDE_WATCH,
    PortfolioPositionUpdateDict,
)
from src.portfolio.store import (
    DuplicatePositionError,
    InvalidPositionError,
    PortfolioNotFoundError,
    SQLitePortfolioStore,
)


@pytest.fixture()
def store(tmp_path) -> SQLitePortfolioStore:
    return SQLitePortfolioStore(tmp_path / "portfolio.db")


def test_add_and_get_position_roundtrip(store):
    created = store.add_position(
        user_id="u1",
        stock_code="600519",
        stock_name="贵州茅台",
        position_side=POSITION_SIDE_LONG,
        weight=12.5,
    )

    loaded = store.get_position(position_id=created["position_id"], user_id="u1")
    assert loaded["stock_code"] == "600519"
    assert loaded["stock_name"] == "贵州茅台"
    assert loaded["weight"] == 12.5
    assert loaded["status"] == "active"
    assert loaded["deleted_at"] is None
    assert loaded["created_at"] and loaded["updated_at"]


def test_update_position_applies_partial_changes(store):
    position = store.add_position(user_id="u1", stock_code="000001", weight=5.0)

    updated = store.update_position(
        position_id=position["position_id"],
        user_id="u1",
        changes=PortfolioPositionUpdateDict(weight=7.5, note="加仓"),
    )

    assert updated["weight"] == 7.5
    assert updated["note"] == "加仓"
    assert updated["stock_code"] == "000001"


def test_cannot_read_another_users_position(store):
    position = store.add_position(user_id="u1", stock_code="600519")

    with pytest.raises(PortfolioNotFoundError):
        store.get_position(position_id=position["position_id"], user_id="u2")


def test_cannot_update_another_users_position(store):
    position = store.add_position(user_id="u1", stock_code="600519", weight=5.0)

    with pytest.raises(PortfolioNotFoundError):
        store.update_position(
            position_id=position["position_id"],
            user_id="u2",
            changes=PortfolioPositionUpdateDict(weight=99.0),
        )
    unchanged = store.get_position(position_id=position["position_id"], user_id="u1")
    assert unchanged["weight"] == 5.0


def test_cannot_remove_another_users_position(store):
    position = store.add_position(user_id="u1", stock_code="600519")

    with pytest.raises(PortfolioNotFoundError):
        store.remove_position(position_id=position["position_id"], user_id="u2")
    assert store.list_positions(user_id="u1")


def test_list_positions_only_returns_own_rows(store):
    store.add_position(user_id="u1", stock_code="600519", weight=10.0)
    store.add_position(user_id="u2", stock_code="000001", weight=20.0)

    mine = store.list_positions(user_id="u1")
    assert [item["stock_code"] for item in mine] == ["600519"]


def test_soft_delete_hides_position_then_readd_revives_it(store):
    position = store.add_position(user_id="u1", stock_code="600519")
    store.remove_position(position_id=position["position_id"], user_id="u1")

    assert store.list_positions(user_id="u1") == []

    revived = store.add_position(user_id="u1", stock_code="600519", weight=3.0)
    assert revived["position_id"] == position["position_id"]
    assert revived["deleted_at"] is None
    assert revived["weight"] == 3.0


def test_duplicate_active_position_is_rejected(store):
    store.add_position(user_id="u1", stock_code="600519", position_side=POSITION_SIDE_LONG)

    with pytest.raises(DuplicatePositionError):
        store.add_position(user_id="u1", stock_code="600519", position_side=POSITION_SIDE_LONG)


def test_same_symbol_allowed_on_different_sides(store):
    store.add_position(user_id="u1", stock_code="600519", position_side=POSITION_SIDE_LONG)

    watch = store.add_position(user_id="u1", stock_code="600519", position_side=POSITION_SIDE_WATCH)

    assert watch["position_side"] == POSITION_SIDE_WATCH
    assert len(store.list_positions(user_id="u1")) == 2


def test_list_positions_filters_by_side(store):
    store.add_position(user_id="u1", stock_code="600519", position_side=POSITION_SIDE_LONG)
    store.add_position(user_id="u1", stock_code="300750", position_side=POSITION_SIDE_WATCH)

    holdings = store.list_positions(user_id="u1", position_side=POSITION_SIDE_LONG)
    watchlist = store.list_positions(user_id="u1", position_side=POSITION_SIDE_WATCH)

    assert [item["stock_code"] for item in holdings] == ["600519"]
    assert [item["stock_code"] for item in watchlist] == ["300750"]


def test_list_symbols_for_user_is_scoped_and_deduped(store):
    store.add_position(user_id="u1", stock_code="600519", position_side=POSITION_SIDE_LONG)
    store.add_position(user_id="u1", stock_code="600519", position_side=POSITION_SIDE_WATCH)
    store.add_position(user_id="u2", stock_code="999999", position_side=POSITION_SIDE_LONG)

    assert store.list_symbols_for_user(user_id="u1") == ["600519"]


def test_invalid_weight_and_side_are_rejected(store):
    with pytest.raises(InvalidPositionError):
        store.add_position(user_id="u1", stock_code="600519", weight=101.0)
    with pytest.raises(InvalidPositionError):
        store.add_position(user_id="u1", stock_code="600519", position_side="short")


def test_table_naming_follows_existing_convention(store):
    store.add_position(user_id="u1", stock_code="600519")

    with sqlite3.connect(str(store.db_path)) as conn:
        tables = [
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
            )
        ]

    assert "portfolio_positions" in tables
    for table in tables:
        assert table == table.lower(), f"{table} is not snake_case"
