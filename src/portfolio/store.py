"""SQLite portfolio persistence scoped per user.

Every read and write is filtered by `user_id`, and a foreign position is
reported as missing rather than forbidden so a caller cannot probe which
positions exist for another user.

Storage follows the existing conversation store conventions: snake_case plural
tables, ISO-8601 `*_at` text timestamps, soft delete via `deleted_at`. The
database path is injected by the caller, keeping this layer free of config.
"""

from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

from src.portfolio.models import (
    POSITION_SIDE_LONG,
    POSITION_SOURCE_MANUAL,
    POSITION_STATUS_ACTIVE,
    POSITION_STATUS_DELETED,
    VALID_POSITION_SIDES,
    VALID_POSITION_SOURCES,
    PortfolioPositionDict,
    PortfolioPositionUpdateDict,
)


class PortfolioNotFoundError(LookupError):
    """Raised when a position is missing, deleted, or owned by another user."""


class DuplicatePositionError(ValueError):
    """Raised when the user already tracks that symbol on the same side."""


class InvalidPositionError(ValueError):
    """Raised for unknown side/source values or out-of-range weights."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SQLitePortfolioStore:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)

    def add_position(
        self,
        *,
        user_id: str,
        stock_code: str,
        stock_name: str = "",
        position_side: str = POSITION_SIDE_LONG,
        weight: float = 0.0,
        source: str = POSITION_SOURCE_MANUAL,
        note: str = "",
    ) -> PortfolioPositionDict:
        """Add a position, reviving a previously removed one instead of duplicating."""
        _validate_side(position_side)
        _validate_source(source)
        _validate_weight(weight)
        code = stock_code.strip()
        if not code:
            raise InvalidPositionError("stock_code is required")

        now = utc_now()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(self.db_path), timeout=5) as conn:
            conn.row_factory = sqlite3.Row
            self._ensure_schema(conn)
            existing = self._find_any_row(
                conn, user_id=user_id, stock_code=code, position_side=position_side
            )
            if existing is not None and existing["status"] == POSITION_STATUS_ACTIVE:
                raise DuplicatePositionError(f"{user_id} already tracks {code} as {position_side}")
            if existing is not None:
                conn.execute(
                    """
                    UPDATE portfolio_positions
                    SET stock_name = ?, weight = ?, source = ?, note = ?,
                        status = ?, updated_at = ?, deleted_at = NULL
                    WHERE position_id = ? AND user_id = ?
                    """,
                    (
                        stock_name.strip(),
                        weight,
                        source,
                        note.strip(),
                        POSITION_STATUS_ACTIVE,
                        now,
                        existing["position_id"],
                        user_id,
                    ),
                )
                return self._to_dict(
                    self._find_row(
                        conn,
                        position_id=existing["position_id"],
                        user_id=user_id,
                        allow_deleted=True,
                    )
                )

            position_id = str(uuid.uuid4())
            conn.execute(
                """
                INSERT INTO portfolio_positions (
                    position_id, user_id, stock_code, stock_name, position_side,
                    weight, source, note, status, created_at, updated_at, deleted_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
                """,
                (
                    position_id,
                    user_id,
                    code,
                    stock_name.strip(),
                    position_side,
                    weight,
                    source,
                    note.strip(),
                    POSITION_STATUS_ACTIVE,
                    now,
                    now,
                ),
            )
        return self.get_position(position_id=position_id, user_id=user_id)

    def get_position(self, *, position_id: str, user_id: str) -> PortfolioPositionDict:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(self.db_path), timeout=5) as conn:
            conn.row_factory = sqlite3.Row
            self._ensure_schema(conn)
            row = self._find_row(conn, position_id=position_id, user_id=user_id)
        return self._to_dict(row)

    def list_positions(
        self,
        *,
        user_id: str,
        position_side: str | None = None,
        include_deleted: bool = False,
    ) -> list[PortfolioPositionDict]:
        """List one user's positions, optionally narrowed to holding or watchlist."""
        if position_side is not None:
            _validate_side(position_side)
        clauses = ["user_id = ?"]
        params: list[object] = [user_id]
        if position_side is not None:
            clauses.append("position_side = ?")
            params.append(position_side)
        if not include_deleted:
            clauses.append("deleted_at IS NULL")
        where = " AND ".join(clauses)

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(self.db_path), timeout=5) as conn:
            conn.row_factory = sqlite3.Row
            self._ensure_schema(conn)
            rows = conn.execute(
                f"""
                SELECT * FROM portfolio_positions
                WHERE {where}
                ORDER BY weight DESC, created_at ASC
                """,
                params,
            ).fetchall()
        return [self._to_dict(row) for row in rows]

    def list_symbols_for_user(self, *, user_id: str, position_side: str | None = None) -> list[str]:
        """Distinct active symbols for one user, for downstream daily scans."""
        positions = self.list_positions(user_id=user_id, position_side=position_side)
        symbols = {item["stock_code"] for item in positions if item.get("stock_code")}
        return sorted(symbols)

    def update_position(
        self,
        *,
        position_id: str,
        user_id: str,
        changes: PortfolioPositionUpdateDict,
    ) -> PortfolioPositionDict:
        """Apply a partial update; only provided keys are touched."""
        assignments: list[str] = []
        params: list[object] = []
        if changes.get("stock_name") is not None:
            assignments.append("stock_name = ?")
            params.append(str(changes["stock_name"]).strip())
        if changes.get("weight") is not None:
            _validate_weight(float(changes["weight"]))  # type: ignore[arg-type]
            assignments.append("weight = ?")
            params.append(float(changes["weight"]))  # type: ignore[arg-type]
        if changes.get("note") is not None:
            assignments.append("note = ?")
            params.append(str(changes["note"]).strip())
        if changes.get("position_side") is not None:
            _validate_side(str(changes["position_side"]))
            assignments.append("position_side = ?")
            params.append(str(changes["position_side"]))
        if not assignments:
            return self.get_position(position_id=position_id, user_id=user_id)

        assignments.append("updated_at = ?")
        params.append(utc_now())
        params.extend([position_id, user_id])

        with sqlite3.connect(str(self.db_path), timeout=5) as conn:
            conn.row_factory = sqlite3.Row
            self._ensure_schema(conn)
            self._find_row(conn, position_id=position_id, user_id=user_id)
            conn.execute(
                f"""
                UPDATE portfolio_positions SET {", ".join(assignments)}
                WHERE position_id = ? AND user_id = ? AND deleted_at IS NULL
                """,
                params,
            )
        return self.get_position(position_id=position_id, user_id=user_id)

    def remove_position(self, *, position_id: str, user_id: str) -> PortfolioPositionDict:
        """Soft-delete a position and return its final state."""
        now = utc_now()
        with sqlite3.connect(str(self.db_path), timeout=5) as conn:
            conn.row_factory = sqlite3.Row
            self._ensure_schema(conn)
            row = self._find_row(conn, position_id=position_id, user_id=user_id)
            conn.execute(
                """
                UPDATE portfolio_positions
                SET status = ?, deleted_at = ?, updated_at = ?
                WHERE position_id = ? AND user_id = ?
                """,
                (POSITION_STATUS_DELETED, now, now, position_id, user_id),
            )
        return self._to_dict(row)

    def _find_row(
        self,
        conn: sqlite3.Connection,
        *,
        position_id: str,
        user_id: str,
        allow_deleted: bool = False,
    ) -> sqlite3.Row:
        sql = "SELECT * FROM portfolio_positions WHERE position_id = ? AND user_id = ?"
        if not allow_deleted:
            sql += " AND deleted_at IS NULL"
        row = conn.execute(sql, (position_id, user_id)).fetchone()
        if row is None:
            raise PortfolioNotFoundError(f"position not found or not accessible: {position_id}")
        return row

    def _find_any_row(
        self, conn: sqlite3.Connection, *, user_id: str, stock_code: str, position_side: str
    ) -> sqlite3.Row | None:
        return conn.execute(
            """
            SELECT * FROM portfolio_positions
            WHERE user_id = ? AND stock_code = ? AND position_side = ?
            """,
            (user_id, stock_code, position_side),
        ).fetchone()

    def _to_dict(self, row: sqlite3.Row) -> PortfolioPositionDict:
        return cast(PortfolioPositionDict, dict(row))

    def _ensure_schema(self, conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS portfolio_positions (
                position_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                stock_code TEXT NOT NULL,
                stock_name TEXT NOT NULL,
                position_side TEXT NOT NULL,
                weight REAL NOT NULL DEFAULT 0,
                source TEXT NOT NULL,
                note TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                deleted_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_positions_user_side
                ON portfolio_positions(user_id, position_side, deleted_at);
            CREATE INDEX IF NOT EXISTS idx_positions_user_symbol
                ON portfolio_positions(user_id, stock_code);
            """
        )


def _validate_side(value: str) -> None:
    if value not in VALID_POSITION_SIDES:
        raise InvalidPositionError(f"unsupported position_side: {value}")


def _validate_source(value: str) -> None:
    if value not in VALID_POSITION_SOURCES:
        raise InvalidPositionError(f"unsupported source: {value}")


def _validate_weight(value: float) -> None:
    weight = float(value)
    if weight < 0 or weight > 100:
        raise InvalidPositionError(f"weight must be within [0, 100], got {weight}")
