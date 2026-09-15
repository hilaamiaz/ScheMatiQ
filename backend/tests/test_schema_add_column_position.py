"""Tests for POST /schema/add-column/{session_id}'s optional `position` field.

The endpoint always appended a recreated column to the end of the schema.
Delete-column undo (frontend/src/pages/Workspace/SpreadsheetSurface.tsx)
needs it to insert at a specific index instead, so an undone delete restores
the column to where it originally was. Route handler is called directly with
session_manager/websocket_manager faked, matching
test_chat_messages_endpoint.py.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from app.api.routes import schema as schema_routes
from app.models.session import (
    ColumnInfo,
    DataStatistics,
    SchemaEvolution,
    SessionMetadata,
    SessionType,
    VisualizationSession,
)

SESSION_ID = "sess-position"


def _make_session(names: list[str]) -> VisualizationSession:
    columns = [ColumnInfo(name=n, definition=f"def {n}") for n in names]
    return VisualizationSession(
        id=SESSION_ID,
        type=SessionType.UPLOAD,
        metadata=SessionMetadata(source="test"),
        columns=columns,
        statistics=DataStatistics(
            total_rows=0,
            total_columns=len(columns),
            completeness=100.0,
            column_stats=[ColumnInfo(name=n, definition=f"def {n}") for n in names],
            schema_evolution=SchemaEvolution(),
        ),
    )


@pytest.fixture
def session(monkeypatch):
    """A fresh 3-column session (A, B, C), with session_manager/websocket_manager faked."""
    sess = _make_session(["A", "B", "C"])

    def _get_session(session_id: str):
        return sess if session_id == SESSION_ID else None

    async def _broadcast(*args, **kwargs):
        return None

    monkeypatch.setattr(schema_routes.session_manager, "get_session", _get_session, raising=True)
    monkeypatch.setattr(schema_routes.session_manager, "update_session", lambda s: None, raising=True)
    monkeypatch.setattr(schema_routes.websocket_manager, "broadcast_schema_updated", _broadcast, raising=True)
    return sess


def _names(sess: VisualizationSession) -> list[str]:
    return [c.name for c in sess.columns]


def _stats_names(sess: VisualizationSession) -> list[str]:
    return [c.name for c in sess.statistics.column_stats]


@pytest.mark.asyncio
async def test_no_position_still_appends(session):
    """Regression guard: the manual "Add column" dialog and the spare-row
    auto-create flow never send `position` and must keep appending."""
    await schema_routes.add_column(
        SESSION_ID, schema_routes.ColumnAddRequest(name="new", definition="d"),
    )
    assert _names(session) == ["A", "B", "C", "new"]
    assert _stats_names(session) == ["A", "B", "C", "new"]


@pytest.mark.asyncio
async def test_position_inserts_in_the_middle(session):
    await schema_routes.add_column(
        SESSION_ID, schema_routes.ColumnAddRequest(name="new", definition="d", position=1),
    )
    assert _names(session) == ["A", "new", "B", "C"]
    assert _stats_names(session) == ["A", "new", "B", "C"]


@pytest.mark.asyncio
async def test_position_zero_inserts_at_front(session):
    await schema_routes.add_column(
        SESSION_ID, schema_routes.ColumnAddRequest(name="new", definition="d", position=0),
    )
    assert _names(session) == ["new", "A", "B", "C"]


@pytest.mark.asyncio
async def test_position_beyond_length_clamps_to_append(session):
    """Covers the schema-shrank-since-delete case (e.g. another column was
    deleted after the one being undone, shifting indices)."""
    await schema_routes.add_column(
        SESSION_ID, schema_routes.ColumnAddRequest(name="new", definition="d", position=99),
    )
    assert _names(session) == ["A", "B", "C", "new"]
    assert _stats_names(session) == ["A", "B", "C", "new"]


@pytest.mark.asyncio
async def test_duplicate_name_still_rejected_regardless_of_position(session):
    with pytest.raises(HTTPException) as exc:
        await schema_routes.add_column(
            SESSION_ID, schema_routes.ColumnAddRequest(name="A", definition="d", position=1),
        )
    assert exc.value.status_code == 400
    assert _names(session) == ["A", "B", "C"]
