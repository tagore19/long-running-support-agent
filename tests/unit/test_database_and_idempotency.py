"""Unit tests for SQLite append-only Event Store and atomic tool idempotency."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from src.config import BusinessRulesConfig
from src.database.connection import get_connection
from src.database.repositories import SupportRepository
from src.mcp_server.refund_tools import handle_create_refund
from src.mcp_server.support_tools import handle_create_support_ticket


def test_events_table_is_strictly_append_only(seeded_db: tuple[Path, SupportRepository]) -> None:
    db_path, repo = seeded_db
    evt = repo.append_event(
        customer_id="CUST-1001",
        thread_id="t-append",
        event_type="order_looked_up",
        order_id="ORD-1001",
        actor="unit_test",
    )
    with pytest.raises(sqlite3.DatabaseError, match="append-only"), get_connection(db_path) as conn:
        conn.execute("UPDATE events SET event_type = 'refund_approved' WHERE event_id = ?", (evt.event_id,))

    with pytest.raises(sqlite3.DatabaseError, match="append-only"), get_connection(db_path) as conn:
        conn.execute("DELETE FROM events WHERE event_id = ?", (evt.event_id,))


def test_create_refund_idempotency_and_duplicate_block(
    seeded_db: tuple[Path, SupportRepository], rules: BusinessRulesConfig
) -> None:
    _, repo = seeded_db
    res1 = handle_create_refund(
        repo,
        rules,
        customer_id="CUST-1001",
        order_id="ORD-1001",
        amount=189.90,
        idempotency_key="idem-refund-1",
    )
    assert res1["ok"] is True
    assert res1["idempotent_replay"] is False

    # Same idempotency key returns identical refund without creating a 2nd row
    res2 = handle_create_refund(
        repo,
        rules,
        customer_id="CUST-1001",
        order_id="ORD-1001",
        amount=189.90,
        idempotency_key="idem-refund-1",
    )
    assert res2["ok"] is True
    assert res2["idempotent_replay"] is True
    assert res2["refund_id"] == res1["refund_id"]

    # Different idempotency key on same order is blocked by duplicate policy
    res3 = handle_create_refund(
        repo,
        rules,
        customer_id="CUST-1001",
        order_id="ORD-1001",
        amount=50.0,
        idempotency_key="idem-refund-2",
    )
    assert res3["ok"] is False
    assert "already has an existing refund" in res3["reason"]
    assert len(repo.get_refunds_for_order("ORD-1001")) == 1


def test_create_support_ticket_idempotency(
    seeded_db: tuple[Path, SupportRepository], rules: BusinessRulesConfig
) -> None:
    _, repo = seeded_db
    t1 = handle_create_support_ticket(
        repo,
        rules,
        customer_id="CUST-1004",
        order_id="ORD-1010",
        reason="Broken armrest over 500 BRL limit",
        priority="high",
        idempotency_key="idem-tkt-1",
    )
    t2 = handle_create_support_ticket(
        repo,
        rules,
        customer_id="CUST-1004",
        order_id="ORD-1010",
        reason="Broken armrest over 500 BRL limit",
        priority="high",
        idempotency_key="idem-tkt-1",
    )
    assert t1["ok"] is True
    assert t2["ok"] is True
    assert t2["idempotent_replay"] is True
    assert t1["ticket_id"] == t2["ticket_id"]
    assert len(repo.list_support_tickets_for_customer("CUST-1004")) == 1
