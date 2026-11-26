"""Unit tests for deterministic refund policies, ownership gates, and append-only events."""

from __future__ import annotations

import sqlite3

import pytest
from src.config import BusinessRulesConfig
from src.database.connection import get_connection
from src.database.repositories import SupportRepository
from src.mcp_server.refund_tools import handle_check_refund_eligibility, handle_create_refund
from src.policies.authorization import verify_order_ownership


def test_order_ownership_enforced(repo: SupportRepository) -> None:
    order = repo.get_order("ORD-1004")  # belongs to CUST-1002
    assert verify_order_ownership(order, "CUST-1002").allowed is True
    denied = verify_order_ownership(order, "CUST-1001")
    assert denied.allowed is False
    assert "does not belong" in denied.reason


def test_refund_eligibility_valid_and_boundary_500(
    repo: SupportRepository, rules: BusinessRulesConfig
) -> None:
    res_valid = handle_check_refund_eligibility(
        repo, rules, customer_id="CUST-1001", order_id="ORD-1001", requested_amount=189.90
    )
    assert res_valid["eligible"] is True
    assert res_valid["max_allowed_amount"] == 189.90

    res_500 = handle_check_refund_eligibility(
        repo, rules, customer_id="CUST-1007", order_id="ORD-1019", requested_amount=500.00
    )
    assert res_500["eligible"] is True
    assert res_500["max_allowed_amount"] == 500.00


def test_refund_blocked_when_over_limit_expired_or_undelivered(
    repo: SupportRepository, rules: BusinessRulesConfig
) -> None:
    over_limit = handle_check_refund_eligibility(
        repo, rules, customer_id="CUST-1004", order_id="ORD-1010", requested_amount=780.00
    )
    assert over_limit["eligible"] is False
    assert over_limit["max_allowed_amount"] == 500.0
    assert "500.00" in over_limit["reason"]

    expired = handle_check_refund_eligibility(
        repo, rules, customer_id="CUST-1001", order_id="ORD-1003", requested_amount=120.50
    )
    assert expired["eligible"] is False
    assert "30-day" in expired["reason"]

    shipped = handle_check_refund_eligibility(
        repo, rules, customer_id="CUST-1002", order_id="ORD-1004", requested_amount=259.00
    )
    assert shipped["eligible"] is False
    assert "shipped" in shipped["reason"]


def test_duplicate_refund_blocked_and_idempotent_replay_safe(
    repo: SupportRepository, rules: BusinessRulesConfig
) -> None:
    first = handle_create_refund(
        repo,
        rules,
        customer_id="CUST-1005",
        order_id="ORD-1013",
        amount=150.0,
        idempotency_key="key-partial-1",
    )
    assert first["ok"] is True
    assert first["idempotent_replay"] is False

    # Same idempotency key returns identical result without creating a second refund
    replay = handle_create_refund(
        repo,
        rules,
        customer_id="CUST-1005",
        order_id="ORD-1013",
        amount=150.0,
        idempotency_key="key-partial-1",
    )
    assert replay["ok"] is True
    assert replay["idempotent_replay"] is True
    assert replay["refund_id"] == first["refund_id"]

    # New idempotency key on same order is blocked by allow_duplicate_refunds=false
    duplicate = handle_create_refund(
        repo,
        rules,
        customer_id="CUST-1005",
        order_id="ORD-1013",
        amount=50.0,
        idempotency_key="key-partial-2",
    )
    assert duplicate["ok"] is False
    assert "already has an existing refund" in duplicate["reason"]
    assert len(repo.get_refunds_for_order("ORD-1013")) == 1


def test_events_table_is_strictly_append_only(repo: SupportRepository) -> None:
    evt = repo.append_event(
        customer_id="CUST-1001",
        thread_id="t-1",
        event_type="order_looked_up",
        order_id="ORD-1001",
        actor="test",
    )
    with pytest.raises(sqlite3.IntegrityError, match="append-only"), get_connection(repo.db_path) as conn:
        conn.execute("UPDATE events SET event_type = 'refund_approved' WHERE event_id = ?", (evt.event_id,))

    with pytest.raises(sqlite3.IntegrityError, match="append-only"), get_connection(repo.db_path) as conn:
        conn.execute("DELETE FROM events WHERE event_id = ?", (evt.event_id,))
