"""Stage 1 foundation smoke tests: config, idempotent seeding, append-only events, and multi-turn graph."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from scripts.seed_data import seed_databases
from src.agent.graph import SupportAgentRunner
from src.config import (
    AppConfig,
    ContextBudgetConfig,
    GuardrailConfig,
    RefundPolicyConfig,
    load_app_config,
)
from src.database.connection import get_connection
from src.database.repositories import EventRepository, OrderRepository, SupportRepository


def test_stage1_config_loading() -> None:
    cfg = load_app_config("config/business_rules.yaml")
    assert isinstance(cfg, AppConfig)
    assert isinstance(cfg.refund_policy, RefundPolicyConfig)
    assert isinstance(cfg.context, ContextBudgetConfig)
    assert isinstance(cfg.guardrails, GuardrailConfig)

    assert cfg.refund_policy.refund_window_days == 30
    assert cfg.refund_policy.maximum_refund_amount == 500.0
    assert cfg.refund_policy.currency == "BRL"
    assert cfg.context.compaction_trigger_tokens == 8500
    assert cfg.guardrails.max_steps == 8
    assert cfg.guardrails.max_input_chars == 4000


def test_stage1_idempotent_seeding_and_repositories(tmp_path: Path) -> None:
    seed_db = tmp_path / "olist_seed.db"
    app_db = tmp_path / "app.db"

    first = seed_databases(seed_db_path=seed_db, app_db_path=app_db, reset=False)
    second = seed_databases(seed_db_path=seed_db, app_db_path=app_db, reset=False)

    assert first["orders"] == 35
    assert second == first

    order_repo = OrderRepository(app_db)
    ord_1001 = order_repo.get("ORD-1001")
    assert ord_1001 is not None
    assert ord_1001.customer_id == "CUST-1001"
    assert ord_1001.status == "delivered"

    # Verify prompt-injection fixture order_note exists for later security tests
    ord_1010 = order_repo.get("ORD-1010")
    assert ord_1010 is not None
    assert "IGNORE PREVIOUS INSTRUCTIONS" in ord_1010.order_note


def test_stage1_append_only_events(tmp_path: Path) -> None:
    app_db = tmp_path / "app.db"
    SupportRepository(app_db)
    event_repo = EventRepository(app_db)

    evt = event_repo.append(
        customer_id="CUST-1001",
        thread_id="stage1-thread",
        event_type="order_looked_up",
        order_id="ORD-1001",
        actor="stage1_test",
        metadata={"source": "smoke_test"},
    )
    events = event_repo.list(customer_id="CUST-1001", thread_id="stage1-thread")
    assert len(events) == 1
    assert events[0].event_id == evt.event_id

    with pytest.raises(sqlite3.DatabaseError, match="append-only"), get_connection(app_db) as conn:
        conn.execute(
            "UPDATE events SET actor = 'tampered' WHERE event_id = ?",
            (evt.event_id,),
        )

    with pytest.raises(sqlite3.DatabaseError, match="append-only"), get_connection(app_db) as conn:
        conn.execute("DELETE FROM events WHERE event_id = ?", (evt.event_id,))


async def test_stage1_multi_turn_graph_with_sqlite_checkpointer(tmp_path: Path) -> None:
    seed_db = tmp_path / "olist_seed.db"
    app_db = tmp_path / "app.db"
    ckpt_db = tmp_path / "checkpoints.db"
    trace_log = tmp_path / "traces.jsonl"
    seed_databases(seed_db_path=seed_db, app_db_path=app_db, reset=True)

    async with SupportAgentRunner(
        app_db_path=app_db,
        checkpoint_db_path=ckpt_db,
        trace_log_path=trace_log,
        mode="improved",
    ) as runner:
        turn1 = await runner.arun_turn(
            thread_id="stage1-multi-1",
            customer_id="CUST-1001",
            user_message="Please note I prefer email updates only.",
        )
        assert turn1["thread_id"] == "stage1-multi-1"
        assert len(turn1["messages"]) == 2
        assert turn1["response"]

        turn2 = await runner.arun_turn(
            thread_id="stage1-multi-1",
            customer_id="CUST-1001",
            user_message="What is the status of my order ORD-1001?",
        )
        assert len(turn2["messages"]) == 4
        assert "ORD-1001" in turn2["response"]
        assert "delivered" in turn2["response"].lower()
