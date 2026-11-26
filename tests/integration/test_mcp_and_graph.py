"""Stage 2 integration tests for FastMCP stdio server, idempotency, LangGraph, and FastAPI."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from src.agent.graph import SupportAgentRunner
from src.config import AppConfig, Settings
from src.database.repositories import SupportRepository
from src.mcp_server.client import MCPToolClient


@pytest.mark.asyncio
async def test_mcp_stdio_server_tools_and_idempotency(
    seeded_env: dict[str, Path], rules: AppConfig
) -> None:
    app_db = seeded_env["app_db"]
    repo = SupportRepository(app_db)
    client = MCPToolClient(
        db_path=app_db,
        rules_path=Path("config/business_rules.yaml").resolve(),
        timeout_seconds=10.0,
        max_retries=1,
    )
    async with client:
        tools = await client.list_tools()
        assert set(tools) >= {
            "get_order",
            "get_order_status",
            "check_refund_eligibility",
            "create_refund",
            "get_refund_status",
            "create_support_ticket",
            "get_support_ticket",
        }

        order_res, _ = await client.call_tool(
            "get_order_status", {"customer_id": "CUST-1001", "order_id": "ORD-1001"}
        )
        assert order_res["ok"] is True
        assert order_res["status"] == "delivered"

        # Cross-customer access must be denied
        denied_res, _ = await client.call_tool(
            "get_order", {"customer_id": "CUST-1001", "order_id": "ORD-1004"}
        )
        assert denied_res["ok"] is False

        # Idempotent create_refund retry creates exactly one refund and one refund_approved event
        ref1, _ = await client.call_tool(
            "create_refund",
            {
                "customer_id": "CUST-1001",
                "order_id": "ORD-1001",
                "amount": 189.90,
                "reason": "Leaking coffee maker",
                "idempotency_key": "mcp-idem-001",
                "thread_id": "mcp-test-thread",
            },
        )
        ref2, _ = await client.call_tool(
            "create_refund",
            {
                "customer_id": "CUST-1001",
                "order_id": "ORD-1001",
                "amount": 189.90,
                "reason": "Leaking coffee maker",
                "idempotency_key": "mcp-idem-001",
                "thread_id": "mcp-test-thread",
            },
        )
        assert ref1["ok"] is True
        assert ref2["ok"] is True
        assert ref2["idempotent_replay"] is True
        assert ref1["refund_id"] == ref2["refund_id"]

        assert len(repo.get_refunds_for_order("ORD-1001")) == 1
        approved_events = [
            e
            for e in repo.list_events(order_id="ORD-1001")
            if e.event_type == "refund_approved"
        ]
        assert len(approved_events) == 1

        # Idempotent create_support_ticket creates exactly one ticket and one escalation_created event
        tkt1, _ = await client.call_tool(
            "create_support_ticket",
            {
                "customer_id": "CUST-1004",
                "order_id": "ORD-1010",
                "reason": "Broken gaming chair over 500 BRL cap",
                "priority": "high",
                "idempotency_key": "mcp-tkt-001",
                "thread_id": "mcp-test-thread",
            },
        )
        tkt2, _ = await client.call_tool(
            "create_support_ticket",
            {
                "customer_id": "CUST-1004",
                "order_id": "ORD-1010",
                "reason": "Broken gaming chair over 500 BRL cap",
                "priority": "high",
                "idempotency_key": "mcp-tkt-001",
                "thread_id": "mcp-test-thread",
            },
        )
        assert tkt1["ok"] is True
        assert tkt2["idempotent_replay"] is True
        assert tkt1["ticket_id"] == tkt2["ticket_id"]
        assert len(repo.list_support_tickets_for_customer("CUST-1004")) == 1
        esc_events = [
            e
            for e in repo.list_events(order_id="ORD-1010")
            if e.event_type == "escalation_created"
        ]
        assert len(esc_events) == 1


@pytest.mark.asyncio
async def test_langgraph_sqlite_checkpoint_and_multiturn_flow(
    test_settings: Settings, rules: AppConfig, seeded_env: dict[str, Path]
) -> None:
    async with SupportAgentRunner(
        settings=test_settings,
        rules=rules,
        app_db_path=seeded_env["app_db"],
        checkpoint_db_path=seeded_env["ckpt_db"],
        trace_log_path=seeded_env["trace_log"],
        mode="improved",
    ) as runner:
        t1 = await runner.arun_turn(
            thread_id="int-thread-1",
            customer_id="CUST-1001",
            user_message="Please note I prefer email updates for all follow-ups.",
        )
        assert (
            "preference" in str(t1["final_response"]["answer"]).lower()
            or "noted" in str(t1["final_response"]["answer"]).lower()
        )

        t2 = await runner.arun_turn(
            thread_id="int-thread-1",
            customer_id="CUST-1001",
            user_message="Issue a refund of 189.90 BRL for order ORD-1001.",
        )
        assert "REF-" in t2["final_response"]["answer"]

        async with SupportAgentRunner(
            settings=test_settings,
            rules=rules,
            app_db_path=seeded_env["app_db"],
            checkpoint_db_path=seeded_env["ckpt_db"],
            trace_log_path=seeded_env["trace_log"],
            mode="improved",
        ) as runner2:
            t3 = await runner2.arun_turn(
                thread_id="int-thread-1",
                customer_id="CUST-1001",
                user_message="Remind me, what contact method did I tell you I prefer at the start?",
            )
            assert "email" in t3["final_response"]["answer"].lower()


def test_fastapi_endpoints(monkeypatch: pytest.MonkeyPatch, seeded_env: dict[str, Path]) -> None:
    monkeypatch.setenv("LLM_MODE", "mock")
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("DB_PATH", str(seeded_env["app_db"]))
    monkeypatch.setenv("APP_DB_PATH", str(seeded_env["app_db"]))
    monkeypatch.setenv("SEED_DB_PATH", str(seeded_env["seed_db"]))
    monkeypatch.setenv("CHECKPOINT_DB_PATH", str(seeded_env["ckpt_db"]))
    monkeypatch.setenv("TRACE_LOG_PATH", str(seeded_env["trace_log"]))

    from src.api.app import app

    with TestClient(app) as client:
        h = client.get("/health")
        assert h.status_code == 200
        assert h.json()["status"] == "ok"

        customers = client.get("/customers")
        assert customers.status_code == 200
        assert len(customers.json()) >= 10

        chat = client.post(
            "/chat",
            json={
                "thread_id": "api-int-1",
                "customer_id": "CUST-1001",
                "message": "Check order ORD-1001 please.",
            },
        )
        assert chat.status_code == 200
        assert "ORD-1001" in chat.json()["answer"]

        traces = client.get("/traces")
        assert traces.status_code == 200
        assert traces.json()["metrics"]["total_events"] > 0
