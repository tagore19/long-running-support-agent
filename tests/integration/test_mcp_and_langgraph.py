"""Integration tests for FastMCP stdio server, LangGraph SqliteSaver, and FastAPI."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
from src.agent.graph import SupportAgentRunner
from src.config import AppConfig
from src.mcp_server.client import MCPToolClient


async def test_fastmcp_stdio_server_tools_end_to_end(
    seeded_db: tuple[Path, object], rules: AppConfig
) -> None:
    app_db, _ = seeded_db
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

        status_res, _ = await client.call_tool(
            "get_order_status", {"customer_id": "CUST-1002", "order_id": "ORD-1004"}
        )
        assert status_res["ok"] is True
        assert "AZ1004004RJ" in status_res["order_note"]

        denied_res, _ = await client.call_tool(
            "get_order", {"customer_id": "CUST-1001", "order_id": "ORD-1004"}
        )
        assert denied_res["ok"] is False


async def test_langgraph_checkpointer_persistence_across_runner_restart(
    seeded_db: tuple[Path, object], tmp_path: Path
) -> None:
    app_db, _ = seeded_db
    ckpt_db = tmp_path / "checkpoints.db"
    trace_log = tmp_path / "traces.jsonl"

    async with SupportAgentRunner(
        app_db_path=app_db,
        checkpoint_db_path=ckpt_db,
        trace_log_path=trace_log,
        mode="improved",
    ) as runner1:
        r1 = await runner1.arun_turn(
            thread_id="persist-thread-1",
            customer_id="CUST-1001",
            user_message="Please remember I prefer email updates only.",
        )
        assert len(r1["messages"]) == 2

    async with SupportAgentRunner(
        app_db_path=app_db,
        checkpoint_db_path=ckpt_db,
        trace_log_path=trace_log,
        mode="improved",
    ) as runner2:
        r2 = await runner2.arun_turn(
            thread_id="persist-thread-1",
            customer_id="CUST-1001",
            user_message="Remind me what contact method I prefer from the start?",
        )
        assert "email" in r2["response"].lower()
        assert len(r2["messages"]) == 4


def test_fastapi_endpoints_smoke(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DB_PATH", str(tmp_path / "api_app.db"))
    monkeypatch.setenv("APP_DB_PATH", str(tmp_path / "api_app.db"))
    monkeypatch.setenv("SEED_DB_PATH", str(tmp_path / "api_seed.db"))
    monkeypatch.setenv("CHECKPOINT_DB_PATH", str(tmp_path / "api_ckpt.db"))
    monkeypatch.setenv("TRACE_LOG_PATH", str(tmp_path / "api_traces.jsonl"))

    from src.api.app import app

    with TestClient(app) as client:
        h = client.get("/health")
        assert h.status_code == 200
        assert h.json()["status"] == "ok"

        customers = client.get("/customers")
        assert customers.status_code == 200
        assert len(customers.json()) == 12

        chat_res = client.post(
            "/chat",
            json={
                "thread_id": "api-th-1",
                "customer_id": "CUST-1001",
                "message": "Check order ORD-1001 please.",
            },
        )
        assert chat_res.status_code == 200
        body = chat_res.json()
        assert "ORD-1001" in body["response"]

        traces = client.get("/traces")
        assert traces.status_code == 200
        assert traces.json()["metrics"]["total_events"] > 0
