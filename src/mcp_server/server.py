"""FastMCP server exposing customer support tools over stdio transport."""

from __future__ import annotations

import os
from typing import Any

from mcp.server.fastmcp import FastMCP

from src.config import AppConfig, get_settings, load_app_config
from src.database.repositories import SupportRepository
from src.mcp_server.order_tools import handle_get_order, handle_get_order_status
from src.mcp_server.refund_tools import (
    handle_check_refund_eligibility,
    handle_create_refund,
    handle_get_refund_status,
)
from src.mcp_server.support_tools import (
    handle_create_support_ticket,
    handle_get_support_ticket,
)

mcp = FastMCP("CustomerSupportMCP")


def _deps() -> tuple[SupportRepository, AppConfig]:
    settings = get_settings()
    db_path = (
        os.environ.get("APP_DB_PATH")
        or os.environ.get("DB_PATH")
        or str(settings.resolve_path(settings.effective_db_path))
    )
    rules_path = os.environ.get("BUSINESS_RULES_PATH") or settings.business_rules_path
    return SupportRepository(db_path), load_app_config(rules_path)


@mcp.tool()
def get_order(customer_id: str, order_id: str) -> dict[str, Any]:
    """Look up full details for an order owned by customer_id."""
    repo, rules = _deps()
    return handle_get_order(repo, rules, customer_id=customer_id, order_id=order_id)


@mcp.tool()
def get_order_status(customer_id: str, order_id: str) -> dict[str, Any]:
    """Get delivery status and tracking notes for a customer's order."""
    repo, rules = _deps()
    return handle_get_order_status(repo, rules, customer_id=customer_id, order_id=order_id)


@mcp.tool()
def check_refund_eligibility(
    customer_id: str, order_id: str, requested_amount: float
) -> dict[str, Any]:
    """Check deterministic refund eligibility under business policy rules."""
    repo, rules = _deps()
    return handle_check_refund_eligibility(
        repo,
        rules,
        customer_id=customer_id,
        order_id=order_id,
        requested_amount=requested_amount,
    )


@mcp.tool()
def create_refund(
    customer_id: str,
    order_id: str,
    amount: float,
    reason: str = "Customer requested refund",
    idempotency_key: str = "",
    thread_id: str = "mcp",
) -> dict[str, Any]:
    """Create a refund atomically inside a SQLite BEGIN IMMEDIATE transaction."""
    repo, rules = _deps()
    return handle_create_refund(
        repo,
        rules,
        customer_id=customer_id,
        order_id=order_id,
        amount=amount,
        reason=reason,
        idempotency_key=idempotency_key,
        thread_id=thread_id,
    )


@mcp.tool()
def get_refund_status(customer_id: str, order_id: str) -> dict[str, Any]:
    """Check existing refunds for a customer's order."""
    repo, rules = _deps()
    return handle_get_refund_status(repo, rules, customer_id=customer_id, order_id=order_id)


@mcp.tool()
def create_support_ticket(
    customer_id: str,
    order_id: str | None,
    reason: str,
    priority: str,
    idempotency_key: str,
    thread_id: str = "mcp",
) -> dict[str, Any]:
    """Create an escalated support ticket idempotently."""
    repo, rules = _deps()
    return handle_create_support_ticket(
        repo,
        rules,
        customer_id=customer_id,
        order_id=order_id,
        reason=reason,
        priority=priority,
        idempotency_key=idempotency_key,
        thread_id=thread_id,
    )


@mcp.tool()
def get_support_ticket(customer_id: str, ticket_id: str) -> dict[str, Any]:
    """Get details of an escalated support ticket owned by the customer."""
    repo, rules = _deps()
    return handle_get_support_ticket(repo, rules, customer_id=customer_id, ticket_id=ticket_id)


if __name__ == "__main__":
    mcp.run(transport="stdio")
