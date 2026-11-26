"""MCP order lookup and status tools with strict customer ownership checks."""

from __future__ import annotations

from typing import Any

from src.config import AppConfig
from src.database.repositories import SupportRepository
from src.policies.authorization import verify_order_ownership


def handle_get_order(
    repo: SupportRepository,
    rules: AppConfig,
    customer_id: str,
    order_id: str,
) -> dict[str, Any]:
    """Fetch full order details after verifying customer ownership."""
    order = repo.get_order(order_id)
    auth = verify_order_ownership(order, customer_id)
    if not auth.allowed or order is None:
        return {
            "ok": False,
            "reason": auth.reason,
            "order_id": order_id,
            "customer_id": customer_id,
            "policy_version": rules.policy_version,
        }
    return {
        "ok": True,
        "order": order.model_dump(),
        "policy_version": rules.policy_version,
    }


def handle_get_order_status(
    repo: SupportRepository,
    rules: AppConfig,
    customer_id: str,
    order_id: str,
) -> dict[str, Any]:
    """Fetch concise delivery and tracking status after ownership verification."""
    order = repo.get_order(order_id)
    auth = verify_order_ownership(order, customer_id)
    if not auth.allowed or order is None:
        return {
            "ok": False,
            "reason": auth.reason,
            "order_id": order_id,
            "customer_id": customer_id,
            "policy_version": rules.policy_version,
        }
    return {
        "ok": True,
        "order_id": order.order_id,
        "customer_id": order.customer_id,
        "status": order.status,
        "amount": order.amount,
        "currency": order.currency,
        "category": order.category,
        "purchase_date": order.purchase_date,
        "delivered_date": order.delivered_date,
        "estimated_delivery_date": order.estimated_delivery_date,
        "order_note": order.order_note,
        "refunded_amount": order.refunded_amount,
        "policy_version": rules.policy_version,
    }
