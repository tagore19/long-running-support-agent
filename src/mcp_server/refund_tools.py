"""MCP refund eligibility, creation, and status tools with SQLite transaction guarantees."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from src.config import AppConfig
from src.database.repositories import SupportRepository
from src.policies.authorization import verify_order_ownership
from src.policies.refund_policy import evaluate_refund_eligibility


def handle_check_refund_eligibility(
    repo: SupportRepository,
    rules: AppConfig,
    customer_id: str,
    order_id: str,
    requested_amount: float | None = None,
    as_of: datetime | None = None,
) -> dict[str, Any]:
    """Deterministically evaluate refund eligibility for an order."""
    order = repo.get_order(order_id)
    existing_refunds = repo.get_refunds_for_order(order_id) if order else []
    decision = evaluate_refund_eligibility(
        order=order,
        existing_refunds=existing_refunds,
        pending_refunds=[],
        requested_amount=requested_amount,
        policy=rules.refund_policy,
        reference_now=as_of,
        customer_id=customer_id,
        rules=rules,
    )
    return decision.model_dump()


def handle_create_refund(
    repo: SupportRepository,
    rules: AppConfig,
    customer_id: str,
    order_id: str,
    amount: float,
    reason: str = "Customer requested refund",
    idempotency_key: str = "",
    thread_id: str = "mcp",
    as_of: datetime | None = None,
) -> dict[str, Any]:
    """Re-validate policy rules and atomically persist refund + events inside SQLite."""
    if not idempotency_key or not idempotency_key.strip():
        return {
            "ok": False,
            "approved": False,
            "status": "rejected",
            "reason_code": "MISSING_IDEMPOTENCY_KEY",
            "reason": "idempotency_key is required for create_refund.",
            "message": "idempotency_key is required for create_refund.",
            "policy_version": rules.policy_version,
        }

    _, result, _ = repo.create_refund_atomic(
        customer_id=customer_id,
        order_id=order_id,
        amount=amount,
        currency=rules.refund_policy.currency,
        reason=reason,
        policy_version=rules.policy_version,
        idempotency_key=idempotency_key,
        allow_duplicate_refunds=rules.refund_policy.allow_duplicate_refunds,
        max_allowed_amount=rules.refund_policy.maximum_refund_amount,
        thread_id=thread_id,
        rules=rules,
        as_of=as_of,
    )
    return result


def handle_get_refund_status(
    repo: SupportRepository,
    rules: AppConfig,
    customer_id: str,
    order_id: str,
) -> dict[str, Any]:
    """Return all refunds associated with an order owned by customer_id."""
    order = repo.get_order(order_id)
    allowed, reason = verify_order_ownership(order, customer_id)
    if not allowed or order is None:
        return {
            "ok": False,
            "reason": reason,
            "order_id": order_id,
            "customer_id": customer_id,
            "policy_version": rules.policy_version,
        }
    refunds = repo.get_refunds_for_order(order_id)
    return {
        "ok": True,
        "order_id": order_id,
        "customer_id": customer_id,
        "refunds": [r.model_dump() for r in refunds],
        "policy_version": rules.policy_version,
    }
