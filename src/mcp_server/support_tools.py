"""Support ticket and human escalation tools."""

from __future__ import annotations

from typing import Any

from src.config import BusinessRulesConfig
from src.database.repositories import SupportRepository
from src.policies.authorization import verify_ticket_ownership


def handle_create_support_ticket(
    repo: SupportRepository,
    rules: BusinessRulesConfig,
    *,
    customer_id: str,
    order_id: str | None,
    reason: str,
    priority: str,
    idempotency_key: str,
    thread_id: str = "mcp",
) -> dict[str, Any]:
    """Create an idempotent support ticket / human escalation."""
    if not idempotency_key or not idempotency_key.strip():
        return {
            "ok": False,
            "reason": "idempotency_key is required for create_support_ticket.",
            "policy_version": rules.policy_version,
        }

    normalized_priority = priority.lower().strip()
    if normalized_priority not in rules.support_policy.valid_priorities:
        normalized_priority = rules.support_policy.default_priority

    _, result, _ = repo.create_support_ticket_atomic(
        customer_id=customer_id,
        order_id=order_id,
        reason=reason,
        priority=normalized_priority,
        idempotency_key=idempotency_key,
        policy_version=rules.policy_version,
        thread_id=thread_id,
    )
    return result


def handle_get_support_ticket(
    repo: SupportRepository,
    rules: BusinessRulesConfig,
    *,
    customer_id: str,
    ticket_id: str,
) -> dict[str, Any]:
    """Fetch a support ticket after verifying ownership."""
    ticket = repo.get_support_ticket(ticket_id)
    auth = verify_ticket_ownership(ticket, customer_id)
    if not auth.allowed or ticket is None:
        return {"ok": False, "error": auth.reason, "ticket_id": ticket_id}
    return {"ok": True, "ticket": ticket.model_dump(), "policy_version": rules.policy_version}
