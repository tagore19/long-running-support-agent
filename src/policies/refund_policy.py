"""Deterministic refund eligibility evaluation grounded in YAML policy rules."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, model_validator

from src.config import AppConfig, RefundPolicyConfig
from src.database.models import OrderRecord, RefundRecord
from src.policies.authorization import verify_order_ownership


class EligibilityResult(BaseModel):
    """Structured output returned by deterministic refund eligibility checks."""

    eligible: bool
    max_allowed_amount: float
    remaining_refundable: float = 0.0
    reason_code: str
    message: str
    policy_version: str = "2026.1"
    order_status: str | None = None
    days_since_delivery: int | None = None
    reason: str = ""

    @model_validator(mode="after")
    def _sync_reason(self) -> EligibilityResult:
        if not self.reason:
            self.reason = self.message
        return self


RefundEligibilityResult = EligibilityResult


class RefundDecision(BaseModel):
    """Structured decision record for a refund creation attempt."""

    approved: bool
    status: str
    reason_code: str
    message: str
    idempotency_key: str


def _parse_iso(ts: str) -> datetime:
    normalized = ts.replace("Z", "+00:00")
    dt = datetime.fromisoformat(normalized)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


def _get_field(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def evaluate_refund_eligibility(
    order: OrderRecord | dict[str, Any] | None,
    existing_refunds: list[RefundRecord | dict[str, Any]],
    pending_refunds: list[RefundRecord | dict[str, Any]] | None = None,
    requested_amount: float | None = None,
    policy: RefundPolicyConfig | AppConfig | None = None,
    reference_now: datetime | None = None,
    *,
    customer_id: str | None = None,
    rules: AppConfig | None = None,
    as_of: datetime | None = None,
) -> EligibilityResult:
    """Evaluate refund eligibility strictly via Python code and YAML configuration."""
    resolved_rules = rules
    if isinstance(policy, AppConfig):
        resolved_rules = policy
        policy_cfg = policy.refund_policy
        policy_version = policy.policy_version
    elif isinstance(policy, RefundPolicyConfig):
        policy_cfg = policy
        policy_version = resolved_rules.policy_version if resolved_rules else "2026.1"
    elif resolved_rules is not None:
        policy_cfg = resolved_rules.refund_policy
        policy_version = resolved_rules.policy_version
    else:
        policy_cfg = RefundPolicyConfig()
        policy_version = "2026.1"

    if order is None:
        return EligibilityResult(
            eligible=False,
            max_allowed_amount=0.0,
            remaining_refundable=0.0,
            reason_code="ORDER_NOT_FOUND",
            message="Order not found.",
            policy_version=policy_version,
        )

    if customer_id is not None:
        allowed, auth_msg = verify_order_ownership(order, customer_id)
        if not allowed:
            return EligibilityResult(
                eligible=False,
                max_allowed_amount=0.0,
                remaining_refundable=0.0,
                reason_code="OWNERSHIP_DENIED",
                message=auth_msg,
                policy_version=policy_version,
            )

    order_id = str(_get_field(order, "order_id", ""))
    order_status = str(_get_field(order, "status", ""))
    order_amount = float(_get_field(order, "amount", 0.0))
    refunded_so_far = float(_get_field(order, "refunded_amount", 0.0))
    delivered_date = _get_field(order, "delivered_date", None)
    purchase_date = _get_field(order, "purchase_date", None)

    if order_status == "canceled":
        return EligibilityResult(
            eligible=False,
            max_allowed_amount=0.0,
            remaining_refundable=0.0,
            reason_code="ORDER_CANCELED",
            message=f"Order {order_id} has status 'canceled'. Refunds are not allowed on canceled orders.",
            policy_version=policy_version,
            order_status=order_status,
        )

    if policy_cfg.require_delivered_order and order_status != "delivered":
        return EligibilityResult(
            eligible=False,
            max_allowed_amount=0.0,
            remaining_refundable=0.0,
            reason_code="ORDER_NOT_DELIVERED",
            message=(
                f"Order {order_id} has status '{order_status}'. "
                "Refunds are only allowed for delivered orders."
            ),
            policy_version=policy_version,
            order_status=order_status,
        )

    anchor_ts = delivered_date or (None if policy_cfg.require_delivered_order else purchase_date)
    if anchor_ts is None:
        return EligibilityResult(
            eligible=False,
            max_allowed_amount=0.0,
            remaining_refundable=0.0,
            reason_code="MISSING_DELIVERY_DATE",
            message=f"Order {order_id} has no recorded delivery timestamp.",
            policy_version=policy_version,
            order_status=order_status,
        )

    now = reference_now or as_of or datetime.now(UTC)
    anchor_dt = _parse_iso(str(anchor_ts))
    days_since_delivery = max(0, (now - anchor_dt).days)
    if days_since_delivery > policy_cfg.refund_window_days:
        return EligibilityResult(
            eligible=False,
            max_allowed_amount=0.0,
            remaining_refundable=0.0,
            reason_code="REFUND_WINDOW_EXPIRED",
            message=(
                f"Order {order_id} was delivered {days_since_delivery} days ago, "
                f"which exceeds the {policy_cfg.refund_window_days}-day refund window."
            ),
            policy_version=policy_version,
            order_status=order_status,
            days_since_delivery=days_since_delivery,
        )

    active_refunds = [
        r for r in existing_refunds if _get_field(r, "status") in {"approved", "completed"}
    ]
    active_pending = list(pending_refunds or []) + [
        r for r in existing_refunds if _get_field(r, "status") == "pending"
    ]

    total_refunded = max(
        refunded_so_far,
        sum(float(_get_field(r, "amount", 0.0)) for r in active_refunds),
    )
    remaining_refundable = max(0.0, round(order_amount - total_refunded, 2))

    if active_pending:
        return EligibilityResult(
            eligible=False,
            max_allowed_amount=0.0,
            remaining_refundable=remaining_refundable,
            reason_code="PENDING_REFUND_EXISTS",
            message=f"Order {order_id} already has an active pending refund request.",
            policy_version=policy_version,
            order_status=order_status,
            days_since_delivery=days_since_delivery,
        )

    if active_refunds and not policy_cfg.allow_duplicate_refunds:
        first_id = _get_field(active_refunds[0], "refund_id", "existing")
        first_amt = float(_get_field(active_refunds[0], "amount", 0.0))
        return EligibilityResult(
            eligible=False,
            max_allowed_amount=0.0,
            remaining_refundable=remaining_refundable,
            reason_code="DUPLICATE_REFUND_BLOCKED",
            message=(
                f"Order {order_id} already has an existing refund "
                f"({first_id}) of {first_amt:.2f} {policy_cfg.currency}. "
                "Duplicate refunds are not allowed."
            ),
            policy_version=policy_version,
            order_status=order_status,
            days_since_delivery=days_since_delivery,
        )

    if remaining_refundable <= 0.0:
        return EligibilityResult(
            eligible=False,
            max_allowed_amount=0.0,
            remaining_refundable=0.0,
            reason_code="NO_REMAINING_REFUNDABLE",
            message=f"Order {order_id} has 0.00 {policy_cfg.currency} remaining refundable balance.",
            policy_version=policy_version,
            order_status=order_status,
            days_since_delivery=days_since_delivery,
        )

    max_allowed = round(min(remaining_refundable, policy_cfg.maximum_refund_amount), 2)
    amount_to_check = remaining_refundable if requested_amount is None else float(requested_amount)

    if amount_to_check <= 0:
        return EligibilityResult(
            eligible=False,
            max_allowed_amount=max_allowed,
            remaining_refundable=remaining_refundable,
            reason_code="INVALID_AMOUNT",
            message="Requested refund amount must be greater than zero.",
            policy_version=policy_version,
            order_status=order_status,
            days_since_delivery=days_since_delivery,
        )

    if amount_to_check > remaining_refundable + 1e-6:
        return EligibilityResult(
            eligible=False,
            max_allowed_amount=max_allowed,
            remaining_refundable=remaining_refundable,
            reason_code="EXCEEDS_REMAINING_REFUNDABLE",
            message=(
                f"Requested amount {amount_to_check:.2f} {policy_cfg.currency} exceeds "
                f"remaining refundable balance {remaining_refundable:.2f} {policy_cfg.currency}."
            ),
            policy_version=policy_version,
            order_status=order_status,
            days_since_delivery=days_since_delivery,
        )

    if amount_to_check > policy_cfg.maximum_refund_amount + 1e-6:
        return EligibilityResult(
            eligible=False,
            max_allowed_amount=max_allowed,
            remaining_refundable=remaining_refundable,
            reason_code="EXCEEDS_MAXIMUM_REFUND_AMOUNT",
            message=(
                f"Requested amount {amount_to_check:.2f} {policy_cfg.currency} exceeds the "
                f"maximum automated refund limit of {policy_cfg.maximum_refund_amount:.2f} {policy_cfg.currency}. "
                "Human escalation is required."
            ),
            policy_version=policy_version,
            order_status=order_status,
            days_since_delivery=days_since_delivery,
        )

    if not policy_cfg.allow_partial_refunds and abs(amount_to_check - order_amount) > 1e-6:
        return EligibilityResult(
            eligible=False,
            max_allowed_amount=max_allowed,
            remaining_refundable=remaining_refundable,
            reason_code="PARTIAL_REFUND_DISABLED",
            message="Partial refunds are disabled by policy; full order amount is required.",
            policy_version=policy_version,
            order_status=order_status,
            days_since_delivery=days_since_delivery,
        )

    return EligibilityResult(
        eligible=True,
        max_allowed_amount=max_allowed,
        remaining_refundable=remaining_refundable,
        reason_code="ELIGIBLE",
        message=(
            f"Order {order_id} is eligible for a refund of "
            f"{amount_to_check:.2f} {policy_cfg.currency} (delivered {days_since_delivery} days ago)."
        ),
        policy_version=policy_version,
        order_status=order_status,
        days_since_delivery=days_since_delivery,
    )
