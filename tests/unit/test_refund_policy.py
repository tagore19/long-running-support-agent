"""Stage 2 unit tests for deterministic refund policy and ownership authorization."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from src.config import RefundPolicyConfig
from src.policies.authorization import verify_order_ownership
from src.policies.refund_policy import (
    EligibilityResult,
    RefundDecision,
    evaluate_refund_eligibility,
)


def _sample_order(
    *,
    order_id: str = "ORD-2001",
    customer_id: str = "CUST-1001",
    status: str = "delivered",
    amount: float = 250.0,
    days_ago: int = 5,
    refunded_amount: float = 0.0,
) -> dict[str, object]:
    now = datetime(2026, 11, 26, 12, 0, tzinfo=UTC)
    delivered = (now - timedelta(days=days_ago)).isoformat().replace("+00:00", "Z")
    purchase = (now - timedelta(days=days_ago + 3)).isoformat().replace("+00:00", "Z")
    return {
        "order_id": order_id,
        "customer_id": customer_id,
        "status": status,
        "purchase_date": purchase,
        "delivered_date": delivered if status == "delivered" else None,
        "amount": amount,
        "currency": "BRL",
        "refunded_amount": refunded_amount,
    }


def test_verify_order_ownership_tuple_and_attributes() -> None:
    order = _sample_order(customer_id="CUST-1001")
    allowed, msg = verify_order_ownership(order, "CUST-1001")
    assert allowed is True
    assert "verified" in msg.lower()

    denied_ok, denied_msg = verify_order_ownership(order, "CUST-9999")
    assert denied_ok is False
    assert "does not belong" in denied_msg


def test_evaluate_refund_eligibility_valid_and_partial() -> None:
    now = datetime(2026, 11, 26, 12, 0, tzinfo=UTC)
    policy = RefundPolicyConfig()
    order = _sample_order(amount=300.0, days_ago=10)

    res = evaluate_refund_eligibility(
        order=order,
        existing_refunds=[],
        pending_refunds=[],
        requested_amount=120.0,
        policy=policy,
        reference_now=now,
    )
    assert isinstance(res, EligibilityResult)
    assert res.eligible is True
    assert res.remaining_refundable == 300.0
    assert res.max_allowed_amount == 300.0
    assert res.reason_code == "ELIGIBLE"


def test_evaluate_refund_eligibility_expired_window() -> None:
    now = datetime(2026, 11, 26, 12, 0, tzinfo=UTC)
    policy = RefundPolicyConfig(refund_window_days=30)
    order = _sample_order(amount=150.0, days_ago=35)

    res = evaluate_refund_eligibility(
        order=order,
        existing_refunds=[],
        pending_refunds=[],
        requested_amount=150.0,
        policy=policy,
        reference_now=now,
    )
    assert res.eligible is False
    assert res.reason_code == "REFUND_WINDOW_EXPIRED"


def test_evaluate_refund_eligibility_exceeds_maximum_refund_amount() -> None:
    now = datetime(2026, 11, 26, 12, 0, tzinfo=UTC)
    policy = RefundPolicyConfig(maximum_refund_amount=500.0)
    order = _sample_order(amount=780.0, days_ago=4)

    res = evaluate_refund_eligibility(
        order=order,
        existing_refunds=[],
        pending_refunds=[],
        requested_amount=780.0,
        policy=policy,
        reference_now=now,
    )
    assert res.eligible is False
    assert res.reason_code == "EXCEEDS_MAXIMUM_REFUND_AMOUNT"
    assert res.max_allowed_amount == 500.0


def test_evaluate_refund_eligibility_pending_and_duplicate_blocks() -> None:
    now = datetime(2026, 11, 26, 12, 0, tzinfo=UTC)
    policy = RefundPolicyConfig(allow_duplicate_refunds=False)
    order = _sample_order(amount=200.0, days_ago=5)

    pending_res = evaluate_refund_eligibility(
        order=order,
        existing_refunds=[],
        pending_refunds=[{"refund_id": "REF-P1", "amount": 50.0, "status": "pending"}],
        requested_amount=50.0,
        policy=policy,
        reference_now=now,
    )
    assert pending_res.eligible is False
    assert pending_res.reason_code == "PENDING_REFUND_EXISTS"

    dup_res = evaluate_refund_eligibility(
        order=order,
        existing_refunds=[{"refund_id": "REF-1", "amount": 200.0, "status": "completed"}],
        pending_refunds=[],
        requested_amount=50.0,
        policy=policy,
        reference_now=now,
    )
    assert dup_res.eligible is False
    assert dup_res.reason_code == "DUPLICATE_REFUND_BLOCKED"

    decision = RefundDecision(
        approved=False,
        status="rejected",
        reason_code=dup_res.reason_code,
        message=dup_res.message,
        idempotency_key="idem-test-1",
    )
    assert decision.approved is False
