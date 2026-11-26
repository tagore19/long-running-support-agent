"""Pydantic v2 data models for SQLite persistence layers."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class OrderItem(BaseModel):
    """Line item inside an Olist order."""

    product_name: str
    category: str
    quantity: int = Field(default=1, ge=1)
    unit_price: float = Field(ge=0.0)


class CustomerRecord(BaseModel):
    """Layer 2 customer record."""

    customer_id: str
    full_name: str
    email_masked: str
    tier: str = "standard"
    city: str = ""
    state: str = ""
    preferred_language: str = "en"
    created_at: str

    @property
    def name(self) -> str:
        return self.full_name

    @property
    def email(self) -> str:
        return self.email_masked


class OrderRecord(BaseModel):
    """Layer 2 order record."""

    order_id: str
    customer_id: str
    status: Literal["delivered", "shipped", "processing", "canceled"]
    purchase_date: str
    delivered_date: str | None = None
    estimated_delivery_date: str | None = None
    amount: float
    currency: str = "BRL"
    category: str = "general"
    order_note: str = ""
    refunded_amount: float = 0.0
    shipping_carrier: str | None = None
    tracking_code: str | None = None
    items: list[OrderItem] = Field(default_factory=list)

    @property
    def created_at(self) -> str:
        return self.purchase_date

    @property
    def delivered_at(self) -> str | None:
        return self.delivered_date

    @property
    def estimated_delivery_at(self) -> str | None:
        return self.estimated_delivery_date


class RefundRecord(BaseModel):
    """Layer 2 refund record."""

    refund_id: str
    order_id: str
    customer_id: str
    amount: float
    currency: str = "BRL"
    status: Literal["approved", "completed", "rejected"]
    reason: str
    policy_version: str = "2026.1"
    idempotency_key: str
    created_at: str


class SupportCaseRecord(BaseModel):
    """Layer 2 human escalation / support ticket record."""

    ticket_id: str
    customer_id: str
    order_id: str | None = None
    status: Literal["open", "in_progress", "escalated", "resolved"]
    priority: Literal["low", "medium", "high", "urgent"]
    summary: str
    idempotency_key: str | None = None
    created_at: str
    updated_at: str

    @property
    def reason(self) -> str:
        return self.summary


class ToolOperationRecord(BaseModel):
    """Idempotency ledger entry for mutating tool executions."""

    idempotency_key: str
    tool_name: str
    customer_id: str
    order_id: str | None = None
    status: Literal["completed", "blocked", "error"]
    response: dict[str, Any]
    created_at: str

    @property
    def result(self) -> dict[str, Any]:
        return self.response


EventType = Literal[
    "order_looked_up",
    "refund_requested",
    "refund_rejected",
    "refund_approved",
    "refund_completed",
    "escalation_created",
    "policy_blocked",
]


class EventRecord(BaseModel):
    """Layer 1 append-only audit event."""

    seq: int | None = None
    event_id: str
    customer_id: str
    thread_id: str
    event_type: EventType
    order_id: str | None = None
    amount: float | None = None
    timestamp: str
    actor: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class EpisodeRecord(BaseModel):
    """Layer 3 episodic memory entry."""

    episode_id: str
    customer_id: str
    thread_id: str
    content: str
    entities: list[str] = Field(default_factory=list)
    topic: str
    importance: float = Field(ge=0.0, le=1.0)
    created_at: str


class ConversationTurnRecord(BaseModel):
    """Archived conversation turn for Layer 4 hybrid retrieval."""

    turn_id: int | None = None
    thread_id: str
    customer_id: str
    turn_index: int
    role: Literal["user", "assistant", "tool"]
    content: str
    entities: list[str] = Field(default_factory=list)
    compacted: bool = False
    created_at: str


class CompactionSnapshotRecord(BaseModel):
    """Persisted structured compaction checkpoint."""

    snapshot_id: str
    thread_id: str
    customer_id: str
    summary: dict[str, Any]
    compacted_up_to_turn: int
    token_estimate_before: int
    token_estimate_after: int
    created_at: str
