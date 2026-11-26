"""Repository layer for SQLite transactional state, event store, and memories."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.database.connection import get_connection, init_app_db, transaction
from src.database.models import (
    CompactionSnapshotRecord,
    ConversationTurnRecord,
    CustomerRecord,
    EpisodeRecord,
    EventRecord,
    EventType,
    OrderItem,
    OrderRecord,
    RefundRecord,
    SupportCaseRecord,
    ToolOperationRecord,
)


def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO-8601 format."""
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _row_to_order(row: sqlite3.Row) -> OrderRecord:
    items_raw = json.loads(row["items_json"]) if row["items_json"] else []
    return OrderRecord(
        order_id=row["order_id"],
        customer_id=row["customer_id"],
        status=row["status"],
        purchase_date=row["purchase_date"],
        delivered_date=row["delivered_date"],
        estimated_delivery_date=row["estimated_delivery_date"],
        amount=float(row["amount"]),
        currency=row["currency"],
        category=row["category"],
        order_note=row["order_note"],
        refunded_amount=float(row["refunded_amount"]),
        shipping_carrier=row["shipping_carrier"],
        tracking_code=row["tracking_code"],
        items=[OrderItem.model_validate(item) for item in items_raw],
    )


def _row_to_refund(row: sqlite3.Row) -> RefundRecord:
    return RefundRecord(
        refund_id=row["refund_id"],
        order_id=row["order_id"],
        customer_id=row["customer_id"],
        amount=float(row["amount"]),
        currency=row["currency"],
        status=row["status"],
        reason=row["reason"],
        policy_version=row["policy_version"],
        idempotency_key=row["idempotency_key"],
        created_at=row["created_at"],
    )


def _row_to_ticket(row: sqlite3.Row) -> SupportCaseRecord:
    return SupportCaseRecord(
        ticket_id=row["ticket_id"],
        customer_id=row["customer_id"],
        order_id=row["order_id"],
        status=row["status"],
        priority=row["priority"],
        summary=row["summary"],
        idempotency_key=row["idempotency_key"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


class OrderRepository:
    """Parameterized SQLite repository for customers and orders."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        init_app_db(self.db_path)

    def upsert_customer(self, customer: CustomerRecord) -> None:
        with get_connection(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO customers (
                    customer_id, full_name, email_masked, tier, city, state, preferred_language, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(customer_id) DO UPDATE SET
                    full_name = excluded.full_name,
                    email_masked = excluded.email_masked,
                    tier = excluded.tier,
                    city = excluded.city,
                    state = excluded.state,
                    preferred_language = excluded.preferred_language
                """,
                (
                    customer.customer_id,
                    customer.full_name,
                    customer.email_masked,
                    customer.tier,
                    customer.city,
                    customer.state,
                    customer.preferred_language,
                    customer.created_at,
                ),
            )

    def get_customer(self, customer_id: str) -> CustomerRecord | None:
        with get_connection(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM customers WHERE customer_id = ?",
                (customer_id,),
            ).fetchone()
            return CustomerRecord(**dict(row)) if row else None

    def list_customers(self) -> list[CustomerRecord]:
        with get_connection(self.db_path) as conn:
            rows = conn.execute("SELECT * FROM customers ORDER BY customer_id").fetchall()
            return [CustomerRecord(**dict(row)) for row in rows]

    def upsert(self, order: OrderRecord) -> None:
        with get_connection(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO orders (
                    order_id, customer_id, status, purchase_date, delivered_date,
                    estimated_delivery_date, amount, currency, category, order_note,
                    refunded_amount, shipping_carrier, tracking_code, items_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(order_id) DO UPDATE SET
                    status = excluded.status,
                    purchase_date = excluded.purchase_date,
                    delivered_date = excluded.delivered_date,
                    estimated_delivery_date = excluded.estimated_delivery_date,
                    amount = excluded.amount,
                    category = excluded.category,
                    order_note = excluded.order_note,
                    shipping_carrier = excluded.shipping_carrier,
                    tracking_code = excluded.tracking_code,
                    items_json = excluded.items_json
                """,
                (
                    order.order_id,
                    order.customer_id,
                    order.status,
                    order.purchase_date,
                    order.delivered_date,
                    order.estimated_delivery_date,
                    order.amount,
                    order.currency,
                    order.category,
                    order.order_note,
                    order.refunded_amount,
                    order.shipping_carrier,
                    order.tracking_code,
                    json.dumps([item.model_dump() for item in order.items]),
                ),
            )

    def get(self, order_id: str) -> OrderRecord | None:
        with get_connection(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM orders WHERE order_id = ?",
                (order_id,),
            ).fetchone()
            return _row_to_order(row) if row else None

    def list_for_customer(self, customer_id: str) -> list[OrderRecord]:
        with get_connection(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM orders WHERE customer_id = ? ORDER BY purchase_date DESC",
                (customer_id,),
            ).fetchall()
            return [_row_to_order(row) for row in rows]


class RefundRepository:
    """Parameterized SQLite repository for refunds."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        init_app_db(self.db_path)

    def list_for_order(self, order_id: str) -> list[RefundRecord]:
        with get_connection(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM refunds WHERE order_id = ? ORDER BY created_at DESC",
                (order_id,),
            ).fetchall()
            return [_row_to_refund(row) for row in rows]

    def list_for_customer(self, customer_id: str) -> list[RefundRecord]:
        with get_connection(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM refunds WHERE customer_id = ? ORDER BY created_at DESC",
                (customer_id,),
            ).fetchall()
            return [_row_to_refund(row) for row in rows]


class SupportTicketRepository:
    """Parameterized SQLite repository for support cases / escalations."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        init_app_db(self.db_path)

    def get(self, ticket_id: str) -> SupportCaseRecord | None:
        with get_connection(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM support_cases WHERE ticket_id = ?",
                (ticket_id,),
            ).fetchone()
            return _row_to_ticket(row) if row else None

    def list_for_customer(self, customer_id: str) -> list[SupportCaseRecord]:
        with get_connection(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM support_cases WHERE customer_id = ? ORDER BY created_at DESC",
                (customer_id,),
            ).fetchall()
            return [_row_to_ticket(row) for row in rows]


class EventRepository:
    """Append-only event store repository (Layer 1)."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        init_app_db(self.db_path)

    def append(
        self,
        *,
        customer_id: str,
        thread_id: str,
        event_type: EventType,
        order_id: str | None = None,
        amount: float | None = None,
        actor: str = "agent",
        metadata: dict[str, Any] | None = None,
    ) -> EventRecord:
        event_id = f"EVT-{uuid.uuid4().hex[:12].upper()}"
        ts = utc_now_iso()
        meta = metadata or {}
        with get_connection(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO events (
                    event_id, customer_id, thread_id, event_type, order_id,
                    amount, timestamp, actor, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    customer_id,
                    thread_id,
                    event_type,
                    order_id,
                    amount,
                    ts,
                    actor,
                    json.dumps(meta),
                ),
            )
        return EventRecord(
            event_id=event_id,
            customer_id=customer_id,
            thread_id=thread_id,
            event_type=event_type,
            order_id=order_id,
            amount=amount,
            timestamp=ts,
            actor=actor,
            metadata=meta,
        )

    def list(
        self,
        *,
        customer_id: str | None = None,
        thread_id: str | None = None,
        order_id: str | None = None,
        limit: int = 100,
    ) -> list[EventRecord]:
        return self.list_events(
            customer_id=customer_id,
            thread_id=thread_id,
            order_id=order_id,
            limit=limit,
        )

    def list_events(
        self,
        *,
        customer_id: str | None = None,
        thread_id: str | None = None,
        order_id: str | None = None,
        limit: int = 100,
    ) -> list[EventRecord]:
        clauses: list[str] = []
        params: list[Any] = []
        if customer_id:
            clauses.append("customer_id = ?")
            params.append(customer_id)
        if thread_id:
            clauses.append("thread_id = ?")
            params.append(thread_id)
        if order_id:
            clauses.append("order_id = ?")
            params.append(order_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT * FROM events {where} ORDER BY seq ASC LIMIT ?"
        params.append(limit)
        with get_connection(self.db_path) as conn:
            rows = conn.execute(sql, params).fetchall()
            return [
                EventRecord(
                    seq=row["seq"],
                    event_id=row["event_id"],
                    customer_id=row["customer_id"],
                    thread_id=row["thread_id"],
                    event_type=row["event_type"],
                    order_id=row["order_id"],
                    amount=row["amount"],
                    timestamp=row["timestamp"],
                    actor=row["actor"],
                    metadata=json.loads(row["metadata_json"]),
                )
                for row in rows
            ]


class ToolOperationRepository:
    """Idempotency ledger repository for mutating tool calls."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        init_app_db(self.db_path)

    def get(self, idempotency_key: str) -> ToolOperationRecord | None:
        with get_connection(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM tool_operations WHERE idempotency_key = ?",
                (idempotency_key,),
            ).fetchone()
            if row is None:
                return None
            return ToolOperationRecord(
                idempotency_key=row["idempotency_key"],
                tool_name=row["tool_name"],
                customer_id=row["customer_id"],
                order_id=row["order_id"],
                status=row["status"],
                response=json.loads(row["response_json"]),
                created_at=row["created_at"],
            )

    def save(self, op: ToolOperationRecord) -> None:
        with get_connection(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO tool_operations (
                    idempotency_key, tool_name, customer_id, order_id, status, response_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    op.idempotency_key,
                    op.tool_name,
                    op.customer_id,
                    op.order_id,
                    op.status,
                    json.dumps(op.response),
                    op.created_at,
                ),
            )


class EpisodeRepository:
    """Layer 3 episodic memory repository."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        init_app_db(self.db_path)

    def add(
        self,
        *,
        customer_id: str,
        thread_id: str,
        content: str,
        entities: list[str],
        topic: str,
        importance: float,
    ) -> EpisodeRecord:
        episode_id = f"EP-{uuid.uuid4().hex[:10].upper()}"
        now = utc_now_iso()
        with get_connection(self.db_path) as conn:
            existing = conn.execute(
                "SELECT * FROM episodes WHERE thread_id = ? AND content = ?",
                (thread_id, content),
            ).fetchone()
            if existing is not None:
                return EpisodeRecord(
                    episode_id=existing["episode_id"],
                    customer_id=existing["customer_id"],
                    thread_id=existing["thread_id"],
                    content=existing["content"],
                    entities=json.loads(existing["entities_json"]),
                    topic=existing["topic"],
                    importance=float(existing["importance"]),
                    created_at=existing["created_at"],
                )
            conn.execute(
                """
                INSERT INTO episodes (
                    episode_id, customer_id, thread_id, content, entities_json, topic, importance, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    episode_id,
                    customer_id,
                    thread_id,
                    content,
                    json.dumps(entities),
                    topic,
                    importance,
                    now,
                ),
            )
        return EpisodeRecord(
            episode_id=episode_id,
            customer_id=customer_id,
            thread_id=thread_id,
            content=content,
            entities=entities,
            topic=topic,
            importance=importance,
            created_at=now,
        )

    def list_episodes(
        self,
        *,
        customer_id: str | None = None,
        thread_id: str | None = None,
        min_importance: float = 0.0,
        limit: int = 50,
    ) -> list[EpisodeRecord]:
        clauses = ["importance >= ?"]
        params: list[Any] = [min_importance]
        if customer_id:
            clauses.append("customer_id = ?")
            params.append(customer_id)
        if thread_id:
            clauses.append("thread_id = ?")
            params.append(thread_id)
        sql = (
            f"SELECT * FROM episodes WHERE {' AND '.join(clauses)} "
            "ORDER BY importance DESC, created_at DESC LIMIT ?"
        )
        params.append(limit)
        with get_connection(self.db_path) as conn:
            rows = conn.execute(sql, params).fetchall()
            return [
                EpisodeRecord(
                    episode_id=row["episode_id"],
                    customer_id=row["customer_id"],
                    thread_id=row["thread_id"],
                    content=row["content"],
                    entities=json.loads(row["entities_json"]),
                    topic=row["topic"],
                    importance=float(row["importance"]),
                    created_at=row["created_at"],
                )
                for row in rows
            ]


class SupportRepository:
    """Unified transactional data access facade for Layers 1, 2, 3, and archived turns."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        init_app_db(self.db_path)
        self.orders = OrderRepository(self.db_path)
        self.refunds = RefundRepository(self.db_path)
        self.tickets = SupportTicketRepository(self.db_path)
        self.events = EventRepository(self.db_path)
        self.tool_operations = ToolOperationRepository(self.db_path)
        self.episodes = EpisodeRepository(self.db_path)

    def upsert_customer(self, customer: CustomerRecord) -> None:
        self.orders.upsert_customer(customer)

    def upsert_order(self, order: OrderRecord) -> None:
        self.orders.upsert(order)

    def get_customer(self, customer_id: str) -> CustomerRecord | None:
        return self.orders.get_customer(customer_id)

    def list_customers(self) -> list[CustomerRecord]:
        return self.orders.list_customers()

    def get_order(self, order_id: str) -> OrderRecord | None:
        return self.orders.get(order_id)

    def list_orders_for_customer(self, customer_id: str) -> list[OrderRecord]:
        return self.orders.list_for_customer(customer_id)

    def get_tool_operation(self, idempotency_key: str) -> ToolOperationRecord | None:
        return self.tool_operations.get(idempotency_key)

    def list_tool_operations(self, customer_id: str | None = None, limit: int = 50) -> list[ToolOperationRecord]:
        sql = (
            "SELECT * FROM tool_operations WHERE customer_id = ? ORDER BY created_at DESC LIMIT ?"
            if customer_id
            else "SELECT * FROM tool_operations ORDER BY created_at DESC LIMIT ?"
        )
        params: list[Any] = [customer_id, limit] if customer_id else [limit]
        with get_connection(self.db_path) as conn:
            rows = conn.execute(sql, params).fetchall()
            return [
                ToolOperationRecord(
                    idempotency_key=row["idempotency_key"],
                    tool_name=row["tool_name"],
                    customer_id=row["customer_id"],
                    order_id=row["order_id"],
                    status=row["status"],
                    response=json.loads(row["response_json"]),
                    created_at=row["created_at"],
                )
                for row in rows
            ]

    def save_tool_operation(self, op: ToolOperationRecord) -> None:
        self.tool_operations.save(op)

    def get_refunds_for_order(self, order_id: str) -> list[RefundRecord]:
        return self.refunds.list_for_order(order_id)

    def get_refunds_for_customer(self, customer_id: str) -> list[RefundRecord]:
        return self.refunds.list_for_customer(customer_id)

    def get_support_ticket(self, ticket_id: str) -> SupportCaseRecord | None:
        return self.tickets.get(ticket_id)

    def list_support_tickets_for_customer(self, customer_id: str) -> list[SupportCaseRecord]:
        return self.tickets.list_for_customer(customer_id)

    def append_event(
        self,
        *,
        customer_id: str,
        thread_id: str,
        event_type: EventType,
        order_id: str | None = None,
        amount: float | None = None,
        actor: str = "agent",
        metadata: dict[str, Any] | None = None,
    ) -> EventRecord:
        return self.events.append(
            customer_id=customer_id,
            thread_id=thread_id,
            event_type=event_type,
            order_id=order_id,
            amount=amount,
            actor=actor,
            metadata=metadata,
        )

    def list(
        self,
        *,
        customer_id: str | None = None,
        thread_id: str | None = None,
        order_id: str | None = None,
        limit: int = 100,
    ) -> list[EventRecord]:
        return self.list_events(
            customer_id=customer_id,
            thread_id=thread_id,
            order_id=order_id,
            limit=limit,
        )

    def list_events(
        self,
        *,
        customer_id: str | None = None,
        thread_id: str | None = None,
        order_id: str | None = None,
        limit: int = 100,
    ) -> list[EventRecord]:
        return self.events.list_events(
            customer_id=customer_id,
            thread_id=thread_id,
            order_id=order_id,
            limit=limit,
        )

    def add_episode(
        self,
        *,
        customer_id: str,
        thread_id: str,
        content: str,
        entities: list[str],
        topic: str,
        importance: float,
    ) -> EpisodeRecord:
        return self.episodes.add(
            customer_id=customer_id,
            thread_id=thread_id,
            content=content,
            entities=entities,
            topic=topic,
            importance=importance,
        )

    def list_episodes(
        self,
        *,
        customer_id: str | None = None,
        thread_id: str | None = None,
        min_importance: float = 0.0,
        limit: int = 50,
    ) -> list[EpisodeRecord]:
        return self.episodes.list_episodes(
            customer_id=customer_id,
            thread_id=thread_id,
            min_importance=min_importance,
            limit=limit,
        )

    @staticmethod
    def _insert_tool_op_conn(
        conn: sqlite3.Connection,
        idempotency_key: str,
        tool_name: str,
        customer_id: str,
        order_id: str | None,
        status: str,
        result: dict[str, Any],
    ) -> None:
        conn.execute(
            """
            INSERT OR IGNORE INTO tool_operations (
                idempotency_key, tool_name, customer_id, order_id, status, response_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                idempotency_key,
                tool_name,
                customer_id,
                order_id,
                status,
                json.dumps(result),
                utc_now_iso(),
            ),
        )

    @staticmethod
    def _append_event_conn(
        conn: sqlite3.Connection,
        *,
        customer_id: str,
        thread_id: str,
        event_type: EventType,
        order_id: str | None,
        amount: float | None,
        actor: str,
        metadata: dict[str, Any],
    ) -> EventRecord:
        event_id = f"EVT-{uuid.uuid4().hex[:12].upper()}"
        ts = utc_now_iso()
        conn.execute(
            """
            INSERT INTO events (
                event_id, customer_id, thread_id, event_type, order_id,
                amount, timestamp, actor, metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                customer_id,
                thread_id,
                event_type,
                order_id,
                amount,
                ts,
                actor,
                json.dumps(metadata),
            ),
        )
        return EventRecord(
            event_id=event_id,
            customer_id=customer_id,
            thread_id=thread_id,
            event_type=event_type,
            order_id=order_id,
            amount=amount,
            timestamp=ts,
            actor=actor,
            metadata=metadata,
        )

    def create_refund_atomic(
        self,
        *,
        customer_id: str,
        order_id: str,
        amount: float,
        currency: str,
        reason: str,
        policy_version: str,
        idempotency_key: str,
        allow_duplicate_refunds: bool,
        max_allowed_amount: float,
        thread_id: str = "mcp",
        rules: Any = None,
        as_of: datetime | None = None,
    ) -> tuple[RefundRecord | None, dict[str, Any], bool]:
        """Atomically re-validate and persist a refund within a BEGIN IMMEDIATE transaction."""
        from src.config import RefundPolicyConfig
        from src.policies.refund_policy import evaluate_refund_eligibility

        with transaction(self.db_path) as conn:
            existing_op = conn.execute(
                "SELECT * FROM tool_operations WHERE idempotency_key = ?",
                (idempotency_key,),
            ).fetchone()
            if existing_op is not None:
                payload = json.loads(existing_op["response_json"])
                payload["idempotent_replay"] = True
                refund_row = conn.execute(
                    "SELECT * FROM refunds WHERE idempotency_key = ?",
                    (idempotency_key,),
                ).fetchone()
                return (
                    _row_to_refund(refund_row) if refund_row else None,
                    payload,
                    True,
                )

            order_row = conn.execute(
                "SELECT * FROM orders WHERE order_id = ?",
                (order_id,),
            ).fetchone()
            order_obj = _row_to_order(order_row) if order_row else None
            existing_rows = conn.execute(
                "SELECT * FROM refunds WHERE order_id = ?",
                (order_id,),
            ).fetchall()
            existing_refunds = [_row_to_refund(r) for r in existing_rows]

            policy_cfg = (
                rules.refund_policy
                if rules is not None
                else RefundPolicyConfig(
                    maximum_refund_amount=max_allowed_amount,
                    currency=currency,
                    allow_duplicate_refunds=allow_duplicate_refunds,
                )
            )
            elig = evaluate_refund_eligibility(
                order=order_obj,
                existing_refunds=existing_refunds,
                pending_refunds=[],
                requested_amount=amount,
                policy=policy_cfg,
                reference_now=as_of,
                customer_id=customer_id,
                rules=rules,
            )
            if not elig.eligible:
                result: dict[str, Any] = {
                    "ok": False,
                    "approved": False,
                    "status": "rejected",
                    "reason_code": elig.reason_code,
                    "reason": elig.message,
                    "message": elig.message,
                    "max_allowed_amount": elig.max_allowed_amount,
                    "policy_version": policy_version,
                    "idempotency_key": idempotency_key,
                    "idempotent_replay": False,
                }
                self._insert_tool_op_conn(
                    conn, idempotency_key, "create_refund", customer_id, order_id, "blocked", result
                )
                self._append_event_conn(
                    conn,
                    customer_id=customer_id,
                    thread_id=thread_id,
                    event_type="refund_rejected" if order_obj and order_obj.customer_id == customer_id else "policy_blocked",
                    order_id=order_id,
                    amount=amount,
                    actor="mcp_refund_gate",
                    metadata={
                        "reason_code": elig.reason_code,
                        "reason": elig.message,
                        "policy_version": policy_version,
                        "idempotency_key": idempotency_key,
                    },
                )
                return None, result, False

            now = utc_now_iso()
            refund_id = f"REF-{uuid.uuid4().hex[:10].upper()}"
            rounded_amount = round(amount, 2)
            conn.execute(
                """
                INSERT INTO refunds (
                    refund_id, order_id, customer_id, amount, currency,
                    status, reason, policy_version, idempotency_key, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    refund_id,
                    order_id,
                    customer_id,
                    rounded_amount,
                    currency,
                    "completed",
                    reason,
                    policy_version,
                    idempotency_key,
                    now,
                ),
            )
            conn.execute(
                "UPDATE orders SET refunded_amount = refunded_amount + ? WHERE order_id = ?",
                (rounded_amount, order_id),
            )
            refund_record = RefundRecord(
                refund_id=refund_id,
                order_id=order_id,
                customer_id=customer_id,
                amount=rounded_amount,
                currency=currency,
                status="completed",
                reason=reason,
                policy_version=policy_version,
                idempotency_key=idempotency_key,
                created_at=now,
            )
            result = {
                "ok": True,
                "refund_id": refund_id,
                "order_id": order_id,
                "customer_id": customer_id,
                "amount": rounded_amount,
                "currency": currency,
                "status": "completed",
                "policy_version": policy_version,
                "idempotency_key": idempotency_key,
                "idempotent_replay": False,
                "created_at": now,
            }
            self._insert_tool_op_conn(
                conn, idempotency_key, "create_refund", customer_id, order_id, "completed", result
            )
            self._append_event_conn(
                conn,
                customer_id=customer_id,
                thread_id=thread_id,
                event_type="refund_approved",
                order_id=order_id,
                amount=rounded_amount,
                actor="mcp_refund_gate",
                metadata={
                    "refund_id": refund_id,
                    "policy_version": policy_version,
                    "idempotency_key": idempotency_key,
                },
            )
            self._append_event_conn(
                conn,
                customer_id=customer_id,
                thread_id=thread_id,
                event_type="refund_completed",
                order_id=order_id,
                amount=rounded_amount,
                actor="mcp_refund_gate",
                metadata={
                    "refund_id": refund_id,
                    "policy_version": policy_version,
                    "idempotency_key": idempotency_key,
                },
            )
            return refund_record, result, False

    def create_support_ticket_atomic(
        self,
        *,
        customer_id: str,
        order_id: str | None,
        reason: str,
        priority: str,
        idempotency_key: str,
        policy_version: str,
        thread_id: str = "mcp",
    ) -> tuple[SupportCaseRecord | None, dict[str, Any], bool]:
        """Atomically create a support ticket with idempotency protection."""
        with transaction(self.db_path) as conn:
            existing_op = conn.execute(
                "SELECT * FROM tool_operations WHERE idempotency_key = ?",
                (idempotency_key,),
            ).fetchone()
            if existing_op is not None:
                payload = json.loads(existing_op["response_json"])
                payload["idempotent_replay"] = True
                ticket_row = conn.execute(
                    "SELECT * FROM support_cases WHERE idempotency_key = ?",
                    (idempotency_key,),
                ).fetchone()
                return (
                    _row_to_ticket(ticket_row) if ticket_row else None,
                    payload,
                    True,
                )

            customer_row = conn.execute(
                "SELECT customer_id FROM customers WHERE customer_id = ?",
                (customer_id,),
            ).fetchone()
            if customer_row is None:
                result: dict[str, Any] = {
                    "ok": False,
                    "reason": "Customer not found.",
                    "policy_version": policy_version,
                    "idempotency_key": idempotency_key,
                }
                self._insert_tool_op_conn(
                    conn, idempotency_key, "create_support_ticket", customer_id, order_id, "blocked", result
                )
                return None, result, False

            if order_id:
                order_row = conn.execute(
                    "SELECT customer_id FROM orders WHERE order_id = ?",
                    (order_id,),
                ).fetchone()
                if order_row is None or order_row["customer_id"] != customer_id:
                    result = {
                        "ok": False,
                        "reason": "Order not found or does not belong to customer.",
                        "policy_version": policy_version,
                        "idempotency_key": idempotency_key,
                    }
                    self._insert_tool_op_conn(
                        conn,
                        idempotency_key,
                        "create_support_ticket",
                        customer_id,
                        order_id,
                        "blocked",
                        result,
                    )
                    self._append_event_conn(
                        conn,
                        customer_id=customer_id,
                        thread_id=thread_id,
                        event_type="policy_blocked",
                        order_id=order_id,
                        amount=None,
                        actor="mcp_support_gate",
                        metadata={
                            "reason": result["reason"],
                            "policy_version": policy_version,
                            "idempotency_key": idempotency_key,
                        },
                    )
                    return None, result, False

            now = utc_now_iso()
            ticket_id = f"TKT-{uuid.uuid4().hex[:10].upper()}"
            conn.execute(
                """
                INSERT INTO support_cases (
                    ticket_id, customer_id, order_id, status, priority, summary,
                    idempotency_key, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ticket_id,
                    customer_id,
                    order_id,
                    "escalated",
                    priority,
                    reason,
                    idempotency_key,
                    now,
                    now,
                ),
            )
            record = SupportCaseRecord(
                ticket_id=ticket_id,
                customer_id=customer_id,
                order_id=order_id,
                status="escalated",
                priority=priority,  # type: ignore[arg-type]
                summary=reason,
                idempotency_key=idempotency_key,
                created_at=now,
                updated_at=now,
            )
            result = {
                "ok": True,
                "ticket_id": ticket_id,
                "customer_id": customer_id,
                "order_id": order_id,
                "priority": priority,
                "status": "escalated",
                "reason": reason,
                "summary": reason,
                "policy_version": policy_version,
                "idempotency_key": idempotency_key,
                "idempotent_replay": False,
                "created_at": now,
            }
            self._insert_tool_op_conn(
                conn, idempotency_key, "create_support_ticket", customer_id, order_id, "completed", result
            )
            self._append_event_conn(
                conn,
                customer_id=customer_id,
                thread_id=thread_id,
                event_type="escalation_created",
                order_id=order_id,
                amount=None,
                actor="mcp_support_gate",
                metadata={
                    "ticket_id": ticket_id,
                    "priority": priority,
                    "policy_version": policy_version,
                    "idempotency_key": idempotency_key,
                },
            )
            return record, result, False

    def append_turn(
        self,
        *,
        thread_id: str,
        customer_id: str,
        turn_index: int,
        role: str,
        content: str,
        entities: list[str],
    ) -> ConversationTurnRecord:
        now = utc_now_iso()
        with get_connection(self.db_path) as conn:
            cur = conn.execute(
                """
                INSERT INTO conversation_turns (
                    thread_id, customer_id, turn_index, role, content, entities_json, compacted, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, 0, ?)
                """,
                (thread_id, customer_id, turn_index, role, content, json.dumps(entities), now),
            )
            turn_id = cur.lastrowid
        return ConversationTurnRecord(
            turn_id=turn_id,
            thread_id=thread_id,
            customer_id=customer_id,
            turn_index=turn_index,
            role=role,  # type: ignore[arg-type]
            content=content,
            entities=entities,
            compacted=False,
            created_at=now,
        )

    def list_turns(
        self,
        *,
        thread_id: str,
        customer_id: str | None = None,
        limit: int = 200,
    ) -> list[ConversationTurnRecord]:
        clauses = ["thread_id = ?"]
        params: list[Any] = [thread_id]
        if customer_id:
            clauses.append("customer_id = ?")
            params.append(customer_id)
        sql = (
            f"SELECT * FROM conversation_turns WHERE {' AND '.join(clauses)} "
            "ORDER BY turn_index ASC, turn_id ASC LIMIT ?"
        )
        params.append(limit)
        with get_connection(self.db_path) as conn:
            rows = conn.execute(sql, params).fetchall()
            return [
                ConversationTurnRecord(
                    turn_id=row["turn_id"],
                    thread_id=row["thread_id"],
                    customer_id=row["customer_id"],
                    turn_index=row["turn_index"],
                    role=row["role"],
                    content=row["content"],
                    entities=json.loads(row["entities_json"]),
                    compacted=bool(row["compacted"]),
                    created_at=row["created_at"],
                )
                for row in rows
            ]

    def mark_turns_compacted(self, thread_id: str, up_to_turn_index: int) -> None:
        with get_connection(self.db_path) as conn:
            conn.execute(
                """
                UPDATE conversation_turns
                SET compacted = 1
                WHERE thread_id = ? AND turn_index <= ?
                """,
                (thread_id, up_to_turn_index),
            )

    def save_compaction_snapshot(
        self,
        *,
        thread_id: str,
        customer_id: str,
        summary: dict[str, Any],
        compacted_up_to_turn: int,
        token_estimate_before: int,
        token_estimate_after: int,
    ) -> CompactionSnapshotRecord:
        snapshot_id = f"CMP-{uuid.uuid4().hex[:10].upper()}"
        now = utc_now_iso()
        with get_connection(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO compaction_snapshots (
                    snapshot_id, thread_id, customer_id, summary_json,
                    compacted_up_to_turn, token_estimate_before, token_estimate_after, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot_id,
                    thread_id,
                    customer_id,
                    json.dumps(summary),
                    compacted_up_to_turn,
                    token_estimate_before,
                    token_estimate_after,
                    now,
                ),
            )
        return CompactionSnapshotRecord(
            snapshot_id=snapshot_id,
            thread_id=thread_id,
            customer_id=customer_id,
            summary=summary,
            compacted_up_to_turn=compacted_up_to_turn,
            token_estimate_before=token_estimate_before,
            token_estimate_after=token_estimate_after,
            created_at=now,
        )

    def get_latest_compaction_snapshot(self, thread_id: str) -> CompactionSnapshotRecord | None:
        with get_connection(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT * FROM compaction_snapshots
                WHERE thread_id = ?
                ORDER BY created_at DESC, rowid DESC
                LIMIT 1
                """,
                (thread_id,),
            ).fetchone()
            if row is None:
                return None
            return CompactionSnapshotRecord(
                snapshot_id=row["snapshot_id"],
                thread_id=row["thread_id"],
                customer_id=row["customer_id"],
                summary=json.loads(row["summary_json"]),
                compacted_up_to_turn=row["compacted_up_to_turn"],
                token_estimate_before=row["token_estimate_before"],
                token_estimate_after=row["token_estimate_after"],
                created_at=row["created_at"],
            )
