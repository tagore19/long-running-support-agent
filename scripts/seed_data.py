"""Idempotent SQLite fixture seeder for data/olist_seed.db and data/app.db."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from src.config import get_settings
from src.database.connection import get_connection, init_app_db, init_seed_db
from src.database.models import CustomerRecord, OrderRecord
from src.database.repositories import OrderRepository, SupportRepository


def _now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def seed_databases(
    *,
    fixtures_path: str | Path = "data/fixtures/olist_orders.json",
    seed_db_path: str | Path | None = None,
    app_db_path: str | Path | None = None,
    reset: bool = False,
) -> dict[str, int]:
    """Idempotently seed Olist customers and orders into olist_seed.db and app.db."""
    settings = get_settings()
    project_root = Path(__file__).resolve().parent.parent

    f_path = Path(fixtures_path)
    if not f_path.is_absolute():
        f_path = project_root / f_path

    seed_path = Path(seed_db_path) if seed_db_path else settings.resolve_path(settings.seed_db_path)
    app_path = Path(app_db_path) if app_db_path else settings.resolve_path(settings.effective_db_path)

    if reset:
        for p in (seed_path, app_path):
            if p.exists():
                p.unlink()
            for suffix in ("-wal", "-shm"):
                sidecar = Path(f"{p}{suffix}")
                if sidecar.exists():
                    sidecar.unlink()

    init_seed_db(seed_path)
    init_app_db(app_path)

    with f_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    repo = SupportRepository(app_path)
    order_repo = OrderRepository(app_path)
    now_iso = _now_iso()

    customers: list[CustomerRecord] = []
    for raw_c in payload["customers"]:
        customer = CustomerRecord(
            customer_id=raw_c["customer_id"],
            full_name=raw_c["full_name"],
            email_masked=raw_c["email_masked"],
            tier=raw_c.get("tier", "standard"),
            created_at=now_iso,
        )
        customers.append(customer)
        repo.upsert_customer(customer)

    orders: list[OrderRecord] = []
    for raw_o in payload["orders"]:
        total_amount = round(float(raw_o["price"]) + float(raw_o.get("freight_value", 0.0)), 2)
        note = raw_o.get("order_note", "")
        tracking_code = raw_o.get("tracking_code")
        if not tracking_code and "Tracking " in note:
            tracking_code = note.split("Tracking ", 1)[1].split(" ", 1)[0].strip(".;,")
        order = OrderRecord(
            order_id=raw_o["order_id"],
            customer_id=raw_o["customer_id"],
            status=raw_o["order_status"],
            purchase_date=raw_o["order_purchase_timestamp"],
            delivered_date=raw_o.get("order_delivered_customer_date"),
            estimated_delivery_date=raw_o.get("order_estimated_delivery_date"),
            amount=total_amount,
            currency="BRL",
            category=raw_o["product_category_name"],
            order_note=note,
            refunded_amount=0.0,
            shipping_carrier=raw_o.get("shipping_carrier"),
            tracking_code=tracking_code,
        )
        orders.append(order)
        order_repo.upsert(order)

    # Mirror fixture rows idempotently into data/olist_seed.db
    with get_connection(seed_path) as seed_conn:
        for c in customers:
            seed_conn.execute(
                """
                INSERT INTO customers (customer_id, full_name, email_masked, tier, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(customer_id) DO UPDATE SET
                    full_name = excluded.full_name,
                    email_masked = excluded.email_masked,
                    tier = excluded.tier
                """,
                (c.customer_id, c.full_name, c.email_masked, c.tier, c.created_at),
            )
        for o in orders:
            seed_conn.execute(
                """
                INSERT INTO orders (
                    order_id, customer_id, status, purchase_date, delivered_date,
                    estimated_delivery_date, amount, currency, category, order_note, refunded_amount
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(order_id) DO UPDATE SET
                    customer_id = excluded.customer_id,
                    status = excluded.status,
                    purchase_date = excluded.purchase_date,
                    delivered_date = excluded.delivered_date,
                    estimated_delivery_date = excluded.estimated_delivery_date,
                    amount = excluded.amount,
                    category = excluded.category,
                    order_note = excluded.order_note
                """,
                (
                    o.order_id,
                    o.customer_id,
                    o.status,
                    o.purchase_date,
                    o.delivered_date,
                    o.estimated_delivery_date,
                    o.amount,
                    o.currency,
                    o.category,
                    o.order_note,
                    o.refunded_amount,
                ),
            )

    return {"customers": len(customers), "orders": len(orders)}


if __name__ == "__main__":
    counts = seed_databases()
    print(f"Idempotently seeded {counts['customers']} customers and {counts['orders']} orders.")
