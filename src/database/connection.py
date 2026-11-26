"""SQLite connection management, WAL configuration, and schema initialization."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

APP_SCHEMA_SQL = """
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS customers (
    customer_id TEXT PRIMARY KEY,
    full_name TEXT NOT NULL,
    email_masked TEXT NOT NULL,
    tier TEXT NOT NULL DEFAULT 'standard',
    city TEXT NOT NULL DEFAULT '',
    state TEXT NOT NULL DEFAULT '',
    preferred_language TEXT NOT NULL DEFAULT 'en',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS orders (
    order_id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL REFERENCES customers(customer_id),
    status TEXT NOT NULL,
    purchase_date TEXT NOT NULL,
    delivered_date TEXT,
    estimated_delivery_date TEXT,
    amount REAL NOT NULL,
    currency TEXT NOT NULL DEFAULT 'BRL',
    category TEXT NOT NULL DEFAULT 'general',
    order_note TEXT NOT NULL DEFAULT '',
    refunded_amount REAL NOT NULL DEFAULT 0.0,
    shipping_carrier TEXT,
    tracking_code TEXT,
    items_json TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_orders_customer_id ON orders(customer_id);

CREATE TABLE IF NOT EXISTS refunds (
    refund_id TEXT PRIMARY KEY,
    order_id TEXT NOT NULL REFERENCES orders(order_id),
    customer_id TEXT NOT NULL REFERENCES customers(customer_id),
    amount REAL NOT NULL,
    currency TEXT NOT NULL DEFAULT 'BRL',
    status TEXT NOT NULL,
    reason TEXT NOT NULL,
    policy_version TEXT NOT NULL DEFAULT '2026.1',
    idempotency_key TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_refunds_order_id ON refunds(order_id);
CREATE INDEX IF NOT EXISTS idx_refunds_customer_id ON refunds(customer_id);

CREATE TABLE IF NOT EXISTS support_cases (
    ticket_id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL REFERENCES customers(customer_id),
    order_id TEXT,
    status TEXT NOT NULL,
    priority TEXT NOT NULL,
    summary TEXT NOT NULL,
    idempotency_key TEXT UNIQUE,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_support_cases_customer_id ON support_cases(customer_id);
CREATE INDEX IF NOT EXISTS idx_support_cases_order_id ON support_cases(order_id);

CREATE TABLE IF NOT EXISTS tool_operations (
    idempotency_key TEXT PRIMARY KEY,
    tool_name TEXT NOT NULL,
    customer_id TEXT NOT NULL,
    order_id TEXT,
    status TEXT NOT NULL,
    response_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT NOT NULL UNIQUE,
    customer_id TEXT NOT NULL,
    thread_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    order_id TEXT,
    amount REAL,
    timestamp TEXT NOT NULL,
    actor TEXT NOT NULL,
    metadata_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_customer_thread ON events(customer_id, thread_id);
CREATE INDEX IF NOT EXISTS idx_events_order_id ON events(order_id);

CREATE TRIGGER IF NOT EXISTS prevent_events_update
BEFORE UPDATE ON events
BEGIN
    SELECT RAISE(ABORT, 'events table is append-only');
END;

CREATE TRIGGER IF NOT EXISTS prevent_events_delete
BEFORE DELETE ON events
BEGIN
    SELECT RAISE(ABORT, 'events table is append-only');
END;

CREATE TABLE IF NOT EXISTS episodes (
    episode_id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    thread_id TEXT NOT NULL,
    content TEXT NOT NULL,
    entities_json TEXT NOT NULL,
    topic TEXT NOT NULL,
    importance REAL NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_episodes_customer_thread ON episodes(customer_id, thread_id);

CREATE TABLE IF NOT EXISTS conversation_turns (
    turn_id INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id TEXT NOT NULL,
    customer_id TEXT NOT NULL,
    turn_index INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    entities_json TEXT NOT NULL,
    compacted INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_turns_thread_id ON conversation_turns(thread_id, turn_index);

CREATE TABLE IF NOT EXISTS compaction_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    thread_id TEXT NOT NULL,
    customer_id TEXT NOT NULL,
    summary_json TEXT NOT NULL,
    compacted_up_to_turn INTEGER NOT NULL,
    token_estimate_before INTEGER NOT NULL,
    token_estimate_after INTEGER NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_compaction_thread ON compaction_snapshots(thread_id, created_at);
"""

SEED_SCHEMA_SQL = """
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS customers (
    customer_id TEXT PRIMARY KEY,
    full_name TEXT NOT NULL,
    email_masked TEXT NOT NULL,
    tier TEXT NOT NULL DEFAULT 'standard',
    city TEXT NOT NULL DEFAULT '',
    state TEXT NOT NULL DEFAULT '',
    preferred_language TEXT NOT NULL DEFAULT 'en',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS orders (
    order_id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL REFERENCES customers(customer_id),
    status TEXT NOT NULL,
    purchase_date TEXT NOT NULL,
    delivered_date TEXT,
    estimated_delivery_date TEXT,
    amount REAL NOT NULL,
    currency TEXT NOT NULL DEFAULT 'BRL',
    category TEXT NOT NULL DEFAULT 'general',
    order_note TEXT NOT NULL DEFAULT '',
    refunded_amount REAL NOT NULL DEFAULT 0.0,
    shipping_carrier TEXT,
    tracking_code TEXT,
    items_json TEXT NOT NULL DEFAULT '[]'
);
"""


def _ensure_parent_dir(db_path: str | Path) -> Path:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _open_conn(db_path: str | Path, *, isolation_level: str | None = "") -> sqlite3.Connection:
    path = _ensure_parent_dir(db_path)
    conn = sqlite3.connect(
        str(path),
        timeout=30.0,
        isolation_level=isolation_level,
        check_same_thread=False,
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA busy_timeout = 30000;")
    return conn


@contextmanager
def get_connection(db_path: str | Path) -> Iterator[sqlite3.Connection]:
    """Open a configured SQLite connection with WAL mode, foreign keys, and commit/rollback."""
    conn = _open_conn(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


@contextmanager
def transaction(db_path: str | Path) -> Iterator[sqlite3.Connection]:
    """Open an explicit BEGIN IMMEDIATE write transaction with automatic commit/rollback."""
    conn = _open_conn(db_path, isolation_level=None)
    try:
        conn.execute("BEGIN IMMEDIATE;")
        yield conn
        conn.execute("COMMIT;")
    except Exception:
        conn.execute("ROLLBACK;")
        raise
    finally:
        conn.close()


def init_app_db(db_path: str | Path) -> None:
    """Create all runtime application tables and append-only event triggers."""
    path = _ensure_parent_dir(db_path)
    conn = sqlite3.connect(str(path), timeout=30.0, check_same_thread=False)
    try:
        conn.executescript(APP_SCHEMA_SQL)
        conn.commit()
    finally:
        conn.close()


def init_seed_db(db_path: str | Path) -> None:
    """Create the seed fixture tables in olist_seed.db."""
    path = _ensure_parent_dir(db_path)
    conn = sqlite3.connect(str(path), timeout=30.0, check_same_thread=False)
    try:
        conn.executescript(SEED_SCHEMA_SQL)
        conn.commit()
    finally:
        conn.close()
