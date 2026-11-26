"""Adaptive token-budget compaction with Layer 1/2 SQLite cross-verification and provenance."""

from __future__ import annotations

import json
import re
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from src.config import ContextBudgetConfig
from src.database.models import CompactionSnapshotRecord
from src.database.repositories import SupportRepository
from src.memory.memory_schemas import (
    CompactedMemoryPayload,
    estimate_messages_tokens,
    extract_entities,
)

_FABRICATED_REFUND_CLAIM_RE = re.compile(
    r"\b(?:refund(?:ed)?|reimbursed|credited)\b.*\b(?:ORD-\d+|REF-[A-Z0-9]+)\b|\b(?:REF-[A-Z0-9]+)\b",
    re.IGNORECASE,
)


def _utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


class CompactionOutcome(BaseModel):
    """Structured result returned by TokenBudgetCompactor.compact()."""

    compacted: bool
    preserved_messages: list[dict[str, str]]
    payload: CompactedMemoryPayload | None = None
    snapshot: CompactionSnapshotRecord | None = None
    tokens_before: int
    tokens_after: int


class TokenBudgetCompactor:
    """Adaptive token-budget compactor cross-checking claims against Layer 1/2 SQLite truth."""

    def __init__(self, repo: SupportRepository | None = None) -> None:
        self.repo = repo

    @staticmethod
    def should_compact(
        messages: list[dict[str, str]],
        context_config: ContextBudgetConfig,
    ) -> bool:
        """Return True when estimated tokens cross compaction_trigger_tokens and older messages exist."""
        if len(messages) <= context_config.recent_messages_to_preserve:
            return False
        return estimate_messages_tokens(messages) >= context_config.compaction_trigger_tokens

    @staticmethod
    def _load_db_truth(
        *,
        customer_id: str,
        thread_id: str,
        db_conn: sqlite3.Connection | SupportRepository | str | Path | None,
        fallback_repo: SupportRepository | None,
    ) -> dict[str, Any]:
        """Load authoritative Layer 1/2 orders, refunds, tickets, events, and Layer 3 episodes."""
        repo: SupportRepository | None = fallback_repo
        if isinstance(db_conn, SupportRepository):
            repo = db_conn
        elif isinstance(db_conn, (str, Path)):
            repo = SupportRepository(db_conn)

        if isinstance(db_conn, sqlite3.Connection):
            db_conn.row_factory = sqlite3.Row
            orders = {
                r["order_id"]: dict(r)
                for r in db_conn.execute(
                    "SELECT * FROM orders WHERE customer_id = ?", (customer_id,)
                ).fetchall()
            }
            refunds = [
                dict(r)
                for r in db_conn.execute(
                    "SELECT * FROM refunds WHERE customer_id = ?", (customer_id,)
                ).fetchall()
            ]
            tickets = [
                dict(r)
                for r in db_conn.execute(
                    "SELECT * FROM support_cases WHERE customer_id = ?", (customer_id,)
                ).fetchall()
            ]
            events = [
                dict(r)
                for r in db_conn.execute(
                    "SELECT * FROM events WHERE customer_id = ? AND thread_id = ? ORDER BY seq ASC",
                    (customer_id, thread_id),
                ).fetchall()
            ]
            episodes = [
                dict(r)
                for r in db_conn.execute(
                    "SELECT * FROM episodes WHERE customer_id = ? AND thread_id = ? ORDER BY created_at ASC",
                    (customer_id, thread_id),
                ).fetchall()
            ]
            return {
                "orders": orders,
                "refunds": refunds,
                "tickets": tickets,
                "events": events,
                "episodes": episodes,
            }

        assert repo is not None, "Either db_conn or repo must be provided to compact()."
        return {
            "orders": {o.order_id: o.model_dump() for o in repo.list_orders_for_customer(customer_id)},
            "refunds": [r.model_dump() for r in repo.get_refunds_for_customer(customer_id)],
            "tickets": [
                t.model_dump() for t in repo.list_support_tickets_for_customer(customer_id)
            ],
            "events": [
                e.model_dump()
                for e in repo.list_events(customer_id=customer_id, thread_id=thread_id, limit=200)
            ],
            "episodes": [
                ep.model_dump()
                for ep in repo.list_episodes(customer_id=customer_id, thread_id=thread_id, limit=100)
            ],
        }

    @staticmethod
    def _cross_verify_payload(
        *,
        customer_id: str,
        raw_payload: CompactedMemoryPayload,
        candidate_claims: list[str],
        truth: dict[str, Any],
    ) -> CompactedMemoryPayload:
        """Cross-check entities, promises, and refund claims against Layer 1/2 SQLite records."""
        orders: dict[str, dict[str, Any]] = truth["orders"]
        refunds: list[dict[str, Any]] = truth["refunds"]
        tickets: list[dict[str, Any]] = truth["tickets"]

        valid_refund_ids = {r["refund_id"] for r in refunds}
        refunded_order_ids = {
            r["order_id"] for r in refunds if r.get("status") in {"approved", "completed"}
        }
        valid_ticket_ids = {t["ticket_id"] for t in tickets}

        verified_entities: list[str] = []
        for ent in raw_payload.entities:
            if (
                ent == customer_id
                or ent in orders
                or ent in valid_refund_ids
                or ent in valid_ticket_ids
            ):
                verified_entities.append(ent)

        # Strip unverified or fabricated refund claims not backed by Layer 2 SQLite rows
        stripped_claims: list[str] = []
        verified_promises: list[dict[str, str]] = []
        for p in raw_payload.promises_made:
            text = p.get("promise", "")
            p_ents = extract_entities(text)
            p_refs = [e for e in p_ents if e.startswith("REF-")]
            p_ords = [e for e in p_ents if e.startswith("ORD-")]

            is_refund_claim = bool(_FABRICATED_REFUND_CLAIM_RE.search(text))
            if is_refund_claim:
                ref_ok = all(rid in valid_refund_ids for rid in p_refs) if p_refs else True
                ord_ok = any(oid in refunded_order_ids for oid in p_ords) if p_ords else bool(refunds)
                if not (ref_ok and ord_ok):
                    stripped_claims.append(text)
                    continue
            verified_promises.append(p)

        for claim in candidate_claims:
            if _FABRICATED_REFUND_CLAIM_RE.search(claim):
                c_ents = extract_entities(claim)
                c_refs = [e for e in c_ents if e.startswith("REF-")]
                c_ords = [e for e in c_ents if e.startswith("ORD-")]
                ref_ok = all(rid in valid_refund_ids for rid in c_refs) if c_refs else True
                ord_ok = any(oid in refunded_order_ids for oid in c_ords) if c_ords else bool(refunds)
                if not (ref_ok and ord_ok):
                    stripped_claims.append(claim)

        # Authoritative important_events always derived from verified Layer 1/2 SQLite records
        verified_events: list[str] = []
        for r in refunds:
            verified_events.append(
                f"Verified refund {r['refund_id']} on order {r['order_id']}: "
                f"{float(r['amount']):.2f} {r.get('currency', 'BRL')} ({r['status']})."
            )
        for t in tickets:
            verified_events.append(
                f"Verified ticket {t['ticket_id']} (order={t.get('order_id') or 'N/A'}, "
                f"status={t['status']}, priority={t['priority']})."
            )
        for order_id, order in orders.items():
            if not any(r["order_id"] == order_id for r in refunds):
                verified_events.append(
                    f"Verified order {order_id}: status={order['status']}, "
                    f"amount={float(order['amount']):.2f} {order.get('currency', 'BRL')}."
                )

        # Ensure verified promises include real SQLite refund/ticket commitments
        existing_promise_texts = {p.get("promise", "") for p in verified_promises}
        for r in refunds:
            summary = (
                f"Refund {r['refund_id']} ({float(r['amount']):.2f} {r.get('currency', 'BRL')}) "
                f"for {r['order_id']}"
            )
            if summary not in existing_promise_texts:
                verified_promises.append(
                    {"promise": summary, "status": str(r["status"]), "refund_id": str(r["refund_id"])}
                )
        for t in tickets:
            summary = f"Support ticket {t['ticket_id']}: {t['summary']}"
            if summary not in existing_promise_texts:
                verified_promises.append(
                    {"promise": summary, "status": str(t["status"]), "ticket_id": str(t["ticket_id"])}
                )

        return CompactedMemoryPayload(
            customer_preferences=list(dict.fromkeys(raw_payload.customer_preferences))[:6],
            active_issues=list(dict.fromkeys(raw_payload.active_issues))[:6],
            promises_made=verified_promises[:8],
            unresolved_questions=list(dict.fromkeys(raw_payload.unresolved_questions))[:5],
            important_events=verified_events[:10],
            entities=list(dict.fromkeys(verified_entities)),
            conversation_summary=raw_payload.conversation_summary,
            source_thread_id=raw_payload.source_thread_id,
            source_turn_range=raw_payload.source_turn_range,
            compacted_at=raw_payload.compacted_at,
            stripped_unverified_claims=list(dict.fromkeys(stripped_claims)),
        )

    def compact(
        self,
        *,
        customer_id: str,
        thread_id: str,
        messages: list[dict[str, str]],
        db_conn: sqlite3.Connection | SupportRepository | str | Path | None = None,
        context_config: ContextBudgetConfig | None = None,
        llm_client: Any = None,
    ) -> CompactionOutcome:
        """Compact older turns beyond recent_messages_to_preserve when token budget is exceeded."""
        cfg = context_config or ContextBudgetConfig()
        tokens_before = estimate_messages_tokens(messages)
        preserve_n = cfg.recent_messages_to_preserve

        if not self.should_compact(messages, cfg):
            return CompactionOutcome(
                compacted=False,
                preserved_messages=list(messages),
                tokens_before=tokens_before,
                tokens_after=tokens_before,
            )

        older = messages[:-preserve_n]
        preserved = list(messages[-preserve_n:])
        truth = self._load_db_truth(
            customer_id=customer_id,
            thread_id=thread_id,
            db_conn=db_conn,
            fallback_repo=self.repo,
        )

        # Extract candidate preferences, issues, promises, questions deterministically
        preferences: list[str] = []
        issues: list[str] = []
        promises: list[dict[str, str]] = []
        unresolved: list[str] = []
        candidate_claims: list[str] = []
        all_entities: list[str] = []

        for ep in truth["episodes"]:
            if ep["topic"] == "preference":
                preferences.append(ep["content"])
            elif ep["topic"] == "issue":
                issues.append(ep["content"])
            elif ep["topic"] == "commitment":
                candidate_claims.append(ep["content"])
                promises.append({"promise": ep["content"], "status": "recorded"})

        for msg in older:
            content = msg.get("content", "")
            all_entities.extend(extract_entities(content))
            lower = content.lower()
            if msg.get("role") == "user":
                if any(k in lower for k in ["prefer", "only email", "never sms", "never call"]):
                    preferences.append(content)
                if "?" in content:
                    unresolved.append(content)
            elif msg.get("role") == "assistant" and _FABRICATED_REFUND_CLAIM_RE.search(content):
                candidate_claims.append(content)
                promises.append({"promise": content[:180], "status": "claimed"})

        # Determine source turn range
        start_turn = 1
        end_turn = max(1, len(older) // 2)
        if self.repo is not None:
            prior = self.repo.get_latest_compaction_snapshot(thread_id)
            if prior is not None:
                start_turn = prior.compacted_up_to_turn + 1
            turns = self.repo.list_turns(thread_id=thread_id, customer_id=customer_id, limit=500)
            if len(turns) > preserve_n:
                end_turn = max(start_turn, turns[-preserve_n - 1].turn_index)

        now_iso = _utc_now_iso()
        prior_summary = ""
        if self.repo is not None:
            prior_snap = self.repo.get_latest_compaction_snapshot(thread_id)
            if prior_snap is not None:
                prior_summary = str(prior_snap.summary.get("conversation_summary", ""))

        # Optional LLM structured extraction hook if live llm_client supplied
        summary_text = (
            f"{prior_summary} Compacted turns {start_turn}-{end_turn} ({len(older)} messages)."
        ).strip()
        if llm_client is not None and hasattr(llm_client, "summarize_compaction"):
            try:
                llm_payload = llm_client.summarize_compaction(older)
                validated_llm = CompactedMemoryPayload.model_validate(llm_payload)
                preferences.extend(validated_llm.customer_preferences)
                issues.extend(validated_llm.active_issues)
                promises.extend(validated_llm.promises_made)
                unresolved.extend(validated_llm.unresolved_questions)
                summary_text = validated_llm.conversation_summary or summary_text
            except Exception:
                pass

        raw_payload = CompactedMemoryPayload(
            customer_preferences=preferences,
            active_issues=issues,
            promises_made=promises,
            unresolved_questions=unresolved[-3:],
            important_events=[],
            entities=list(dict.fromkeys(all_entities)),
            conversation_summary=summary_text,
            source_thread_id=thread_id,
            source_turn_range=(start_turn, end_turn),
            compacted_at=now_iso,
        )

        verified_payload = self._cross_verify_payload(
            customer_id=customer_id,
            raw_payload=raw_payload,
            candidate_claims=candidate_claims,
            truth=truth,
        )

        tokens_after = estimate_messages_tokens(preserved)
        snapshot: CompactionSnapshotRecord | None = None

        if isinstance(db_conn, sqlite3.Connection):
            snapshot_id = f"CMP-{uuid.uuid4().hex[:10].upper()}"
            db_conn.execute(
                """
                INSERT INTO compaction_snapshots (
                    snapshot_id, thread_id, customer_id, summary_json,
                    compacted_up_to_turn, token_estimate_before, token_estimate_after, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot_id,
                    thread_id,
                    customer_id,
                    verified_payload.model_dump_json(),
                    end_turn,
                    tokens_before,
                    tokens_after,
                    now_iso,
                ),
            )
            db_conn.execute(
                "UPDATE conversation_turns SET compacted = 1 WHERE thread_id = ? AND turn_index <= ?",
                (thread_id, end_turn),
            )
            db_conn.commit()
            snapshot = CompactionSnapshotRecord(
                snapshot_id=snapshot_id,
                thread_id=thread_id,
                customer_id=customer_id,
                summary=verified_payload.model_dump(),
                compacted_up_to_turn=end_turn,
                token_estimate_before=tokens_before,
                token_estimate_after=tokens_after,
                created_at=now_iso,
            )
        else:
            target_repo = (
                db_conn
                if isinstance(db_conn, SupportRepository)
                else (SupportRepository(db_conn) if isinstance(db_conn, (str, Path)) else self.repo)
            )
            if target_repo is not None:
                target_repo.mark_turns_compacted(thread_id, end_turn)
                snapshot = target_repo.save_compaction_snapshot(
                    thread_id=thread_id,
                    customer_id=customer_id,
                    summary=verified_payload.model_dump(),
                    compacted_up_to_turn=end_turn,
                    token_estimate_before=tokens_before,
                    token_estimate_after=tokens_after,
                )
                # Persist validated compaction provenance episode without deleting prior rows
                provenance_meta = json.dumps(
                    {
                        "source_thread_id": thread_id,
                        "source_turn_range": list(verified_payload.source_turn_range),
                        "compacted_at": now_iso,
                    }
                )
                target_repo.add_episode(
                    customer_id=customer_id,
                    thread_id=thread_id,
                    content=f"Compaction summary {provenance_meta}: {verified_payload.conversation_summary}",
                    entities=verified_payload.entities,
                    topic="compaction_summary",
                    importance=0.75,
                )

        return CompactionOutcome(
            compacted=True,
            preserved_messages=preserved,
            payload=verified_payload,
            snapshot=snapshot,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
        )


def maybe_compact_memory(
    *,
    repo: SupportRepository,
    customer_id: str,
    thread_id: str,
    messages: list[dict[str, str]],
    budget: ContextBudgetConfig,
) -> tuple[list[dict[str, str]], CompactionSnapshotRecord | None]:
    """Convenience wrapper executing TokenBudgetCompactor against SupportRepository."""
    compactor = TokenBudgetCompactor(repo=repo)
    outcome = compactor.compact(
        customer_id=customer_id,
        thread_id=thread_id,
        messages=messages,
        db_conn=repo,
        context_config=budget,
    )
    return outcome.preserved_messages, outcome.snapshot


def run_compaction(
    *,
    repo: SupportRepository,
    customer_id: str,
    thread_id: str,
    messages: list[dict[str, str]],
    budget: ContextBudgetConfig,
) -> tuple[list[dict[str, str]], CompactionSnapshotRecord | None]:
    """Alias for maybe_compact_memory."""
    return maybe_compact_memory(
        repo=repo,
        customer_id=customer_id,
        thread_id=thread_id,
        messages=messages,
        budget=budget,
    )
