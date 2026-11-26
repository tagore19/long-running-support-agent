"""Layer 5 deterministic Active Context Builder with strict token budget enforcement."""

from __future__ import annotations

from src.config import ContextBudgetConfig
from src.database.models import EpisodeRecord
from src.database.repositories import SupportRepository
from src.memory.memory_schemas import (
    AssembledContext,
    BuiltContext,
    CustomerStateSnapshot,
    RetrievedMemoryItem,
    estimate_messages_tokens,
    estimate_tokens,
)
from src.memory.retrieval import HybridMemoryRetriever
from src.policies.authorization import wrap_untrusted_data


def load_customer_state_snapshot(
    *,
    repo: SupportRepository,
    customer_id: str,
    thread_id: str | None = None,
) -> CustomerStateSnapshot:
    """Build authoritative Layer 2 CustomerStateSnapshot from SQLite + Layer 1 events."""
    orders = repo.list_orders_for_customer(customer_id)
    refunds = repo.get_refunds_for_customer(customer_id)
    tickets = repo.list_support_tickets_for_customer(customer_id)
    events = repo.list_events(customer_id=customer_id, thread_id=thread_id, limit=15)

    active_issue = None
    open_tickets = [t for t in tickets if t.status in {"open", "in_progress", "escalated"}]
    if open_tickets:
        active_issue = f"{open_tickets[0].ticket_id}: {open_tickets[0].summary}"

    return CustomerStateSnapshot(
        customer_id=customer_id,
        orders=orders,
        refunds=refunds,
        support_tickets=tickets,
        recent_events=events,
        active_issue=active_issue,
    )


def _format_verified_state(
    *,
    repo: SupportRepository,
    customer_id: str,
    thread_id: str | None = None,
) -> str:
    """Format authoritative Layer 2 SQLite state (never summarized/overridden by episodes)."""
    customer = repo.get_customer(customer_id)
    snapshot = load_customer_state_snapshot(
        repo=repo, customer_id=customer_id, thread_id=thread_id
    )

    lines: list[str] = ["[VERIFIED LAYER-2 SQLITE STATE — AUTHORITATIVE]"]
    if customer is not None:
        lines.append(
            f"Customer: {customer.customer_id} | Name: {customer.full_name} | Tier: {customer.tier}"
        )
    else:
        lines.append(f"Customer: {customer_id} (unregistered)")

    if snapshot.orders:
        lines.append("Orders:")
        for o in snapshot.orders:
            lines.append(
                f"  - {o.order_id}: status={o.status}, amount={o.amount:.2f} {o.currency}, "
                f"refunded={o.refunded_amount:.2f}, category={o.category}, "
                f"delivered_date={o.delivered_date or 'N/A'}"
            )
    if snapshot.refunds:
        lines.append("Refunds:")
        for r in snapshot.refunds:
            lines.append(
                f"  - {r.refund_id}: order={r.order_id}, amount={r.amount:.2f} {r.currency}, "
                f"status={r.status}, created_at={r.created_at}"
            )
    if snapshot.support_tickets:
        lines.append("Support Tickets:")
        for t in snapshot.support_tickets:
            lines.append(
                f"  - {t.ticket_id}: order={t.order_id or 'N/A'}, priority={t.priority}, "
                f"status={t.status}, summary={t.summary}"
            )
    if snapshot.recent_events:
        lines.append("Recent Verified Events:")
        for evt in snapshot.recent_events[-5:]:
            lines.append(
                f"  - {evt.event_id}: {evt.event_type} order={evt.order_id or 'N/A'} amount={evt.amount}"
            )
    return "\n".join(lines)


class ActiveContextBuilder:
    """Deterministic Layer 5 context builder enforcing strict token budgets and authority hierarchy."""

    def __init__(
        self,
        repo: SupportRepository,
        budget: ContextBudgetConfig | None = None,
    ) -> None:
        self.repo = repo
        self.budget = budget or ContextBudgetConfig()
        self.retriever = HybridMemoryRetriever(repo)

    def build(
        self,
        *,
        customer_id: str,
        thread_id: str,
        current_message: str,
        messages: list[dict[str, str]],
        token_budget: int | None = None,
        compacted_memories: list[dict[str, object]] | None = None,
    ) -> BuiltContext:
        """Deterministically assemble bounded prompt context following the 5-layer sequence."""
        effective_budget = token_budget if token_budget is not None else self.budget.max_memory_tokens

        # 1. Authoritative Layer 2 + Layer 1 state (never truncated)
        structured_state_text = _format_verified_state(
            repo=self.repo, customer_id=customer_id, thread_id=thread_id
        )

        # 2. Compaction summary (from state or latest SQLite compaction snapshot)
        snapshot = self.repo.get_latest_compaction_snapshot(thread_id)
        compaction_summary_text = ""
        latest_summary: dict[str, object] | None = None
        turn_label = "latest"
        if compacted_memories:
            latest_summary = compacted_memories[-1]
            turn_label = str(latest_summary.get("source_turn_range", "latest"))
        elif snapshot is not None:
            latest_summary = snapshot.summary
            turn_label = str(snapshot.compacted_up_to_turn)

        if latest_summary is not None:
            raw_summary = (
                f"Compacted Summary (turns {turn_label}):\n"
                f"Summary: {latest_summary.get('conversation_summary', '')}\n"
                f"Preferences: {latest_summary.get('customer_preferences', [])}\n"
                f"Active Issues: {latest_summary.get('active_issues', [])}\n"
                f"Promises Made: {latest_summary.get('promises_made', [])}\n"
                f"Important Events: {latest_summary.get('important_events', [])}"
            )
            compaction_summary_text = wrap_untrusted_data("compacted_memory", raw_summary)

        # 3. Hybrid retrieval across Layer 3 episodes & Layer 4 historical turns/exact IDs
        bundle = self.retriever.retrieve_bundle(
            customer_id=customer_id,
            thread_id=thread_id,
            query=current_message,
            recent_messages=messages,
            top_k=self.budget.retrieval_top_k,
            recent_window=self.budget.recent_messages_to_preserve,
            min_importance=self.budget.episode_importance_threshold,
        )

        seen_texts: set[str] = set()
        ep_lines: list[str] = []
        for ep in bundle.episodes[: self.budget.max_episodes_in_context]:
            norm = ep.content.strip().lower()
            if norm not in seen_texts:
                seen_texts.add(norm)
                ep_lines.append(f"- [{ep.topic} | imp={ep.importance:.2f}] {ep.content}")

        ret_lines: list[str] = []
        for item in bundle.relevant_messages[: self.budget.retrieval_top_k]:
            norm = item.content.strip().lower()
            if norm not in seen_texts:
                seen_texts.add(norm)
                ret_lines.append(f"- [{item.source} | score={item.score:.2f}] {item.content}")

        preserved_recent = list(bundle.recent_messages)
        truncated_items_count = 0

        def _render() -> tuple[str, str, str, int]:
            ep_txt = wrap_untrusted_data("episodic_memory", "\n".join(ep_lines)) if ep_lines else ""
            ret_txt = (
                wrap_untrusted_data("retrieved_history", "\n".join(ret_lines)) if ret_lines else ""
            )
            recent_txt = "\n".join(
                f"{m.get('role', 'user')}: {m.get('content', '')}" for m in preserved_recent
            )
            total = (
                estimate_tokens(structured_state_text)
                + estimate_tokens(compaction_summary_text)
                + estimate_tokens(ep_txt)
                + estimate_tokens(ret_txt)
                + estimate_messages_tokens(preserved_recent)
            )
            return ep_txt, ret_txt, recent_txt, total

        episodes_text, retrieved_history_text, recent_messages_text, total_tokens = _render()

        # Priority pruning: preserve Layer 2 state & latest turns; prune lowest-priority history first
        while total_tokens > effective_budget and ret_lines:
            ret_lines.pop()
            truncated_items_count += 1
            episodes_text, retrieved_history_text, recent_messages_text, total_tokens = _render()

        while total_tokens > effective_budget and ep_lines:
            ep_lines.pop()
            truncated_items_count += 1
            episodes_text, retrieved_history_text, recent_messages_text, total_tokens = _render()

        if total_tokens > effective_budget and compaction_summary_text:
            compaction_summary_text = ""
            truncated_items_count += 1
            episodes_text, retrieved_history_text, recent_messages_text, total_tokens = _render()

        while total_tokens > effective_budget and len(preserved_recent) > 1:
            preserved_recent.pop(0)
            truncated_items_count += 1
            episodes_text, retrieved_history_text, recent_messages_text, total_tokens = _render()

        return BuiltContext(
            structured_state_text=structured_state_text,
            episodes_text=episodes_text,
            retrieved_history_text=retrieved_history_text,
            recent_messages_text=recent_messages_text,
            estimated_tokens=total_tokens,
            truncated_items_count=truncated_items_count,
            compaction_summary_text=compaction_summary_text,
            recent_messages=preserved_recent,
            token_budget=effective_budget,
        )


def build_active_context(
    *,
    repo: SupportRepository,
    customer_id: str,
    thread_id: str,
    recent_messages: list[dict[str, str]],
    episodes: list[EpisodeRecord],
    retrieved_items: list[RetrievedMemoryItem],
    budget: ContextBudgetConfig,
) -> AssembledContext:
    """Assemble prompt context respecting strict max_memory_tokens priority pruning."""
    del episodes, retrieved_items  # Builder resolves deduplicated ranked bundle deterministically
    query = recent_messages[-1]["content"] if recent_messages else ""
    built = ActiveContextBuilder(repo, budget).build(
        customer_id=customer_id,
        thread_id=thread_id,
        current_message=query,
        messages=recent_messages,
        token_budget=budget.max_memory_tokens,
    )
    trimmed = ["pruned_items"] if built.truncated_items_count > 0 else []
    return AssembledContext(
        verified_state_block=built.structured_state_text,
        compaction_summary_block=built.compaction_summary_text,
        episodes_block=built.episodes_text,
        retrieved_history_block=built.retrieved_history_text,
        recent_messages=built.recent_messages,
        estimated_tokens=built.estimated_tokens,
        token_budget=built.token_budget,
        trimmed_sections=trimmed,
        truncated_items_count=built.truncated_items_count,
    )
