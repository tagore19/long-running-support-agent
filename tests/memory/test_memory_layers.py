"""Tests for 5-layer memory retrieval, context budgeting, and adaptive compaction."""

from __future__ import annotations

from src.config import BusinessRulesConfig
from src.database.repositories import SupportRepository
from src.memory.compaction import maybe_compact_memory
from src.memory.context_builder import build_active_context
from src.memory.episode_extractor import extract_and_store_episodes
from src.memory.retrieval import HybridMemoryRetriever


def test_hybrid_retrieval_exact_entity_and_customer_isolation(repo: SupportRepository) -> None:
    repo.add_episode(
        customer_id="CUST-1001",
        thread_id="th-1",
        content="Customer prefers email updates only and reported leaking ORD-1001.",
        entities=["ORD-1001"],
        topic="preference",
        importance=0.92,
    )
    repo.add_episode(
        customer_id="CUST-1002",
        thread_id="th-2",
        content="Other customer secret complaint for ORD-1004.",
        entities=["ORD-1004"],
        topic="issue",
        importance=0.95,
    )

    retriever = HybridMemoryRetriever(repo)
    hits = retriever.retrieve(
        customer_id="CUST-1001",
        thread_id="th-1",
        query="What happened with ORD-1001 and my email preference?",
        top_k=5,
    )
    assert len(hits) >= 1
    assert hits[0].exact_entity_match is True
    assert all("CUST-1002" not in h.content and "ORD-1004" not in h.content for h in hits)


def test_adaptive_token_compaction_and_layer2_crosscheck(
    repo: SupportRepository, rules: BusinessRulesConfig
) -> None:
    budget = rules.memory_budget.model_copy(
        update={"compaction_trigger_tokens": 120, "recent_messages_to_preserve": 4}
    )
    extract_and_store_episodes(
        repo=repo,
        customer_id="CUST-1001",
        thread_id="th-compact",
        user_message="Please note I prefer email updates for ORD-1001.",
        assistant_message="Noted! I checked ORD-1001.",
    )
    messages = []
    for i in range(10):
        messages.append({"role": "user", "content": f"Message {i} asking detailed questions about ORD-1001 and delivery schedules in Sao Paulo."})
        messages.append({"role": "assistant", "content": f"Detailed reply {i} confirming we are tracking ORD-1001 via Correios SEDEX."})

    preserved, snapshot = maybe_compact_memory(
        repo=repo,
        customer_id="CUST-1001",
        thread_id="th-compact",
        messages=messages,
        budget=budget,
    )
    assert snapshot is not None
    assert len(preserved) == 4
    assert snapshot.token_estimate_after < snapshot.token_estimate_before
    assert "ORD-1001" in snapshot.summary["entities"]


def test_active_context_builder_strictly_respects_token_budget(
    repo: SupportRepository, rules: BusinessRulesConfig
) -> None:
    tight_budget = rules.memory_budget.model_copy(update={"max_memory_tokens": 350})
    long_msgs = [
        {"role": "user", "content": "Long noisy message " + ("details " * 40)},
        {"role": "assistant", "content": "Long assistant response " + ("context " * 40)},
        {"role": "user", "content": "Check ORD-1001 please."},
    ]
    ctx = build_active_context(
        repo=repo,
        customer_id="CUST-1001",
        thread_id="th-budget",
        recent_messages=long_msgs,
        episodes=[],
        retrieved_items=[],
        budget=tight_budget,
    )
    assert ctx.estimated_tokens <= 350
