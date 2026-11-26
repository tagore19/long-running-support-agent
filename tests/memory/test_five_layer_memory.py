"""Tests for 5-layer memory, hybrid retrieval, token budget, and adaptive compaction."""

from __future__ import annotations

from pathlib import Path

from src.config import BusinessRulesConfig, MemoryBudgetConfig
from src.database.repositories import SupportRepository
from src.memory.compaction import maybe_compact_memory
from src.memory.context_builder import build_active_context
from src.memory.episode_extractor import extract_and_store_episodes
from src.memory.retrieval import HybridMemoryRetriever


def test_episode_extraction_and_hybrid_retrieval(seeded_db: tuple[Path, SupportRepository]) -> None:
    _, repo = seeded_db
    thread_id = "mem-thread-1"
    extract_and_store_episodes(
        repo=repo,
        customer_id="CUST-1001",
        thread_id=thread_id,
        user_message="Please note I prefer email updates only, never call me.",
        assistant_message="Noted! I have saved your email preference.",
    )
    repo.append_turn(
        thread_id=thread_id,
        customer_id="CUST-1001",
        turn_index=1,
        role="user",
        content="Checking order ORD-1001 coffee maker.",
        entities=["ORD-1001"],
    )
    repo.append_turn(
        thread_id=thread_id,
        customer_id="CUST-1001",
        turn_index=2,
        role="user",
        content="What is your weekend delivery schedule?",
        entities=[],
    )

    retriever = HybridMemoryRetriever(repo)
    hits = retriever.retrieve(
        customer_id="CUST-1001",
        thread_id=thread_id,
        query="What happened to ORD-1001?",
        top_k=3,
    )
    assert len(hits) >= 1
    assert any("ORD-1001" in h.entities for h in hits)
    assert hits[0].exact_entity_match is True


def test_adaptive_token_budget_compaction_and_context_bounds(
    seeded_db: tuple[Path, SupportRepository], rules: BusinessRulesConfig
) -> None:
    _, repo = seeded_db
    thread_id = "mem-thread-compact"
    tight_budget = MemoryBudgetConfig(
        max_memory_tokens=600,
        compaction_trigger_tokens=200,
        recent_messages_to_preserve=4,
        retrieval_top_k=4,
        max_episodes_in_context=4,
        episode_importance_threshold=0.4,
    )

    messages: list[dict[str, str]] = []
    for i in range(1, 13):
        u_msg = f"Turn {i}: I prefer email updates and am asking about order ORD-1001 details and delivery status in Sao Paulo."
        a_msg = f"Turn {i}: Order ORD-1001 was delivered and we noted your email preference for updates."
        messages.append({"role": "user", "content": u_msg})
        messages.append({"role": "assistant", "content": a_msg})
        repo.append_turn(
            thread_id=thread_id,
            customer_id="CUST-1001",
            turn_index=i,
            role="user",
            content=u_msg,
            entities=["ORD-1001"],
        )
    extract_and_store_episodes(
        repo=repo,
        customer_id="CUST-1001",
        thread_id=thread_id,
        user_message="I prefer email updates.",
        assistant_message="Preference recorded.",
    )

    preserved, snapshot = maybe_compact_memory(
        repo=repo,
        customer_id="CUST-1001",
        thread_id=thread_id,
        messages=messages,
        budget=tight_budget,
    )
    assert snapshot is not None
    assert len(preserved) == tight_budget.recent_messages_to_preserve
    assert snapshot.token_estimate_after < snapshot.token_estimate_before
    assert "ORD-1001" in snapshot.summary["entities"]

    assembled = build_active_context(
        repo=repo,
        customer_id="CUST-1001",
        thread_id=thread_id,
        recent_messages=preserved,
        episodes=repo.list_episodes(customer_id="CUST-1001", thread_id=thread_id),
        retrieved_items=[],
        budget=tight_budget,
    )
    assert assembled.estimated_tokens <= tight_budget.max_memory_tokens
