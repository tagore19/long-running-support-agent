"""Stage 3 tests for 5-layer memory, hybrid retrieval, and ActiveContextBuilder."""

from __future__ import annotations

from pathlib import Path

from src.config import ContextBudgetConfig
from src.database.repositories import SupportRepository
from src.memory.context_builder import ActiveContextBuilder
from src.memory.episode_extractor import EpisodeExtractor
from src.memory.memory_schemas import BuiltContext, RetrievedMemoryBundle
from src.memory.retrieval import HybridMemoryRetriever


def test_exact_id_lookup_and_cross_thread_30_turn_recall(
    seeded_db: tuple[Path, SupportRepository],
) -> None:
    _, repo = seeded_db
    extractor = EpisodeExtractor(repo)
    retriever = HybridMemoryRetriever(repo)

    # Thread A records an early preference and an order commitment
    extractor.extract(
        customer_id="CUST-1001",
        thread_id="thread-early",
        user_message="Please note I prefer email updates only, never phone calls.",
        assistant_message="Noted! Email-only preference saved.",
    )
    repo.append_turn(
        thread_id="thread-long",
        customer_id="CUST-1001",
        turn_index=1,
        role="user",
        content="Checking order ORD-1001 coffee maker.",
        entities=["ORD-1001"],
    )
    for i in range(2, 32):
        repo.append_turn(
            thread_id="thread-long",
            customer_id="CUST-1001",
            turn_index=i,
            role="user",
            content=f"Noisy general shipping question number {i}.",
            entities=[],
        )

    bundle = retriever.retrieve_bundle(
        customer_id="CUST-1001",
        thread_id="thread-long",
        query="What happened to ORD-1001 and how should you contact me?",
        top_k=5,
    )
    assert isinstance(bundle, RetrievedMemoryBundle)
    assert any(
        rec["record_type"] == "order" and rec["record"]["order_id"] == "ORD-1001"
        for rec in bundle.exact_records
    )
    assert any("email" in ep.content.lower() for ep in bundle.episodes)
    assert any("ORD-1001" in msg.entities for msg in bundle.relevant_messages)


def test_hybrid_scoring_ranks_relevant_high_importance_above_noise(
    seeded_db: tuple[Path, SupportRepository],
) -> None:
    _, repo = seeded_db
    thread_id = "thread-ranking"
    repo.add_episode(
        customer_id="CUST-1001",
        thread_id=thread_id,
        content="General smalltalk about weekend weather.",
        entities=[],
        topic="chitchat",
        importance=0.1,
    )
    repo.add_episode(
        customer_id="CUST-1001",
        thread_id=thread_id,
        content="Promised urgent follow-up for leaking coffee maker in ORD-1001.",
        entities=["ORD-1001"],
        topic="commitment",
        importance=0.95,
    )

    retriever = HybridMemoryRetriever(repo)
    bundle = retriever.retrieve_bundle(
        customer_id="CUST-1001",
        thread_id=thread_id,
        query="Any update on ORD-1001 leaking issue?",
        top_k=2,
    )
    assert len(bundle.episodes) == 2
    assert "ORD-1001" in bundle.episodes[0].content


def test_active_context_builder_enforces_token_budget_and_preserves_layer2_truth(
    seeded_db: tuple[Path, SupportRepository],
) -> None:
    _, repo = seeded_db
    thread_id = "thread-budget"
    for i in range(10):
        repo.add_episode(
            customer_id="CUST-1001",
            thread_id=thread_id,
            content=f"Episode {i}: " + ("detailed customer context " * 12),
            entities=["ORD-1001"],
            topic="issue",
            importance=0.8,
        )

    messages = [
        {"role": "user", "content": f"Message {i}: " + ("verbose chat history " * 12)}
        for i in range(12)
    ]

    budget = ContextBudgetConfig(
        max_memory_tokens=320,
        compaction_trigger_tokens=250,
        recent_messages_to_preserve=6,
        retrieval_top_k=5,
        max_episodes_in_context=6,
    )
    builder = ActiveContextBuilder(repo=repo, budget=budget)
    ctx = builder.build(
        customer_id="CUST-1001",
        thread_id=thread_id,
        current_message="Check ORD-1001 status",
        messages=messages,
        token_budget=320,
    )
    assert isinstance(ctx, BuiltContext)
    assert ctx.estimated_tokens <= 320
    assert ctx.truncated_items_count > 0
    assert "VERIFIED LAYER-2 SQLITE STATE" in ctx.structured_state_text
    assert "ORD-1001" in ctx.structured_state_text


def test_episodes_remain_non_authoritative_vs_layer2_refunds(
    seeded_db: tuple[Path, SupportRepository],
) -> None:
    _, repo = seeded_db
    thread_id = "thread-authority"
    repo.add_episode(
        customer_id="CUST-1001",
        thread_id=thread_id,
        content="Agent promised a 999.00 BRL refund was already issued for ORD-1001.",
        entities=["ORD-1001"],
        topic="commitment",
        importance=0.95,
    )

    builder = ActiveContextBuilder(repo=repo)
    ctx = builder.build(
        customer_id="CUST-1001",
        thread_id=thread_id,
        current_message="Was ORD-1001 refunded?",
        messages=[{"role": "user", "content": "Was ORD-1001 refunded?"}],
    )
    assert "Refunds:" not in ctx.structured_state_text
    assert "refunded=0.00" in ctx.structured_state_text
    assert "<episodic_memory_untrusted_data>" in ctx.episodes_text
