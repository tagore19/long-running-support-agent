"""Stage 4 tests for adaptive token-budget compaction, DB cross-verification, and 35-turn recall."""

from __future__ import annotations

from pathlib import Path

import pytest
from src.agent.graph import SupportAgentRunner
from src.config import AppConfig, ContextBudgetConfig, Settings
from src.database.connection import get_connection
from src.database.repositories import SupportRepository
from src.memory.compaction import TokenBudgetCompactor
from src.memory.memory_schemas import CompactedMemoryPayload


def test_token_budget_trigger_prefers_token_volume_over_turn_count(
    repo: SupportRepository,
) -> None:
    cfg = ContextBudgetConfig(
        compaction_trigger_tokens=300,
        recent_messages_to_preserve=4,
    )
    compactor = TokenBudgetCompactor(repo=repo)

    # 8 very short messages stay below 300 tokens -> no compaction
    short_msgs = [{"role": "user", "content": f"hi {i}"} for i in range(8)]
    assert compactor.should_compact(short_msgs, cfg) is False

    # 6 verbose messages exceed 300 tokens -> triggers compaction despite fewer turns
    verbose_msgs = [
        {"role": "user", "content": "Detailed complaint about shipping delay " * 10}
        for _ in range(6)
    ]
    assert compactor.should_compact(verbose_msgs, cfg) is True


def test_compaction_strips_fabricated_refund_claims_and_preserves_history(
    seeded_db: tuple[Path, SupportRepository],
) -> None:
    db_path, repo = seeded_db
    thread_id = "cmp-verify-1"
    cfg = ContextBudgetConfig(
        compaction_trigger_tokens=100,
        recent_messages_to_preserve=2,
    )
    compactor = TokenBudgetCompactor(repo=repo)

    # Seed an authentic episode and an unverified/fabricated refund claim in messages
    ep = repo.add_episode(
        customer_id="CUST-1001",
        thread_id=thread_id,
        content="Customer prefers email updates only.",
        entities=[],
        topic="preference",
        importance=0.92,
    )
    evt = repo.append_event(
        customer_id="CUST-1001",
        thread_id=thread_id,
        event_type="order_looked_up",
        order_id="ORD-1001",
        actor="test",
    )

    messages = [
        {
            "role": "user",
            "content": "Please note I prefer email updates only for all future order notifications and escalations.",
        },
        {
            "role": "assistant",
            "content": "Fabricated claim: Refund REF-FAKE9999 was credited for ORD-1003 in full! "
            + ("additional unverified filler text " * 5),
        },
        {
            "role": "user",
            "content": "Checking detailed shipping schedules for Sao Paulo deliveries and carrier handoff windows.",
        },
        {
            "role": "assistant",
            "content": "Standard deliveries via Correios SEDEX take 3-5 business days across Sao Paulo.",
        },
        {"role": "user", "content": "Latest question?"},
        {"role": "assistant", "content": "Latest answer."},
    ]

    with get_connection(db_path) as conn:
        outcome = compactor.compact(
            customer_id="CUST-1001",
            thread_id=thread_id,
            messages=messages,
            db_conn=conn,
            context_config=cfg,
        )

    assert outcome.compacted is True
    assert isinstance(outcome.payload, CompactedMemoryPayload)
    assert len(outcome.preserved_messages) == cfg.recent_messages_to_preserve
    assert outcome.tokens_after < outcome.tokens_before
    # Fabricated REF-FAKE9999 refund claim must be stripped since Layer 2 refunds has 0 rows
    assert any("REF-FAKE9999" in s for s in outcome.payload.stripped_unverified_claims)
    assert all("REF-FAKE9999" not in p.get("promise", "") for p in outcome.payload.promises_made)
    assert outcome.payload.source_thread_id == thread_id

    # Historical rows in events & episodes are preserved non-destructively
    assert any(e.event_id == evt.event_id for e in repo.list_events(thread_id=thread_id))
    assert any(e.episode_id == ep.episode_id for e in repo.list_episodes(thread_id=thread_id))


@pytest.mark.asyncio
async def test_35_turn_conversation_multiple_compactions_and_late_recall(
    test_settings: Settings, rules: AppConfig, seeded_env: dict[str, Path]
) -> None:
    tight_rules = rules.model_copy(
        update={
            "context": rules.context.model_copy(
                update={
                    "compaction_trigger_tokens": 450,
                    "recent_messages_to_preserve": 6,
                    "max_memory_tokens": 2500,
                }
            )
        }
    )

    turns = [
        "Hi! Please note I prefer email updates only and never phone calls.",
        "Can you check my order ORD-1001?",
        "The coffee maker in ORD-1001 is leaking. Please issue a full refund of 189.90 BRL.",
        "Thanks! What is the refund status of ORD-1001 now?",
    ]
    # Add 27 noisy complaint/shipping/filler turns (turns 5..31)
    for i in range(5, 32):
        turns.append(
            f"Complaint #{i}: shipping packaging was dented and the mobile app logged me out while checking delivery rules in Sao Paulo."
        )
    # Late-turn recall & duplicate prevention (turns 32..35)
    turns.extend(
        [
            "Remind me, what contact method did I say I prefer at the start?",
            "What happened earlier with ORD-1001 and how much was refunded?",
            "Can you refund ORD-1001 again just in case?",
            "Please summarize my current orders and refund status.",
        ]
    )
    assert len(turns) == 35

    compaction_events = 0
    responses: list[dict] = []
    async with SupportAgentRunner(
        settings=test_settings,
        rules=tight_rules,
        app_db_path=seeded_env["app_db"],
        checkpoint_db_path=seeded_env["ckpt_db"],
        trace_log_path=seeded_env["trace_log"],
        mode="improved",
    ) as runner:
        for msg in turns:
            res = await runner.arun_turn(
                thread_id="thread-35-turns",
                customer_id="CUST-1001",
                user_message=msg,
            )
            responses.append(res)
            if res.get("compaction_triggered"):
                compaction_events += 1

        # Multiple compaction cycles must have fired across 35 verbose turns
        assert compaction_events >= 2
        last_state = responses[-1]
        assert last_state["compaction_count"] >= 2
        assert last_state["last_compaction_tokens_after"] < last_state["last_compaction_tokens_before"]
        for r in responses:
            if r.get("compaction_triggered"):
                assert len(r["messages"]) == tight_rules.context.recent_messages_to_preserve
        assert len(last_state["messages"]) < 20

        # Turn 32: preference recall
        assert "email" in responses[31]["response"].lower()
        # Turn 33: early refund recall
        assert "ORD-1001" in responses[32]["response"]
        assert "REF-" in responses[32]["response"]
        # Turn 34: duplicate refund blocked -> only 1 refund exists in SQLite
        assert len(runner.repo.get_refunds_for_order("ORD-1001")) == 1
        # Turn 35: verified summary includes ORD-1001 and REF-
        assert "ORD-1001" in responses[34]["response"]
        assert "REF-" in responses[34]["response"]
