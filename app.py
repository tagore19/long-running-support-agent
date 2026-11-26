"""Streamlit Operator & Customer Support Console with 5-Layer Memory & Trace Inspector."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Mapping, MutableMapping
from pathlib import Path
from typing import Any

import streamlit as st
from scripts.seed_data import seed_databases
from src.agent.graph import SupportAgentRunner
from src.config import get_settings, load_app_config
from src.database.repositories import SupportRepository
from src.observability.metrics import aggregate_trace_metrics, load_traces_from_jsonl
from streamlit.errors import StreamlitSecretNotFoundError

st.set_page_config(page_title="Long-Running Customer Support Agent", layout="wide")


def _apply_runtime_secrets(
    secrets: Mapping[str, Any], environ: MutableMapping[str, str]
) -> None:
    """Expose Streamlit-hosted Gemini credentials through the app's env-based settings."""
    api_key = secrets.get("GEMINI_API_KEY")
    if not api_key:
        return
    environ.setdefault("GEMINI_API_KEY", str(api_key))
    environ.setdefault("LLM_PROVIDER", str(secrets.get("LLM_PROVIDER") or "gemini"))
    environ.setdefault("LLM_MODE", str(secrets.get("LLM_MODE") or "live"))


def _load_streamlit_secrets() -> None:
    try:
        _apply_runtime_secrets(st.secrets, os.environ)
    except StreamlitSecretNotFoundError:
        return


def _init_repo() -> SupportRepository:
    settings = get_settings()
    app_db = settings.resolve_path(settings.effective_db_path)
    repo = SupportRepository(app_db)
    if not repo.list_customers():
        seed_databases(
            seed_db_path=settings.resolve_path(settings.seed_db_path),
            app_db_path=app_db,
            reset=False,
        )
    return repo


async def _run_turn_async(
    *,
    thread_id: str,
    customer_id: str,
    message: str,
    mode: str,
    document_text: str | None = None,
    image_text: str | None = None,
) -> dict[str, Any]:
    async with SupportAgentRunner(mode=mode) as runner:
        return await runner.arun_turn(
            thread_id=thread_id,
            customer_id=customer_id,
            user_message=message,
            mode=mode,
            document_text=document_text or None,
            image_text=image_text or None,
        )


async def _run_stress_test_async(thread_id: str, customer_id: str, mode: str) -> list[dict[str, str]]:
    fixtures_file = Path(__file__).resolve().parent / "data" / "fixtures" / "long_conversations.json"
    with fixtures_file.open("r", encoding="utf-8") as fh:
        convo = json.load(fh)[0]

    settings = get_settings()
    rules = load_app_config(settings.business_rules_path)
    # Use tighter trigger in UI stress preset so compaction fires live during the 32 turns
    rules = rules.model_copy(
        update={
            "context": rules.context.model_copy(
                update={"compaction_trigger_tokens": 900, "recent_messages_to_preserve": 6}
            )
        }
    )

    entries: list[dict[str, str]] = []
    async with SupportAgentRunner(rules=rules, mode=mode) as runner:
        for turn in convo["turns"]:
            res = await runner.arun_turn(
                thread_id=thread_id,
                customer_id=customer_id,
                user_message=turn["message"],
                mode=mode,
            )
            final = res.get("final_response") or {}
            ctx = res.get("assembled_context") or {}
            entries.append({"role": "user", "content": turn["message"]})
            entries.append(
                {
                    "role": "assistant",
                    "content": str(final.get("answer", "")),
                    "meta": (
                        f"tools={final.get('actions_taken', [])} | "
                        f"policy_blocked={final.get('policy_blocked', False)} | "
                        f"compacted={res.get('compaction_triggered', False)} | "
                        f"tokens={ctx.get('estimated_tokens', 0)}/{ctx.get('token_budget', 0)}"
                    ),
                }
            )
    st.session_state["last_context_summary"] = ctx
    return entries


def main() -> None:
    _load_streamlit_secrets()
    st.title("Long-Running E-Commerce Customer Support Console")
    st.caption(
        "5-Layer Event-Driven Memory | FastMCP Stdio Tools | Deterministic Policy Gate | PII-Redacted JSONL Traces"
    )

    settings = get_settings()
    rules = load_app_config(settings.business_rules_path)
    repo = _init_repo()
    customers = repo.list_customers()
    customer_ids = [c.customer_id for c in customers]

    if "selected_customer" not in st.session_state:
        st.session_state["selected_customer"] = customer_ids[0]
    if "thread_id" not in st.session_state:
        st.session_state["thread_id"] = f"ui-{customer_ids[0].lower()}"
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = {}

    with st.sidebar:
        st.header("Session Controls")
        selected_customer = st.selectbox(
            "Customer ID",
            customer_ids,
            index=customer_ids.index(st.session_state["selected_customer"]),
        )
        st.session_state["selected_customer"] = selected_customer
        thread_id = st.text_input("Thread ID", value=st.session_state["thread_id"])
        st.session_state["thread_id"] = thread_id
        mode = st.radio("Execution Mode", ["improved", "baseline"], index=0)

        st.divider()
        st.subheader("Quick Demo Presets")
        if st.button("Preset: Valid Refund (ORD-1001)"):
            st.session_state["selected_customer"] = "CUST-1001"
            st.session_state["thread_id"] = "ui-preset-valid"
            st.session_state["pending_preset_msg"] = (
                "My coffee maker in ORD-1001 is leaking. Please issue a refund of 189.90 BRL."
            )
            st.rerun()
        if st.button("Preset: Expired Window (ORD-1003)"):
            st.session_state["selected_customer"] = "CUST-1001"
            st.session_state["thread_id"] = "ui-preset-expired"
            st.session_state["pending_preset_msg"] = "Please issue a refund for order ORD-1003."
            st.rerun()
        if st.button("Preset: Over-Limit >500 BRL (ORD-1010)"):
            st.session_state["selected_customer"] = "CUST-1004"
            st.session_state["thread_id"] = "ui-preset-overlimit"
            st.session_state["pending_preset_msg"] = (
                "The gaming chair in ORD-1010 is damaged. Please refund 780.00 BRL."
            )
            st.rerun()
        if st.button("Run 32-Turn Stress Test"):
            stress_thread = "ui-stress-32"
            st.session_state["selected_customer"] = "CUST-1001"
            st.session_state["thread_id"] = stress_thread
            entries = asyncio.run(_run_stress_test_async(stress_thread, "CUST-1001", mode))
            st.session_state["chat_history"][f"{stress_thread}:{mode}"] = entries
            st.rerun()

        st.divider()
        st.subheader("Active Policy Rules")
        st.json(rules.model_dump())
        if st.button("Reset & Re-Seed Demo DB"):
            seed_databases(
                seed_db_path=settings.resolve_path(settings.seed_db_path),
                app_db_path=settings.resolve_path(settings.effective_db_path),
                reset=True,
            )
            st.session_state["chat_history"] = {}
            st.session_state.pop("last_context_summary", None)
            st.success("Databases re-seeded.")
            st.rerun()

    history_key = f"{thread_id}:{mode}"
    history = st.session_state["chat_history"].setdefault(history_key, [])

    left_col, right_col = st.columns([1.15, 1.0])

    with left_col:
        st.subheader("Support Chat")
        with st.expander("Untrusted Attachment Data (Document / Image OCR — Never Trusted as Instructions)"):
            doc_text = st.text_area("Untrusted Document / Uploaded Text", key="ui_doc_text", height=70)
            img_text = st.text_area("Untrusted Image OCR / Caption Text", key="ui_img_text", height=70)

        for msg in history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                if msg.get("meta"):
                    st.caption(msg["meta"])

        preset_msg = st.session_state.pop("pending_preset_msg", None)
        prompt = st.chat_input("Ask about an order, refund, or escalation...") or preset_msg
        if prompt:
            history.append({"role": "user", "content": prompt})
            result = asyncio.run(
                _run_turn_async(
                    thread_id=thread_id,
                    customer_id=selected_customer,
                    message=prompt,
                    mode=mode,
                    document_text=doc_text,
                    image_text=img_text,
                )
            )
            final = result.get("final_response") or {}
            ctx = result.get("assembled_context") or {}
            st.session_state["last_context_summary"] = ctx
            st.session_state["last_compaction_stats"] = {
                "compaction_count": result.get("compaction_count", 0),
                "last_compaction_tokens_before": result.get("last_compaction_tokens_before", 0),
                "last_compaction_tokens_after": result.get("last_compaction_tokens_after", 0),
            }
            meta = (
                f"trace={result.get('trace_id')} | tools={final.get('actions_taken', [])} | "
                f"policy_blocked={final.get('policy_blocked', False)} | "
                f"escalated={final.get('escalated', False)} | "
                f"injection_suspected={result.get('injection_suspected', False)} | "
                f"compacted={result.get('compaction_triggered', False)} | "
                f"tokens={ctx.get('estimated_tokens', 0)}/{ctx.get('token_budget', 0)}"
            )
            history.append(
                {
                    "role": "assistant",
                    "content": str(result.get("response") or final.get("answer") or ""),
                    "meta": meta,
                }
            )
            st.rerun()

    with right_col:
        st.subheader("5-Layer State & Memory Inspector")
        tab_l2, tab_l1, tab_l3, tab_obs = st.tabs(
            ["Layer 2: State", "Layer 1: Events", "Layer 3 & 5: Memory", "PII-Masked Traces"]
        )

        with tab_l2:
            st.markdown("**Orders**")
            st.dataframe(
                [
                    {
                        "order_id": o.order_id,
                        "status": o.status,
                        "amount": f"{o.amount:.2f} {o.currency}",
                        "refunded": f"{o.refunded_amount:.2f}",
                        "category": o.category,
                        "delivered_date": o.delivered_date,
                    }
                    for o in repo.list_orders_for_customer(selected_customer)
                ],
                use_container_width=True,
            )
            st.markdown("**Refunds**")
            st.json([r.model_dump() for r in repo.get_refunds_for_customer(selected_customer)])
            st.markdown("**Support Tickets**")
            st.json(
                [t.model_dump() for t in repo.list_support_tickets_for_customer(selected_customer)]
            )
            st.markdown("**Idempotency Operations (`tool_operations`)**")
            st.json(
                [op.model_dump() for op in repo.list_tool_operations(customer_id=selected_customer)]
            )

        with tab_l1:
            events = repo.list_events(
                customer_id=selected_customer, thread_id=thread_id, limit=100
            )
            st.json([e.model_dump() for e in events])

        with tab_l3:
            st.markdown("**Compaction Stats**")
            st.json(st.session_state.get("last_compaction_stats", {"compaction_count": 0}))
            st.markdown("**Episodic Memories**")
            episodes = repo.list_episodes(
                customer_id=selected_customer, thread_id=thread_id, limit=50
            )
            st.json([ep.model_dump() for ep in episodes])
            snapshot = repo.get_latest_compaction_snapshot(thread_id)
            st.markdown("**Latest Compaction Snapshot**")
            st.json(snapshot.model_dump() if snapshot else {"status": "none yet"})
            st.markdown("**Last Built Context Summary (Layer 5)**")
            st.json(st.session_state.get("last_context_summary", {}))

        with tab_obs:
            traces = load_traces_from_jsonl(
                settings.resolve_path(settings.trace_log_path), limit=100
            )
            metrics = aggregate_trace_metrics(traces)
            st.markdown("**Aggregated Metrics**")
            st.json(metrics.model_dump())
            st.markdown("**Recent PII-Redacted JSONL Spans**")
            st.json(traces[-15:])


if __name__ == "__main__":
    main()
