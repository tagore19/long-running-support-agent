"""LangGraph node implementations enforcing guardrails, MCP stdio execution, and 5-layer memory."""

from __future__ import annotations

import hashlib
import uuid
from typing import Any

from src.agent.state import AgentState, ToolCallRequest
from src.config import AppConfig, Settings
from src.database.repositories import SupportRepository
from src.llm import FinalAgentResponse, LLMClient
from src.mcp_server.client import MCPToolClient
from src.memory.compaction import TokenBudgetCompactor
from src.memory.context_builder import ActiveContextBuilder
from src.memory.episode_extractor import EpisodeExtractor
from src.memory.memory_schemas import (
    estimate_messages_tokens,
    estimate_tokens,
    extract_entities,
)
from src.memory.retrieval import HybridMemoryRetriever
from src.observability.metrics import estimate_cost_usd
from src.observability.tracing import TraceLogger
from src.policies.authorization import (
    detect_prompt_injection,
    wrap_untrusted_document_data,
    wrap_untrusted_image_data,
    wrap_untrusted_memory_data,
    wrap_untrusted_order_data,
    wrap_untrusted_user_input,
)
from src.policies.refund_policy import evaluate_refund_eligibility


class SupportGraphNodes:
    """Container binding runtime dependencies to LangGraph node functions."""

    def __init__(
        self,
        *,
        settings: Settings,
        rules: AppConfig,
        repo: SupportRepository,
        llm: LLMClient,
        mcp_client: MCPToolClient,
        tracer: TraceLogger,
    ) -> None:
        self.settings = settings
        self.rules = rules
        self.repo = repo
        self.llm = llm
        self.mcp_client = mcp_client
        self.tracer = tracer
        self.retriever = HybridMemoryRetriever(repo)
        self.episode_extractor = EpisodeExtractor(repo)
        self.context_builder = ActiveContextBuilder(repo=repo, budget=rules.context)
        self.compactor = TokenBudgetCompactor(repo=repo)

    def validate_input(self, state: AgentState) -> dict[str, Any]:
        t0 = self.tracer.now_ms()
        user_msg = (state.get("user_message") or "").strip()
        customer_id = (state.get("customer_id") or "").strip()
        thread_id = (state.get("thread_id") or "").strip()
        trace_id = f"tr-{uuid.uuid4().hex[:10]}"
        mode = state.get("mode") or self.settings.agent_mode

        if not user_msg:
            raise ValueError("user_message must not be empty.")
        if len(user_msg) > self.rules.guardrails.max_input_chars:
            raise ValueError(
                f"user_message exceeds max_input_chars ({self.rules.guardrails.max_input_chars})."
            )
        if not customer_id:
            raise ValueError("customer_id must not be empty.")
        if not thread_id:
            raise ValueError("thread_id must not be empty.")

        doc_text = (state.get("document_text") or "").strip()
        img_text = (state.get("image_text") or "").strip()
        flags = list(
            dict.fromkeys(
                detect_prompt_injection(user_msg)
                + detect_prompt_injection(doc_text)
                + detect_prompt_injection(img_text)
            )
        )
        wrapped_user = wrap_untrusted_user_input(user_msg)
        wrapped_doc = wrap_untrusted_document_data(doc_text) if doc_text else ""
        wrapped_img = wrap_untrusted_image_data(img_text) if img_text else ""

        existing_messages = list(state.get("messages") or [])
        turn_index = (len(existing_messages) // 2) + 1
        existing_messages.append({"role": "user", "content": user_msg})

        self.tracer.emit(
            trace_id=trace_id,
            thread_id=thread_id,
            step_name="validate_input",
            latency_ms=self.tracer.now_ms() - t0,
            status="flagged_injection" if flags else "ok",
            mode=mode,
        )
        return {
            "trace_id": trace_id,
            "mode": mode,
            "turn_index": turn_index,
            "wrapped_user_message": wrapped_user,
            "wrapped_document_text": wrapped_doc,
            "wrapped_image_text": wrapped_img,
            "injection_flags": flags,
            "injection_suspected": bool(flags),
            "messages": existing_messages,
            "step_count": 0,
            "tool_calls_count": 0,
            "tool_history": [],
            "pending_tool_call": None,
            "tool_blocked": False,
            "tool_block_reason": "",
            "last_tool_result": None,
            "last_retry_count": 0,
            "final_response": None,
            "response": "",
            "compaction_triggered": False,
            "compaction_snapshot_id": None,
            "error": None,
        }

    def load_state(self, state: AgentState) -> dict[str, Any]:
        t0 = self.tracer.now_ms()
        customer_id = state["customer_id"]
        thread_id = state["thread_id"]
        user_msg = state["user_message"]
        turn_index = state["turn_index"]

        self.repo.append_turn(
            thread_id=thread_id,
            customer_id=customer_id,
            turn_index=turn_index,
            role="user",
            content=user_msg,
            entities=extract_entities(user_msg),
        )

        orders = [o.model_dump() for o in self.repo.list_orders_for_customer(customer_id)]
        # Flag any injection embedded in untrusted order_note fields
        note_flags: list[str] = []
        for o in orders:
            note_flags.extend(detect_prompt_injection(str(o.get("order_note", ""))))
        combined_flags = list(dict.fromkeys(list(state.get("injection_flags") or []) + note_flags))
        refunds = [r.model_dump() for r in self.repo.get_refunds_for_customer(customer_id)]
        tickets = [
            t.model_dump() for t in self.repo.list_support_tickets_for_customer(customer_id)
        ]

        self.tracer.emit(
            trace_id=state["trace_id"],
            thread_id=thread_id,
            step_name="load_state",
            latency_ms=self.tracer.now_ms() - t0,
            status="flagged_order_note_injection" if note_flags else "ok",
            mode=state["mode"],
        )
        return {
            "verified_orders": orders,
            "verified_refunds": refunds,
            "verified_tickets": tickets,
            "injection_flags": combined_flags,
            "injection_suspected": bool(combined_flags),
        }

    def retrieve_memory(self, state: AgentState) -> dict[str, Any]:
        t0 = self.tracer.now_ms()
        mode = state["mode"]
        customer_id = state["customer_id"]
        thread_id = state["thread_id"]

        if mode == "baseline":
            self.tracer.emit(
                trace_id=state["trace_id"],
                thread_id=thread_id,
                step_name="retrieve_memory",
                latency_ms=self.tracer.now_ms() - t0,
                status="skipped_baseline",
                mode=mode,
            )
            return {"episodes": [], "retrieved_items": []}

        budget = self.rules.context
        episodes = self.repo.list_episodes(
            customer_id=customer_id,
            thread_id=thread_id,
            min_importance=budget.episode_importance_threshold,
            limit=25,
        )
        retrieved = self.retriever.retrieve(
            customer_id=customer_id,
            thread_id=thread_id,
            query=state["user_message"],
            top_k=budget.retrieval_top_k,
            min_importance=0.0,
        )

        self.tracer.emit(
            trace_id=state["trace_id"],
            thread_id=thread_id,
            step_name="retrieve_memory",
            latency_ms=self.tracer.now_ms() - t0,
            mode=mode,
        )
        return {
            "episodes": [e.model_dump() for e in episodes],
            "retrieved_items": [r.model_dump() for r in retrieved],
        }

    def build_context(self, state: AgentState) -> dict[str, Any]:
        t0 = self.tracer.now_ms()
        mode = state["mode"]
        budget = self.rules.context

        if mode == "baseline":
            recent = list(state["messages"][-budget.recent_messages_to_preserve :])
            assembled = {
                "verified_state_block": "",
                "compaction_summary_block": "",
                "episodes_block": "",
                "retrieved_history_block": "",
                "recent_messages": recent,
                "estimated_tokens": estimate_messages_tokens(recent),
                "token_budget": budget.max_memory_tokens,
                "trimmed_sections": [],
            }
        else:
            ctx = self.context_builder.build(
                customer_id=state["customer_id"],
                thread_id=state["thread_id"],
                current_message=state["user_message"],
                messages=state["messages"],
                token_budget=budget.max_memory_tokens,
                compacted_memories=state.get("compacted_memories"),
            )
            assembled = ctx.model_dump()

        self.tracer.emit(
            trace_id=state["trace_id"],
            thread_id=state["thread_id"],
            step_name="build_context",
            latency_ms=self.tracer.now_ms() - t0,
            input_tokens=int(assembled["estimated_tokens"]),
            mode=mode,
        )
        return {"assembled_context": assembled}

    def agent(self, state: AgentState) -> dict[str, Any]:
        t0 = self.tracer.now_ms()
        step_count = state.get("step_count", 0) + 1
        guardrails = self.rules.guardrails

        if step_count > guardrails.max_steps:
            self.tracer.emit(
                trace_id=state["trace_id"],
                thread_id=state["thread_id"],
                step_name="agent",
                latency_ms=self.tracer.now_ms() - t0,
                status="max_steps_reached",
                mode=state["mode"],
            )
            answer = "I have reached the maximum reasoning step limit for this turn. Let me summarize what I have verified so far."
            return {
                "step_count": step_count,
                "pending_tool_call": None,
                "response": answer,
                "final_response": {
                    "answer": answer,
                    "referenced_order_ids": extract_entities(state["user_message"]),
                    "actions_taken": [t["tool_name"] for t in state.get("tool_history", [])],
                    "escalated": False,
                    "policy_blocked": False,
                },
            }

        ctx = state["assembled_context"]
        order_notes_wrapped = "\n".join(
            wrap_untrusted_order_data(f"{o['order_id']}: {o.get('order_note', '')}")
            for o in state.get("verified_orders", [])
            if o.get("order_note")
        )
        memory_raw = "\n".join(
            part
            for part in [
                ctx.get("compaction_summary_text") or ctx.get("compaction_summary_block", ""),
                ctx.get("episodes_text") or ctx.get("episodes_block", ""),
                ctx.get("retrieved_history_text") or ctx.get("retrieved_history_block", ""),
            ]
            if part
        )
        memory_wrapped = wrap_untrusted_memory_data(memory_raw) if memory_raw else ""
        context_prompt = "\n\n".join(
            part
            for part in [
                ctx.get("structured_state_text") or ctx.get("verified_state_block", ""),
                order_notes_wrapped,
                memory_wrapped,
                f"Recent Conversation:\n{ctx.get('recent_messages', [])}",
                state.get("wrapped_document_text", ""),
                state.get("wrapped_image_text", ""),
                f"Current User Input:\n{state['wrapped_user_message']}",
            ]
            if part
        )

        llm_out = self.llm.plan(
            customer_id=state["customer_id"],
            thread_id=state["thread_id"],
            user_message=state["user_message"],
            context_prompt=context_prompt,
            tool_history=state.get("tool_history", []),
            verified_orders=state.get("verified_orders", []),
            verified_refunds=state.get("verified_refunds", []),
            verified_tickets=state.get("verified_tickets", []),
            episodes=state.get("episodes", []),
            mode=state["mode"],
            recent_messages=ctx.get("recent_messages", []),
        )
        decision = llm_out.decision
        cost = estimate_cost_usd(
            llm_out.input_tokens,
            llm_out.output_tokens,
            self.settings.effective_llm_provider,
        )

        if (
            decision.action == "call_tool"
            and decision.tool_name
            and state.get("tool_calls_count", 0) < guardrails.max_tool_calls
        ):
            self.tracer.emit(
                trace_id=state["trace_id"],
                thread_id=state["thread_id"],
                step_name="agent",
                latency_ms=self.tracer.now_ms() - t0,
                input_tokens=llm_out.input_tokens,
                output_tokens=llm_out.output_tokens,
                estimated_cost_usd=cost,
                tool_name=decision.tool_name,
                mode=state["mode"],
            )
            return {
                "step_count": step_count,
                "pending_tool_call": {
                    "tool_name": decision.tool_name,
                    "arguments": decision.tool_args,
                },
            }

        answer_text = decision.response_text or "How else can I assist you with your order?"
        self.tracer.emit(
            trace_id=state["trace_id"],
            thread_id=state["thread_id"],
            step_name="agent",
            latency_ms=self.tracer.now_ms() - t0,
            input_tokens=llm_out.input_tokens,
            output_tokens=llm_out.output_tokens,
            estimated_cost_usd=cost,
            mode=state["mode"],
        )
        return {
            "step_count": step_count,
            "pending_tool_call": None,
            "response": answer_text,
            "final_response": {
                "answer": answer_text,
                "referenced_order_ids": [
                    e
                    for e in extract_entities(state["user_message"] + " " + answer_text)
                    if e.startswith("ORD-")
                ],
                "actions_taken": [t["tool_name"] for t in state.get("tool_history", [])],
                "escalated": any(
                    t["tool_name"] == "create_support_ticket" and t["result"].get("ok")
                    for t in state.get("tool_history", [])
                ),
                "policy_blocked": bool(state.get("injection_flags"))
                or any(
                    not t["result"].get("ok", True)
                    or (
                        t["tool_name"] == "check_refund_eligibility"
                        and not t["result"].get("eligible")
                    )
                    for t in state.get("tool_history", [])
                ),
            },
        }

    def validate_tool_call(self, state: AgentState) -> dict[str, Any]:
        t0 = self.tracer.now_ms()
        raw_call = state.get("pending_tool_call") or {}
        mode = state["mode"]
        customer_id = state["customer_id"]

        try:
            validated = ToolCallRequest.model_validate(raw_call)
        except Exception as exc:
            self.tracer.emit(
                trace_id=state["trace_id"],
                thread_id=state["thread_id"],
                step_name="validate_tool_call",
                latency_ms=self.tracer.now_ms() - t0,
                status="blocked",
                reason="Invalid tool request schema",
                mode=mode,
            )
            return {
                "tool_blocked": True,
                "tool_block_reason": f"Invalid tool call schema: {exc}",
            }

        args = dict(validated.arguments)
        if mode == "improved":
            args["customer_id"] = customer_id

        if validated.tool_name in {"create_refund", "create_support_ticket"}:
            if not args.get("idempotency_key"):
                digest = hashlib.sha256(
                    f"{state['thread_id']}:{validated.tool_name}:{args.get('order_id')}:{state['turn_index']}".encode()
                ).hexdigest()[:12]
                args["idempotency_key"] = f"idem-{validated.tool_name}-{digest}"
            args.setdefault("thread_id", state["thread_id"])

        if mode == "improved" and validated.tool_name == "create_refund":
            order_id = str(args.get("order_id", ""))
            order = self.repo.get_order(order_id)
            existing_refunds = self.repo.get_refunds_for_order(order_id) if order else []
            elig = evaluate_refund_eligibility(
                order=order,
                existing_refunds=existing_refunds,
                pending_refunds=[],
                requested_amount=float(args.get("amount", 0.0)),
                policy=self.rules.refund_policy,
                customer_id=customer_id,
                rules=self.rules,
            )
            if not elig.eligible:
                self.tracer.emit(
                    trace_id=state["trace_id"],
                    thread_id=state["thread_id"],
                    step_name="validate_tool_call",
                    latency_ms=self.tracer.now_ms() - t0,
                    tool_name=validated.tool_name,
                    status="blocked",
                    reason=elig.reason,
                    mode=mode,
                )
                return {
                    "pending_tool_call": {"tool_name": validated.tool_name, "arguments": args},
                    "tool_blocked": True,
                    "tool_block_reason": elig.reason,
                }

        self.tracer.emit(
            trace_id=state["trace_id"],
            thread_id=state["thread_id"],
            step_name="validate_tool_call",
            latency_ms=self.tracer.now_ms() - t0,
            tool_name=validated.tool_name,
            status="ok",
            mode=mode,
        )
        return {
            "pending_tool_call": {"tool_name": validated.tool_name, "arguments": args},
            "tool_blocked": False,
            "tool_block_reason": "",
        }

    async def execute_tool(self, state: AgentState) -> dict[str, Any]:
        t0 = self.tracer.now_ms()
        call = state.get("pending_tool_call") or {}
        tool_name = str(call.get("tool_name", "unknown"))
        args = dict(call.get("arguments", {}))
        mode = state["mode"]
        tool_count = state.get("tool_calls_count", 0) + 1

        if state.get("tool_blocked"):
            result_payload = {
                "ok": False,
                "status": "rejected",
                "reason": state.get("tool_block_reason", "Blocked by policy gate."),
                "policy_version": self.rules.policy_version,
            }
            history = list(state.get("tool_history", []))
            history.append({"tool_name": tool_name, "arguments": args, "result": result_payload})
            self.tracer.emit(
                trace_id=state["trace_id"],
                thread_id=state["thread_id"],
                step_name="execute_tool",
                latency_ms=self.tracer.now_ms() - t0,
                tool_name=tool_name,
                status="blocked",
                reason=result_payload["reason"],
                mode=mode,
            )
            return {
                "tool_calls_count": tool_count,
                "last_tool_result": result_payload,
                "last_retry_count": 0,
                "tool_history": history,
            }

        if mode == "baseline" and tool_name == "create_refund":
            from src.database.connection import transaction
            from src.database.models import RefundRecord
            from src.database.repositories import utc_now_iso

            order_id = str(args.get("order_id", ""))
            amount = float(args.get("amount", 0.0))
            refund_id = f"BASE-REF-{uuid.uuid4().hex[:8].upper()}"
            order = self.repo.get_order(order_id)
            if order is not None:
                now = utc_now_iso()
                with transaction(self.repo.db_path) as conn:
                    conn.execute(
                        """
                        INSERT INTO refunds (
                            refund_id, order_id, customer_id, amount, currency, status,
                            reason, idempotency_key, policy_version, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            refund_id,
                            order_id,
                            state["customer_id"],
                            amount,
                            "BRL",
                            "completed",
                            "Baseline unguarded refund",
                            f"base-{uuid.uuid4().hex}",
                            "baseline",
                            now,
                        ),
                    )
                result_payload = RefundRecord(
                    refund_id=refund_id,
                    order_id=order_id,
                    customer_id=state["customer_id"],
                    amount=amount,
                    currency="BRL",
                    status="completed",
                    reason="Baseline unguarded refund",
                    policy_version="baseline",
                    idempotency_key="baseline",
                    created_at=now,
                ).model_dump()
                result_payload["ok"] = True
            else:
                result_payload = {"ok": False, "reason": "Order not found"}
            history = list(state.get("tool_history", []))
            history.append({"tool_name": tool_name, "arguments": args, "result": result_payload})
            self.tracer.emit(
                trace_id=state["trace_id"],
                thread_id=state["thread_id"],
                step_name="execute_tool",
                latency_ms=self.tracer.now_ms() - t0,
                tool_name=tool_name,
                status="ok",
                mode=mode,
            )
            return {
                "tool_calls_count": tool_count,
                "last_tool_result": result_payload,
                "last_retry_count": 0,
                "tool_history": history,
            }

        result_payload, retries = await self.mcp_client.call_tool(tool_name, args)
        history = list(state.get("tool_history", []))
        history.append({"tool_name": tool_name, "arguments": args, "result": result_payload})

        self.tracer.emit(
            trace_id=state["trace_id"],
            thread_id=state["thread_id"],
            step_name="execute_tool",
            latency_ms=self.tracer.now_ms() - t0,
            tool_name=tool_name,
            status="ok" if result_payload.get("ok", True) else "rejected",
            retry_count=retries,
            mode=mode,
        )
        return {
            "tool_calls_count": tool_count,
            "last_tool_result": result_payload,
            "last_retry_count": retries,
            "tool_history": history,
        }

    def record_event(self, state: AgentState) -> dict[str, Any]:
        t0 = self.tracer.now_ms()
        mode = state["mode"]
        call = state.get("pending_tool_call") or {}
        tool_name = str(call.get("tool_name", ""))
        args = dict(call.get("arguments", {}))
        result = state.get("last_tool_result") or {}

        if mode == "improved":
            order_id = args.get("order_id")
            if state.get("tool_blocked"):
                self.repo.append_event(
                    customer_id=state["customer_id"],
                    thread_id=state["thread_id"],
                    event_type="policy_blocked",
                    order_id=order_id,
                    amount=args.get("amount"),
                    actor="policy_gate",
                    metadata={
                        "tool_name": tool_name,
                        "reason": state.get("tool_block_reason", ""),
                        "policy_version": self.rules.policy_version,
                        "idempotency_key": args.get("idempotency_key", ""),
                    },
                )
            elif tool_name in {"get_order", "get_order_status"} and result.get("ok"):
                self.repo.append_event(
                    customer_id=state["customer_id"],
                    thread_id=state["thread_id"],
                    event_type="order_looked_up",
                    order_id=order_id,
                    actor="mcp_order_tools",
                    metadata={"policy_version": self.rules.policy_version},
                )
            elif tool_name == "check_refund_eligibility":
                self.repo.append_event(
                    customer_id=state["customer_id"],
                    thread_id=state["thread_id"],
                    event_type="refund_requested",
                    order_id=order_id,
                    amount=args.get("requested_amount"),
                    actor="mcp_refund_tools",
                    metadata={
                        "eligible": result.get("eligible"),
                        "policy_version": self.rules.policy_version,
                    },
                )
                if not result.get("eligible"):
                    self.repo.append_event(
                        customer_id=state["customer_id"],
                        thread_id=state["thread_id"],
                        event_type="refund_rejected",
                        order_id=order_id,
                        amount=args.get("requested_amount"),
                        actor="policy_gate",
                        metadata={
                            "reason": result.get("reason", ""),
                            "policy_version": self.rules.policy_version,
                        },
                    )

        orders = [o.model_dump() for o in self.repo.list_orders_for_customer(state["customer_id"])]
        refunds = [r.model_dump() for r in self.repo.get_refunds_for_customer(state["customer_id"])]
        tickets = [
            t.model_dump()
            for t in self.repo.list_support_tickets_for_customer(state["customer_id"])
        ]

        self.tracer.emit(
            trace_id=state["trace_id"],
            thread_id=state["thread_id"],
            step_name="record_event",
            latency_ms=self.tracer.now_ms() - t0,
            tool_name=tool_name,
            mode=mode,
        )
        return {
            "pending_tool_call": None,
            "tool_blocked": False,
            "tool_block_reason": "",
            "verified_orders": orders,
            "verified_refunds": refunds,
            "verified_tickets": tickets,
        }

    def final_response(self, state: AgentState) -> dict[str, Any]:
        t0 = self.tracer.now_ms()
        raw_final = state.get("final_response") or {"answer": "How can I help you today?"}
        validated = FinalAgentResponse.model_validate(raw_final)

        messages = list(state.get("messages") or [])
        messages.append({"role": "assistant", "content": validated.answer})

        if state["mode"] == "improved":
            self.repo.append_turn(
                thread_id=state["thread_id"],
                customer_id=state["customer_id"],
                turn_index=state["turn_index"],
                role="assistant",
                content=validated.answer,
                entities=extract_entities(validated.answer),
            )
            self.episode_extractor.extract(
                customer_id=state["customer_id"],
                thread_id=state["thread_id"],
                user_message=state["user_message"],
                assistant_message=validated.answer,
            )

        self.tracer.emit(
            trace_id=state["trace_id"],
            thread_id=state["thread_id"],
            step_name="final_response",
            latency_ms=self.tracer.now_ms() - t0,
            output_tokens=estimate_tokens(validated.answer),
            mode=state["mode"],
        )
        return {
            "messages": messages,
            "response": validated.answer,
            "final_response": validated.model_dump(),
        }

    def compact_memory(self, state: AgentState) -> dict[str, Any]:
        t0 = self.tracer.now_ms()
        if state["mode"] == "baseline":
            preserve = self.rules.context.recent_messages_to_preserve
            trimmed = list(state["messages"][-preserve:])
            self.tracer.emit(
                trace_id=state["trace_id"],
                thread_id=state["thread_id"],
                step_name="compact_memory",
                latency_ms=self.tracer.now_ms() - t0,
                status="baseline_sliding_window",
                mode=state["mode"],
            )
            return {
                "messages": trimmed,
                "compaction_triggered": False,
                "compaction_snapshot_id": None,
            }

        outcome = self.compactor.compact(
            customer_id=state["customer_id"],
            thread_id=state["thread_id"],
            messages=state["messages"],
            db_conn=self.repo,
            context_config=self.rules.context,
        )
        prior_compacted = list(state.get("compacted_memories") or [])
        prior_count = int(state.get("compaction_count") or 0)
        if outcome.compacted and outcome.payload is not None:
            prior_compacted.append(outcome.payload.model_dump())
            prior_count += 1

        self.tracer.emit(
            trace_id=state["trace_id"],
            thread_id=state["thread_id"],
            step_name="compact_memory",
            latency_ms=self.tracer.now_ms() - t0,
            status="compacted" if outcome.compacted else "skipped_below_threshold",
            mode=state["mode"],
        )
        tokens_before = (
            outcome.tokens_before
            if outcome.compacted
            else int(state.get("last_compaction_tokens_before") or 0)
        )
        tokens_after = (
            outcome.tokens_after
            if outcome.compacted
            else int(state.get("last_compaction_tokens_after") or 0)
        )
        return {
            "messages": outcome.preserved_messages,
            "compaction_triggered": outcome.compacted,
            "compaction_snapshot_id": outcome.snapshot.snapshot_id if outcome.snapshot else None,
            "compacted_memories": prior_compacted,
            "compaction_count": prior_count,
            "last_compaction_tokens_before": tokens_before,
            "last_compaction_tokens_after": tokens_after,
        }
