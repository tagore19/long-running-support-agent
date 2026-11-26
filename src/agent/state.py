"""Typed LangGraph state definition and Pydantic tool validation models."""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field


class ToolCallRequest(BaseModel):
    """Validated tool invocation request before hitting MCP stdio."""

    tool_name: Literal[
        "get_order",
        "get_order_status",
        "check_refund_eligibility",
        "create_refund",
        "get_refund_status",
        "create_support_ticket",
        "get_support_ticket",
    ]
    arguments: dict[str, Any] = Field(default_factory=dict)


class AgentState(TypedDict, total=False):
    """State carried through the LangGraph StateGraph and persisted in SqliteSaver."""

    thread_id: str
    customer_id: str
    user_message: str
    messages: list[dict[str, str]]
    step_count: int
    tool_calls_count: int
    response: str
    error: str | None

    # Extended runtime state for multi-layer memory, MCP tool loop, and tracing
    trace_id: str
    mode: str
    turn_index: int
    document_text: str | None
    image_text: str | None
    wrapped_user_message: str
    wrapped_document_text: str
    wrapped_image_text: str
    injection_flags: list[str]
    injection_suspected: bool
    verified_orders: list[dict[str, Any]]
    verified_refunds: list[dict[str, Any]]
    verified_tickets: list[dict[str, Any]]
    episodes: list[dict[str, Any]]
    retrieved_items: list[dict[str, Any]]
    memory_bundle: dict[str, Any]
    built_context: dict[str, Any]
    assembled_context: dict[str, Any]
    pending_tool_call: dict[str, Any] | None
    tool_blocked: bool
    tool_block_reason: str
    last_tool_result: dict[str, Any] | None
    last_retry_count: int
    tool_history: list[dict[str, Any]]
    final_response: dict[str, Any] | None
    compaction_triggered: bool
    compaction_snapshot_id: str | None
    compacted_memories: list[dict[str, Any]]
    compaction_count: int
    last_compaction_tokens_before: int
    last_compaction_tokens_after: int
    compacted_memories: list[dict[str, Any]]
    compaction_count: int
    last_compaction_tokens_before: int
    last_compaction_tokens_after: int
    compacted_memories: list[dict[str, Any]]
    compaction_count: int
    last_compaction_tokens_before: int
    last_compaction_tokens_after: int
