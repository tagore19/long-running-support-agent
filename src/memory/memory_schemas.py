"""Pydantic schemas and deterministic token/entity helpers for the 5-layer memory."""

from __future__ import annotations

import math
import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from src.database.models import (
    EpisodeRecord,
    EventRecord,
    OrderRecord,
    RefundRecord,
    SupportCaseRecord,
)

ENTITY_PATTERN = re.compile(r"\b(?:ORD|CUST|REF|TKT|EVT)-[A-Z0-9]+\b", re.IGNORECASE)

try:
    import tiktoken  # type: ignore[import-not-found]

    _TIKTOKEN_ENC = tiktoken.get_encoding("cl100k_base")
except Exception:
    _TIKTOKEN_ENC = None


def estimate_tokens(text: str) -> int:
    """Estimate tokens using tiktoken when available, else deterministic len(text)//4 heuristic."""
    if not text:
        return 0
    if _TIKTOKEN_ENC is not None:
        return len(_TIKTOKEN_ENC.encode(text))
    return max(1, math.ceil(len(text) / 4.0))


def estimate_messages_tokens(messages: list[dict[str, str]]) -> int:
    """Estimate total tokens across a list of role/content messages."""
    return sum(estimate_tokens(m.get("content", "")) + 4 for m in messages)


def extract_entities(text: str) -> list[str]:
    """Extract normalized domain entity IDs (ORD-*, CUST-*, REF-*, TKT-*, EVT-*)."""
    if not text:
        return []
    seen: dict[str, None] = {}
    for match in ENTITY_PATTERN.findall(text):
        seen[match.upper()] = None
    return list(seen.keys())


class CustomerStateSnapshot(BaseModel):
    """Layer 2 authoritative operational state snapshot derived from SQLite/events."""

    customer_id: str
    orders: list[OrderRecord] = Field(default_factory=list)
    refunds: list[RefundRecord] = Field(default_factory=list)
    support_tickets: list[SupportCaseRecord] = Field(default_factory=list)
    recent_events: list[EventRecord] = Field(default_factory=list)
    active_issue: str | None = None


class RetrievedMemoryBundle(BaseModel):
    """Layer 4 hybrid retrieval output combining exact SQL records, episodes, and turns."""

    exact_records: list[dict[str, Any]] = Field(default_factory=list)
    episodes: list[EpisodeRecord] = Field(default_factory=list)
    relevant_messages: list[RetrievedMemoryItem] = Field(default_factory=list)
    recent_messages: list[dict[str, str]] = Field(default_factory=list)


class BuiltContext(BaseModel):
    """Layer 5 bounded context output produced by ActiveContextBuilder."""

    structured_state_text: str
    episodes_text: str
    retrieved_history_text: str
    recent_messages_text: str
    estimated_tokens: int
    truncated_items_count: int = 0
    compaction_summary_text: str = ""
    recent_messages: list[dict[str, str]] = Field(default_factory=list)
    token_budget: int = 3000


class CompactedMemoryPayload(BaseModel):
    """Pydantic-validated structured memory object produced by adaptive compaction."""

    customer_preferences: list[str] = Field(default_factory=list)
    active_issues: list[str] = Field(default_factory=list)
    promises_made: list[dict[str, str]] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    important_events: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    conversation_summary: str = ""
    source_thread_id: str = ""
    source_turn_range: tuple[int, int] = (1, 1)
    compacted_at: str = ""
    stripped_unverified_claims: list[str] = Field(default_factory=list)


CompactedMemory = CompactedMemoryPayload


class RetrievedMemoryItem(BaseModel):
    """Ranked memory hit returned by Layer 4 hybrid retrieval."""

    source: Literal["episode", "turn", "event"]
    item_id: str
    content: str
    entities: list[str] = Field(default_factory=list)
    score: float
    exact_entity_match: bool = False
    created_at: str


class AssembledContext(BaseModel):
    """Bounded prompt context assembled by Layer 5 within token_budget."""

    verified_state_block: str
    compaction_summary_block: str
    episodes_block: str
    retrieved_history_block: str
    recent_messages: list[dict[str, str]]
    estimated_tokens: int
    token_budget: int
    trimmed_sections: list[str] = Field(default_factory=list)
    truncated_items_count: int = 0


RetrievedMemoryBundle.model_rebuild()

__all__ = [
    "AssembledContext",
    "BuiltContext",
    "CompactedMemory",
    "CompactedMemoryPayload",
    "CustomerStateSnapshot",
    "EpisodeRecord",
    "EventRecord",
    "RetrievedMemoryBundle",
    "RetrievedMemoryItem",
    "estimate_messages_tokens",
    "estimate_tokens",
    "extract_entities",
]
