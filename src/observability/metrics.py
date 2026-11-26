"""Metrics aggregation over structured trace records and agent runtime events."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class RunMetrics(BaseModel):
    """Summary metrics computed from trace records and runtime telemetry."""

    total_events: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    avg_tokens_per_turn: float = 0.0
    total_cost_usd: float = 0.0
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    agent_p95_latency_ms: float = 0.0
    retrieval_p95_latency_ms: float = 0.0
    tool_p95_latency_ms: float = 0.0
    tool_calls: int = 0
    blocked_calls: int = 0
    policy_rejection_count: int = 0
    duplicate_prevention_count: int = 0
    compaction_count: int = 0
    compaction_tokens_saved: int = 0
    retries: int = 0
    retry_rate: float = 0.0
    step_latencies_ms: dict[str, float] = Field(default_factory=dict)


def estimate_cost_usd(input_tokens: int, output_tokens: int, provider: str = "mock") -> float:
    """Estimate token cost in USD (0 for mock/ollama, Flash-tier rates for hosted models)."""
    if provider in {"mock", "ollama"}:
        return 0.0
    return (input_tokens * 0.00000015) + (output_tokens * 0.00000060)


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, math.ceil(0.95 * len(ordered)) - 1))
    return round(ordered[idx], 2)


class MetricsCollector:
    """Collects and aggregates latency, token, cost, compaction, policy, and retry metrics."""

    def __init__(self) -> None:
        self.compaction_events: list[tuple[int, int]] = []
        self.policy_rejections: int = 0
        self.duplicate_preventions: int = 0

    def record_compaction(self, tokens_before: int, tokens_after: int) -> None:
        self.compaction_events.append((tokens_before, tokens_after))

    def record_policy_rejection(self, is_duplicate: bool = False) -> None:
        self.policy_rejections += 1
        if is_duplicate:
            self.duplicate_preventions += 1

    def summarize(self, records: list[dict[str, Any]], turn_count: int | None = None) -> RunMetrics:
        """Compute RunMetrics over trace records plus collector counters."""
        if not records and not self.compaction_events:
            return RunMetrics()

        latencies = [float(r.get("latency_ms", 0.0)) for r in records]
        agent_lats = [
            float(r.get("latency_ms", 0.0)) for r in records if r.get("step_name") == "agent"
        ]
        retrieval_lats = [
            float(r.get("latency_ms", 0.0))
            for r in records
            if r.get("step_name") == "retrieve_memory"
        ]
        tool_lats = [
            float(r.get("latency_ms", 0.0)) for r in records if r.get("step_name") == "execute_tool"
        ]

        by_step: dict[str, list[float]] = {}
        for r in records:
            step = str(r.get("step_name", "unknown"))
            by_step.setdefault(step, []).append(float(r.get("latency_ms", 0.0)))
        step_avg = {k: round(sum(v) / len(v), 2) for k, v in by_step.items()}

        total_in = sum(int(r.get("input_tokens", 0)) for r in records)
        total_out = sum(int(r.get("output_tokens", 0)) for r in records)
        total_cost = sum(float(r.get("estimated_cost_usd", 0.0)) for r in records)
        tool_calls = sum(1 for r in records if r.get("step_name") == "execute_tool")
        blocked = sum(1 for r in records if r.get("status") in {"blocked", "rejected"})
        compacted_spans = sum(1 for r in records if r.get("status") == "compacted")
        retries = sum(int(r.get("retry_count", 0)) for r in records)

        inferred_turns = turn_count or max(
            1, sum(1 for r in records if r.get("step_name") == "validate_input")
        )
        tokens_saved = sum(max(0, b - a) for b, a in self.compaction_events)

        return RunMetrics(
            total_events=len(records),
            total_input_tokens=total_in,
            total_output_tokens=total_out,
            avg_tokens_per_turn=round((total_in + total_out) / inferred_turns, 2),
            total_cost_usd=round(total_cost, 6),
            avg_latency_ms=round(sum(latencies) / max(1, len(latencies)), 2),
            p95_latency_ms=_p95(latencies),
            agent_p95_latency_ms=_p95(agent_lats),
            retrieval_p95_latency_ms=_p95(retrieval_lats),
            tool_p95_latency_ms=_p95(tool_lats),
            tool_calls=tool_calls,
            blocked_calls=blocked,
            policy_rejection_count=max(blocked, self.policy_rejections),
            duplicate_prevention_count=self.duplicate_preventions,
            compaction_count=max(compacted_spans, len(self.compaction_events)),
            compaction_tokens_saved=tokens_saved,
            retries=retries,
            retry_rate=round(retries / max(1, tool_calls), 4),
            step_latencies_ms=step_avg,
        )


def aggregate_trace_metrics(records: list[dict[str, Any]]) -> RunMetrics:
    """Aggregate token, cost, latency, and tool call metrics from trace spans."""
    return MetricsCollector().summarize(records)


def load_traces_from_jsonl(path: str | Path, limit: int = 200) -> list[dict[str, Any]]:
    """Read the most recent JSONL trace entries from disk."""
    p = Path(path)
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8") as fh:
        lines = [line.strip() for line in fh if line.strip()]
    return [json.loads(line) for line in lines[-limit:]]
