"""Allowlisted structured JSONL tracing for LangGraph execution steps."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.observability.redaction import sanitize_log_payload


class TraceLogger:
    """Writes allowlisted, PII/secret-redacted JSONL trace spans and retains session spans."""

    def __init__(self, log_path: str | Path = "logs/traces.jsonl", *, echo_stdout: bool = False) -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.echo_stdout = echo_stdout
        self.records: list[dict[str, Any]] = []

    def emit(
        self,
        *,
        trace_id: str,
        thread_id: str,
        step_name: str,
        latency_ms: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
        estimated_cost_usd: float = 0.0,
        tool_name: str | None = None,
        status: str = "ok",
        retry_count: int = 0,
        mode: str = "improved",
        reason: str | None = None,
        error_code: str | None = None,
    ) -> dict[str, Any]:
        """Sanitize and persist a single trace event to JSONL."""
        raw: dict[str, Any] = {
            "timestamp": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "trace_id": trace_id,
            "thread_id": thread_id,
            "step_name": step_name,
            "latency_ms": round(latency_ms, 2),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "estimated_cost_usd": round(estimated_cost_usd, 6),
            "tool_name": tool_name,
            "status": status,
            "retry_count": retry_count,
            "mode": mode,
        }
        if reason:
            raw["reason"] = reason
        if error_code:
            raw["error_code"] = error_code

        clean = sanitize_log_payload(raw)
        self.records.append(clean)
        line = json.dumps(clean)
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        if self.echo_stdout:
            print(line)
        return clean

    @staticmethod
    def now_ms() -> float:
        return time.perf_counter() * 1000.0


StructuredTracer = TraceLogger
