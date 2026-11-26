"""FastAPI service exposing chat, state inspection, and trace metrics endpoints."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from scripts.seed_data import seed_databases

from src.agent.graph import SupportAgentRunner
from src.config import get_settings, load_app_config
from src.database.repositories import SupportRepository
from src.observability.metrics import aggregate_trace_metrics, load_traces_from_jsonl


class ChatRequest(BaseModel):
    """Inbound customer support chat turn."""

    thread_id: str = Field(..., min_length=1)
    customer_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)
    mode: Literal["improved", "baseline"] | None = None
    document_text: str | None = None
    image_text: str | None = None


class ChatResponse(BaseModel):
    """Structured response returned by POST /chat."""

    thread_id: str
    customer_id: str
    trace_id: str
    response: str
    answer: str
    referenced_order_ids: list[str]
    actions_taken: list[str]
    escalated: bool
    policy_blocked: bool
    injection_suspected: bool = False
    compaction_triggered: bool
    compaction_snapshot_id: str | None = None
    compaction_count: int = 0
    last_compaction_tokens_before: int = 0
    last_compaction_tokens_after: int = 0
    context_tokens: int


def _ensure_seeded() -> SupportRepository:
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    _ensure_seeded()
    runner = SupportAgentRunner()
    await runner.__aenter__()
    app.state.runner = runner
    try:
        yield
    finally:
        await runner.__aexit__(None, None, None)


app = FastAPI(
    title="Long-Running Customer Support Agent API",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, Any]:
    settings = get_settings()
    rules = load_app_config(settings.business_rules_path)
    return {
        "status": "ok",
        "policy_version": rules.policy_version,
        "llm_provider": settings.effective_llm_provider,
        "agent_mode": settings.agent_mode,
    }


@app.get("/customers")
def list_customers() -> list[dict[str, Any]]:
    repo = _ensure_seeded()
    return [c.model_dump() for c in repo.list_customers()]


@app.get("/customers/{customer_id}/state")
def get_customer_state(customer_id: str, thread_id: str | None = None) -> dict[str, Any]:
    repo = _ensure_seeded()
    customer = repo.get_customer(customer_id)
    if customer is None:
        raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found.")
    snapshot = repo.get_latest_compaction_snapshot(thread_id) if thread_id else None
    return {
        "customer": customer.model_dump(),
        "orders": [o.model_dump() for o in repo.list_orders_for_customer(customer_id)],
        "refunds": [r.model_dump() for r in repo.get_refunds_for_customer(customer_id)],
        "support_cases": [
            t.model_dump() for t in repo.list_support_tickets_for_customer(customer_id)
        ],
        "tool_operations": [
            op.model_dump() for op in repo.list_tool_operations(customer_id=customer_id, limit=50)
        ],
        "events": [
            e.model_dump()
            for e in repo.list_events(customer_id=customer_id, thread_id=thread_id, limit=200)
        ],
        "episodes": [
            ep.model_dump()
            for ep in repo.list_episodes(customer_id=customer_id, thread_id=thread_id, limit=50)
        ],
        "latest_compaction": snapshot.model_dump() if snapshot else None,
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    runner: SupportAgentRunner = app.state.runner
    state = await runner.arun_turn(
        thread_id=req.thread_id,
        customer_id=req.customer_id,
        user_message=req.message,
        mode=req.mode,
        document_text=req.document_text,
        image_text=req.image_text,
    )
    final = state.get("final_response") or {}
    ctx = state.get("assembled_context") or {}
    answer_text = str(state.get("response") or final.get("answer") or "")
    return ChatResponse(
        thread_id=req.thread_id,
        customer_id=req.customer_id,
        trace_id=str(state.get("trace_id", "")),
        response=answer_text,
        answer=answer_text,
        referenced_order_ids=list(final.get("referenced_order_ids", [])),
        actions_taken=list(final.get("actions_taken", [])),
        escalated=bool(final.get("escalated", False)),
        policy_blocked=bool(final.get("policy_blocked", False)),
        injection_suspected=bool(state.get("injection_suspected", False)),
        compaction_triggered=bool(state.get("compaction_triggered", False)),
        compaction_snapshot_id=state.get("compaction_snapshot_id"),
        compaction_count=int(state.get("compaction_count", 0)),
        last_compaction_tokens_before=int(state.get("last_compaction_tokens_before", 0)),
        last_compaction_tokens_after=int(state.get("last_compaction_tokens_after", 0)),
        context_tokens=int(ctx.get("estimated_tokens", 0)),
    )


@app.get("/traces")
def get_traces(limit: int = 100) -> dict[str, Any]:
    settings = get_settings()
    trace_path = settings.resolve_path(settings.trace_log_path)
    records = load_traces_from_jsonl(trace_path, limit=limit)
    metrics = aggregate_trace_metrics(records)
    return {
        "metrics": metrics.model_dump(),
        "records": records,
    }
