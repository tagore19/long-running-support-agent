# System Design: Long-Running Customer Support Agent

This document records **why** each architectural decision was made — not just what was built.

---

## 1. `AgentState`, LangGraph Nodes & Conditional Routing

### `AgentState` (TypedDict)

```python
class AgentState(TypedDict, total=False):
    thread_id: str               # LangGraph checkpoint key
    customer_id: str             # pinned at session start — never LLM-editable
    user_message: str            # raw input (untrusted)
    mode: str                    # "improved" | "baseline"
    trace_id: str                # propagated through all spans
    messages: list[dict]         # LangGraph message accumulator
    wrapped_user_input: str      # HTML-escaped + <user_input_untrusted_data> tagged
    wrapped_document_text: str   # same for uploaded documents
    wrapped_image_text: str      # same for image captions
    injection_suspected: bool    # set by validate_input; blocks aggressive tool use
    tool_call_request: ToolCallRequest
    tool_result: dict
    compacted_memories: list[CompactedMemoryPayload]
    compaction_count: int
    last_compaction_tokens_before: int
    last_compaction_tokens_after: int
    assembled_context: BuiltContext
    final_response: dict
    compaction_triggered: bool
    event_recorded: bool
```

`customer_id` and `thread_id` are set at turn entry and **never** overwritten by any node — the LLM cannot change who it is talking to.

### Node pipeline

```
START
  → validate_input      # length check, injection scan, untrusted wrapping
  → load_state          # Layer 2 SQLite authoritative snapshot
  → retrieve_memory     # Layer 3/4 hybrid retrieval
  → build_context       # Layer 5 token-budget prompt assembly
  → agent               # LLM: proposes call_tool or final_answer
      ├─ call_tool → validate_tool_call → execute_tool → record_event
      │                   ↑ policy gate: ownership, eligibility, idempotency
      │                   loop back to agent if more tools needed
      └─ final_answer → final_response → compact_memory → END
```

**Why not one giant node?** Each node has a single responsibility and a testable input/output contract. The conditional edges (`route_after_agent`, `route_after_record_event`) are pure functions over `AgentState` — no hidden branching.

### `SqliteSaver` checkpoints vs. 5-layer long-term memory

| | `SqliteSaver` / `AsyncSqliteSaver` | 5-Layer Memory |
|---|---|---|
| **Purpose** | LangGraph internal message history within one session | Cross-session continuity and semantic recall |
| **Scope** | Within-thread `messages` list | Per-customer across all threads |
| **Authority** | Operational (message ordering) | Advisory (episodes, compaction summaries) |
| **Never** | Stores business facts (refund amounts, delivery status) | Overrides Layer 2 SQLite truth |

LangGraph checkpoints are wiped when a thread is deleted. The 5-layer memory persists independently.

---

## 2. Architectural Decision Records

### ADR-01: Append-only Event Store + Transactional Tables (not a single CQRS log)

**Decision:** Separate `events` (append-only, write-protected by `BEFORE UPDATE / BEFORE DELETE` SQL triggers) from transactional tables (`orders`, `refunds`, `support_cases`, `tool_operations`).

**Why not log-only?** The LLM needs authoritative answers like "what is the current refund status of ORD-1001?" in O(1) — scanning an event log for every read is expensive and error-prone. Transactional tables give instant current state; the event log gives an auditable history of exactly what changed and when.

**Why not just the transactional tables?** Without the append-only event log, there is no tamper-evident audit trail. The `BEFORE UPDATE / BEFORE DELETE` triggers abort any attempt to silently mutate or erase an event — even a buggy node cannot retroactively delete a refund record.

---

### ADR-02: Episodic + Hybrid Retrieval (not lossy prose summarization alone)

**Decision:** `EpisodeExtractor` saves structured keyword-tagged episodes (`src/memory/episode_extractor.py`). `HybridMemoryRetriever` scores them using exact entity matching + SQL customer/thread filter + TF-IDF cosine similarity + recency + importance, returning a ranked `RetrievedMemoryBundle`.

**Why not "just summarize at the end of each turn"?**  
- Prose summaries lose exact amounts, order IDs, and ticket IDs — the very facts that matter most for policy decisions.  
- TF-IDF cosine retrieval returns episodes *relevant to the current query*, not just the most recent. A complaint from turn 2 is retrieved when the user asks about it in turn 28.

**Why TF-IDF instead of a vector DB?**  
Deterministic, zero-dependency, works offline. For the scale of this project (< 1,000 episodes per customer), it is fast enough (< 5 ms per query). The retriever is designed with a clean interface so it can be replaced by pgvector or Chroma without changing any other module.

---

### ADR-03: Token-Budget Compaction (not fixed turn counts)

**Decision:** `TokenBudgetCompactor.should_compact()` triggers when `estimated_tokens >= compaction_trigger_tokens` (default: 8,500) rather than every N turns.

**Why?** Short, terse conversations stay compact indefinitely; long, tool-heavy conversations compact earlier. A fixed turn count would either compact too aggressively (wasting tokens on short turns) or not aggressively enough (blowing the context window on long tool chains).

**Cross-verification:** After compaction, the `CompactedMemoryPayload` is cross-checked against Layer 2 (`orders`, `refunds`, `support_cases`) to ensure no authoritative fact in the summary contradicts the DB. If it does, the DB record wins.

---

### ADR-04: `check_refund_eligibility` Separated from Idempotent `create_refund`

**Decision:** Two distinct MCP tools: `check_refund_eligibility` (read-only, no side effects) and `create_refund` (write, idempotency key required).

**Why?** The LLM can call `check_refund_eligibility` speculatively to formulate an answer without risk of accidentally creating a refund. The separation also makes the tool log easy to audit — a `check_refund_eligibility` call never causes a financial mutation.

**Double-gate:** Eligibility is evaluated *twice* — once in `validate_tool_call` (Python, before MCP dispatch) and once inside `create_refund` (Python inside the SQLite transaction). The second check catches any TOCTOU race.

---

### ADR-05: MCP `stdio` Transport Boundary

**Decision:** `FastMCP("CustomerSupportMCP")` runs as a child process over `stdio`; `MCPToolClient` owns a persistent async worker queue.

**Why stdio over HTTP?** stdio is the official MCP reference transport for single-host deployments. It eliminates network configuration, authentication headers, and port collisions. The persistent worker avoids per-call subprocess spawn overhead (critical on Windows where subprocess startup is ~50 ms).

**Why not in-process function calls?** The MCP boundary enforces that tool implementations are owned, versioned, and testable independently of the agent node graph. A buggy tool cannot corrupt `AgentState` through a shared Python object — it can only return a serialized result or raise an exception that the node catches.

---

### ADR-06: PII Redaction Before Logging

**Decision:** `src/observability/redaction.py` applies regex masking + field allowlist before every JSONL trace write. The allowlist (`SAFE_TRACE_FIELDS`) enumerates the exact keys that may appear in a span; any other key is replaced with `[REDACTED]`.

**Why allowlist-first (not denylist)?**  
Denylist-based redaction (block known bad patterns) always has gaps — a new field name can silently leak PII. Allowlist-first means every new field must be explicitly approved; unknown fields are blocked by default.

**Measured regex patterns (all deterministic, no LLM):**

| Pattern | Placeholder |
|---|---|
| `user@domain.com` | `[REDACTED_EMAIL]` |
| `+55 11 99999-9999` | `[REDACTED_PHONE]` |
| CPF `000.000.000-00` | `[REDACTED_PII]` |
| CEP `00000-000` | `[REDACTED_PII]` |
| `sk-...`, `AIza...`, Bearer tokens | `[REDACTED_SECRET]` |
| Street addresses (heuristic) | `[REDACTED_ADDRESS]` |

---

## 3. Model Choices, Latency & Token Figures

The checked-in eval report is a deterministic offline run using `LLM_PROVIDER=mock` on Windows, recorded 2026-11-26. Mock runtime cost is **$0.00**; token counts are application estimates, not billed model usage. Latencies include local graph, SQLite, and MCP stdio overhead and are not model-inference benchmarks.

| Metric | Baseline | Improved |
|---|---:|---:|
| Task success | 50.0% (10/20) | 100.0% (20/20) |
| Memory recall | 0.0% | 100.0% |
| Policy violations | 35.0% | 0.0% |
| Duplicate actions | 15.0% | 0.0% |
| Average tokens per turn | 999.0 | 1,821.1 |
| p95 turn latency | 150.1 ms | 283.6 ms |
| Estimated model cost for these mock evals | $0.00 | $0.00 |

`gemini-2.5-flash` is the optional hosted model for structured `AgentDecision` output; `gemini-2.0-flash-lite` is configured for compaction. No live Gemini latency or cost measurement is included because it depends on account tier, region, network, and current provider pricing. `LLM_PROVIDER=mock` is the reproducible, zero-cost default for tests and CI.

The local 32-turn mock demo also completed with 1 refund, 1 escalation ticket, 9 events, and 4 compactions (last compaction: 712 to 180 estimated tokens). Its measured average was 3,286.66 estimated tokens per turn and p95 latency was 52.11 ms; these timings are specific to this machine and run.

---

## 4. Clean-Clone Verification Checklist

Run these commands in order on a fresh machine (or fresh directory) to verify full reproducibility:

```bash
# Step 1 — Clone
git clone https://github.com/tagore19/long-running-support-agent.git
cd long-running-support-agent

# Step 2 — Environment
cp .env.example .env
# (Optionally) add GEMINI_API_KEY for live LLM. Leave blank for mock mode.

# Step 3 — Virtual environment
python3.11 -m venv .venv
source .venv/bin/activate      # Linux / macOS
# OR: .\.venv\Scripts\Activate.ps1  (Windows PowerShell)

# Step 4 — Install
pip install --upgrade pip
pip install -e ".[dev]"

# Step 5 — Lint
ruff check .
# Expected: "All checks passed!"

# Step 6 — Seed
python -m scripts.seed_data
# Expected: "Seeding complete." (or "already exists, skipping" on re-run)

# Step 7 — Tests
pytest -q
# Expected: all tests pass

# Step 8 — Demo
python -m scripts.run_demo
# Verified locally: 32 turns, 1 refund, 1 ticket, 9 events, 4 compactions

# Step 9 — Evals
python -m evals.runner --mode improved --min-score 0.90 --output data/evals/improved.json
# Expected: 20/20 PASS, Policy 0.0%, Duplicates 0.0%

python -m evals.runner --mode baseline --output data/evals/baseline.json
# Expected: ~10/20 PASS (baseline intentionally weak)

python -m evals.report \
    --baseline data/evals/baseline.json \
    --improved data/evals/improved.json \
    --out data/evals/before_after_report.md
# Expected: Report written to data/evals/before_after_report.md

# Step 10 — Docker (optional, requires Docker Desktop)
docker build -t customer-support-agent .
docker compose up --build
# Expected: api healthy on :8000, ui on :8501
```

All steps must pass before submitting or deploying.
