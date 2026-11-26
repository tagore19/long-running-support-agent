# Long-Running Customer Support Agent

> **Stateful, multi-turn (30+ turns) e-commerce customer-support agent demo**
> built with LangGraph · FastMCP `stdio` · SQLite · FastAPI

[![CI](https://github.com/tagore19/long-running-support-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/tagore19/long-running-support-agent/actions/workflows/ci.yml)

---

## What This Project Does

This agent handles e-commerce support workflows — order lookups, refund requests, and complaint escalation — with multi-turn behavior tested to 32+ turns. Its core safeguards are:

| Guarantee | How |
|---|---|
| **LLM never decides policy** | All refund limits, delivery checks, and ownership rules are enforced in deterministic Python (`src/policies/`) and inside SQLite transactions |
| **No duplicate refunds** | Idempotency keys stored in `tool_operations`; re-issuing the same key returns the same result without a second DB write |
| **Order ownership enforced** | Refund and order tools verify that the requested order belongs to the supplied `customer_id` before returning or mutating state |
| **Prompt-injection defenses** | Untrusted user, document, image, and order-note text is detected, escaped, and wrapped in explicit untrusted-data delimiters; deterministic policy checks remain authoritative |
| **No PII in logs** | Regex + allowlist redaction masks emails, phones, CPF/Aadhaar/CEP numbers, and secrets before every JSONL trace write |
| **Memory across 30+ turns** | 5-layer architecture: append-only event store → transactional SQLite → episodic episodes → hybrid TF-IDF retrieval → token-budget compaction |

---

## Core Engineering Priorities

1. **Framework, Ownership & Verifiability** — Every component is deterministic, inspectable, typed, and tested; no black-box cloud state.
2. **Zero-Friction Handover & Reproducibility** — Clone, copy `.env.example`, install, run. No secrets needed for tests or evals (mock mode).
3. **LLM Is Untrusted for State & Policy** — The LLM only proposes `AgentDecision` JSON; all mutations go through Python policy gates and SQLite transactions.

---

## Architecture

```mermaid
flowchart TD
    U([User / API]) --> VI[validate_input\ninjection scan · untrusted wrap]
    VI --> LS[load_state\nLayer 2 SQLite truth]
    LS --> RM[retrieve_memory\nLayer 3/4 hybrid retrieval]
    RM --> BC[build_context\nLayer 5 token-budget assembly]
    BC --> AG[agent\nGemini / mock LLM\nAgentDecision JSON]
    AG -->|call_tool| VT[validate_tool_call\npolicy gate · idempotency]
    AG -->|final_answer| FR[final_response]
    VT --> ET[execute_tool\nMCP stdio FastMCP]
    ET --> RE[record_event\nLayer 1 append-only]
    RE -->|continue| AG
    RE -->|done| FR
    FR --> CM[compact_memory\ntoken-budget triggered]
    CM --> END([Response])

    subgraph SQLite
        L1[(events\nappend-only triggers)]
        L2[(customers · orders\nrefunds · tickets · ops)]
        L3[(episodes)]
        CK[(LangGraph\ncheckpoints)]
    end

    subgraph Observability
        TR[TraceLogger\nJSONL · PII redacted]
        MT[MetricsCollector\nlatency · tokens · cost]
    end

    LS --- L2
    RE --- L1
    RM --- L3
    CM --- CK
    AG --- TR
    ET --- MT
```

> **MCP boundary:** `src/mcp_server/server.py` runs as a child process over `stdio`. `MCPToolClient` maintains a persistent worker with an async queue — no per-call subprocess spawn overhead.

---

## Setup — Local Python

**Prerequisites:** Git and Python 3.11 or newer. For the quickest start, install Docker Desktop with Docker Compose v2; no model API key is needed in mock mode.

Clone the repository:

```bash
git clone https://github.com/tagore19/long-running-support-agent.git
cd long-running-support-agent
```

Create `.env` and a virtual environment.

**Windows PowerShell:**

```powershell
Copy-Item .env.example .env
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
cp .env.example .env
python3.11 -m venv .venv
source .venv/bin/activate
```

Install dependencies and seed the local SQLite database:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m scripts.seed_data
```

For live Gemini calls only, add `GEMINI_API_KEY` to `.env` and set `LLM_PROVIDER=gemini` and `LLM_MODE=live`. Leave the defaults for fully offline use.

---

## Run the Agent

### Docker run

Create the environment file once:

**Windows PowerShell:**
```powershell
Copy-Item .env.example .env
```

**macOS / Linux:**
```bash
cp .env.example .env
```

Start the API:

```bash
docker compose up --build
```

Open <http://localhost:8000/docs> to chat through the interactive API docs. Stop the service with `docker compose down`.

### Local Python

```bash
uvicorn src.api.app:app --reload --port 8000
```

- API docs: <http://localhost:8000/docs>
- 32-turn deterministic demo: `python -m scripts.run_demo`

For each chat turn, POST to `/chat` with the **same `thread_id` and `customer_id`** so the agent can continue that conversation. Example request:

```json
{
  "thread_id": "demo-order-1",
  "customer_id": "CUST-1001",
  "message": "Where is my order?"
}
```

---

## Tests, Linter & Evals

Run these from the repository root after installing the development dependencies. They use deterministic fixtures and do not require API keys:

```bash
# 1. Lint
ruff check .

# 2. Unit, memory, security, and integration tests
pytest -q

# 3. Improved 20-case evaluation (must meet the 0.90 CI threshold)
python -m evals.runner --mode improved --min-score 0.90 --output data/evals/improved.json

# 4. Baseline comparison
python -m evals.runner --mode baseline --output data/evals/baseline.json

# 5. Generate a before/after report
python -m evals.report --baseline data/evals/baseline.json --improved data/evals/improved.json --out data/evals/before_after_report.md
```

The committed [before/after eval report](evals/reports/before-after.md) records the most recent checked-in run. Latency is machine- and run-dependent; the CI gate checks task success, policy violations, and duplicate actions.

### Verify a clean clone

After cloning, run the setup above, then run `ruff check .`, `pytest -q`, and the improved evaluation command. A successful run exits with status `0`; the eval command also fails if its score is below `0.90`. To validate the running API, start it with Docker Compose or the local Python command, confirm <http://localhost:8000/health> reports `"status": "ok"`, then open <http://localhost:8000/docs> and send two `/chat` requests with the same `thread_id` and `customer_id`.

---

## Data and Evaluation Scope

- `data/fixtures/olist_orders.json` is a compact, hand-authored Olist-shaped fixture used to exercise order and refund workflows; it is not the Kaggle Olist dataset.
- `data/fixtures/bitext_evals.json` contains support-intent examples in a Bitext-style format; it is not a downloaded Bitext corpus.
- `data/fixtures/long_conversations.json` contains an authored 32-turn stress conversation; it is not a LoCoMo or Multi-Session Chat sample.
- The fixed 20-case evaluation set is project-authored and deterministic. It validates framework behavior and guardrails, not model accuracy or performance on an external benchmark.

---

## Local Docker Deployment

The supported demo runs locally in Docker; no hosted demo URL is currently provided. The default configuration uses the offline mock model, so no API key is needed. To enable Gemini, set `GEMINI_API_KEY`, `LLM_PROVIDER=gemini`, and `LLM_MODE=live` in `.env`. Never commit credentials.

The Compose setup persists SQLite databases and trace logs in named Docker volumes. This is a local demo, not a production deployment; do not use it for real customer data or financial transactions.

### Demo transcript

```bash
python -m scripts.run_demo      # runs and saves a deterministic 32-turn transcript
```

---

## Rollback Steps

### Code rollback

```bash
# Tag releases before shipping
git tag v1.0.0
git push origin v1.0.0

# Check out a previous tag (if one exists)
git checkout v0.9.0
pip install -e ".[dev]"
```

### Database rollback

```bash
# Back up before seeding or migration
cp data/app.db data/app.db.bak

# Restore from backup
cp data/app.db.bak data/app.db

# Or wipe and re-seed from scratch (destroys all conversation history)
python -m scripts.seed_data --reset
```

### LangGraph checkpoint rollback

```bash
# Checkpoints are stored in data/checkpoints.db.
# To reset conversation state for a specific thread_id, delete its rows:
sqlite3 data/checkpoints.db \
  "DELETE FROM checkpoints WHERE thread_id = 'thread-your-id';"
```

### Deployment rollback

```bash
# Rebuild and restart the local Docker service from the checked-out revision
docker compose down
docker compose up --build
```

---

## Repository Structure

```
├── config/
│   └── business_rules.yaml     Policy rules, guardrails, token budgets
├── src/
│   ├── agent/
│   │   ├── graph.py            LangGraph StateGraph compilation + SupportAgentRunner
│   │   ├── nodes.py            All 9 LangGraph node functions
│   │   ├── routing.py          Conditional edge routing functions
│   │   └── state.py            AgentState TypedDict
│   ├── api/app.py              FastAPI REST endpoints + lifespan
│   ├── config.py               Pydantic v2 AppConfig, Settings, load_app_config()
│   ├── database/               SQLite connection, schema triggers, models, repositories
│   ├── llm.py                  LLMClient (Gemini / mock) + AgentDecision schema
│   ├── memory/                 5-layer memory schemas, retrieval, episodes, compaction
│   ├── mcp_server/             FastMCP stdio server + MCPToolClient
│   ├── observability/          TraceLogger, MetricsCollector, PII redaction
│   └── policies/               Deterministic refund policy, auth, injection detection
├── evals/
│   ├── dataset.json            20 deterministic evaluation scenarios
│   ├── runner.py               Threshold-gated eval runner (--mode/--min-score/--output)
│   └── report.py               Markdown/JSON before-after comparison report
├── scripts/
│   ├── seed_data.py            Idempotent SQLite seeder (12 customers, 35 orders)
│   └── run_demo.py             32-turn deterministic demo script
├── tests/
│   ├── unit/                   Policy, database, foundation tests
│   ├── memory/                 5-layer memory, compaction, retrieval tests
│   ├── security/               PII redaction, injection, tracing tests
│   └── integration/            End-to-end LangGraph + MCP + FastAPI tests
├── Dockerfile                  Multi-stage build (builder → runtime, non-root)
├── docker-compose.yml          FastAPI (:8000) with persistent data volumes
├── .github/workflows/ci.yml   Ruff → pytest → evals gate on every push/PR
├── DESIGN.md                   Architecture decisions and design rationale
└── AI_USAGE.md                 AI tool usage log and three concrete AI mistakes caught
```

---

## Known Limitations

- **SQLite concurrency:** SQLite's `BEGIN IMMEDIATE` transactions prevent double-write races at the row level but are not suitable for high-fan-out multi-process deployments. Migrate to PostgreSQL for production scale.
- **No end-user authentication:** This is a local demo with synthetic fixtures. The API accepts a caller-supplied `customer_id`. Order ownership checks do not authenticate a real customer; do not connect real accounts or data.
- **Stdio MCP on Windows:** Each MCP server starts as a child process over `stdio`. On Windows, asyncio subprocess handling has additional overhead; the persistent worker queue in `MCPToolClient` mitigates this but does not eliminate it.
- **`aiosqlite` version pin:** `langgraph-checkpoint-sqlite` requires `aiosqlite < 0.22` (pinned to `0.21.x`). Upgrading LangGraph may lift this constraint.
- **In-memory TF-IDF:** `HybridMemoryRetriever` builds a TF-IDF matrix on the fly per query over all episodes for the customer. For customers with thousands of episodes, switch to a vector DB (Chroma, pgvector).
- **Mock LLM:** The offline mock returns deterministic canned responses; it does not test semantic model understanding. All policy, memory, and tool execution tests are deterministic regardless of LLM.

---

## What I'd Do Next

1. **Persistent vector store** — Replace in-memory TF-IDF with Chroma or pgvector for scalable episode retrieval.
2. **PostgreSQL backend** — Migrate from SQLite to PostgreSQL for multi-process deployment and HA.
3. **Streaming responses** — Add Server-Sent Events from FastAPI for real-time token streaming.
4. **Human-in-the-loop dashboard** — A separate operator view that shows pending escalation tickets and lets a human agent inject a resolution back into the conversation thread.
5. **Webhook delivery** — Push refund confirmations and ticket updates to customer email/SMS via a lightweight notification sidecar.
6. **Multi-language support** — Route through a lightweight detection step and pass language hint to the system prompt.
7. **LangSmith / OpenTelemetry tracing** — Export JSONL traces to LangSmith or an OTEL collector for production observability.
8. **Canary / shadow evals in CI** — Run the 20-scenario suite against the shadow deployment before promoting to production.
