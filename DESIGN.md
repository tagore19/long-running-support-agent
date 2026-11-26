# Design

## How a chat works

Each message is sent to the agent with a customer ID and a conversation ID (`thread_id`). Reusing the same IDs lets the agent continue the same conversation.

```text
Message
  → check and prepare the input
  → load the customer's orders and recent conversation
  → decide whether to answer or use a support tool
  → check the request against business rules
  → save the answer and update conversation memory
```

The API and support tools are local to the app. SQLite stores orders, refunds, support tickets, events, and conversation checkpoints.

## Keeping conversation details

SQLite is the source of truth for order and refund details. The conversation history helps the agent understand short follow-ups such as “what about the order I just mentioned?” When a chat gets long, older messages are summarized and recent messages are kept.

This keeps exact order IDs and completed actions available without sending the entire chat to the model on every turn. A regression test covers a conversation that continues after the agent runner restarts.

## Orders, refunds, and escalation

The model can suggest a response or ask to use a tool. Python checks the customer, order, and configured business rules before a tool can change data. Refund eligibility is checked before a refund is created, and repeated requests cannot create duplicate refunds. Delivery disputes can create a support ticket.

The MCP server exposes the support tools as a separate process. This makes their inputs and results explicit and testable.

## Models and measured results

The default mock model is free, deterministic, and works offline. Gemini can be enabled with an API key in the environment. Tests and CI use mock mode; no live Gemini latency or cost has been measured.

The checked-in evaluation report compares the 20 fixed scenarios:

| Result | Baseline | Improved |
|---|---:|---:|
| Scenarios passed | 10/20 | 20/20 |
| Policy violations | 35% | 0% |
| Duplicate actions | 15% | 0% |
| p95 local response time | 150 ms | 284 ms |
| Estimated model cost | $0.00 | $0.00 |

These are local mock-mode measurements. Response times vary by machine, and estimated tokens are not billed model usage.

## Logs and privacy

The app records tool steps, response times, and token estimates in JSONL trace files. PII and unknown fields are masked before the logs are written. Logs and SQLite files are for the local demo and should not contain real customer data.

## Try it

Follow the clone, run, test, and validation steps in [README.md](README.md). The default setup needs no API key; Docker Compose starts the API and its interactive docs at `http://localhost:8000/docs`.
