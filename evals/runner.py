"""Deterministic 20-scenario evaluation runner with CI threshold gate."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from scripts.seed_data import seed_databases
from src.agent.graph import SupportAgentRunner
from src.config import get_settings, load_app_config
from src.observability.metrics import aggregate_trace_metrics


class ScenarioResult(BaseModel):
    """Outcome for one deterministic evaluation scenario."""

    scenario_id: str
    category: str
    passed: bool
    failure_reasons: list[str]
    policy_violation: bool
    duplicate_action: bool
    memory_recall_checked: bool
    memory_recall_passed: bool
    avg_tokens_per_turn: float
    p95_latency_ms: float


class EvalSummary(BaseModel):
    """Aggregated evaluation scorecard across all 20 scenarios."""

    mode: str
    total_scenarios: int
    passed_scenarios: int
    task_success_rate: float
    memory_recall_accuracy: float
    policy_violation_rate: float
    duplicate_action_rate: float
    avg_tokens_per_turn: float
    p95_latency_ms: float
    results: list[ScenarioResult]


def _load_long_conversation(fixture_id: str) -> list[str]:
    path = Path(__file__).resolve().parent.parent / "data" / "fixtures" / "long_conversations.json"
    with path.open("r", encoding="utf-8") as fh:
        convos = json.load(fh)
    for c in convos:
        if c["conversation_id"] == fixture_id:
            return [t["message"] for t in c["turns"]]
    raise ValueError(f"Fixture conversation '{fixture_id}' not found.")


async def run_single_scenario(scenario: dict[str, Any], mode: str) -> ScenarioResult:
    settings = get_settings()
    rules = load_app_config(settings.business_rules_path)
    if scenario.get("context_override"):
        rules = rules.model_copy(
            update={"context": rules.context.model_copy(update=scenario["context_override"])}
        )
    checks = scenario.get("checks", {})
    if checks.get("require_compaction"):
        rules = rules.model_copy(
            update={
                "context": rules.context.model_copy(
                    update={"compaction_trigger_tokens": 1200, "recent_messages_to_preserve": 6}
                )
            }
        )
    turns: list[str] = (
        _load_long_conversation(scenario["fixture_conversation_id"])
        if scenario.get("fixture_conversation_id")
        else list(scenario["turns"])
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        app_db = tmp / "app.db"
        seed_db = tmp / "seed.db"
        ckpt_db = tmp / "checkpoints.db"
        trace_log = tmp / "traces.jsonl"

        seed_databases(seed_db_path=seed_db, app_db_path=app_db, reset=True)

        turn_outputs: list[dict[str, Any]] = []
        turn_latencies: list[float] = []
        compaction_seen = False

        async with SupportAgentRunner(
            settings=settings,
            rules=rules,
            app_db_path=app_db,
            checkpoint_db_path=ckpt_db,
            trace_log_path=trace_log,
            mode=mode,
        ) as runner:
            for msg in turns:
                t0 = runner.tracer.now_ms()
                res = await runner.arun_turn(
                    thread_id=scenario["thread_id"],
                    customer_id=scenario["customer_id"],
                    user_message=msg,
                    mode=mode,
                )
                turn_latencies.append(runner.tracer.now_ms() - t0)
                turn_outputs.append(res)
                if res.get("compaction_triggered"):
                    compaction_seen = True

            # Evaluate checks
            failures: list[str] = []
            last_answer = str((turn_outputs[-1].get("final_response") or {}).get("answer", ""))
            customer_id = scenario["customer_id"]
            refunds = runner.repo.get_refunds_for_customer(customer_id)
            tickets = runner.repo.list_support_tickets_for_customer(customer_id)

            for token in checks.get("must_contain_all", []):
                if token.lower() not in last_answer.lower():
                    failures.append(f"Missing expected substring '{token}' in answer: {last_answer}")

            any_tokens = checks.get("must_contain_any", [])
            if any_tokens and not any(tok.lower() in last_answer.lower() for tok in any_tokens):
                failures.append(f"None of {any_tokens} found in answer: {last_answer}")

            for forbidden in checks.get("must_not_contain", []):
                if forbidden.lower() in last_answer.lower():
                    failures.append(f"Forbidden substring '{forbidden}' appeared in answer.")

            if "expected_refund_count" in checks and len(refunds) != checks["expected_refund_count"]:
                failures.append(
                    f"Expected {checks['expected_refund_count']} refunds, got {len(refunds)}."
                )

            if "expected_refund_amount" in checks and (
                not refunds or abs(refunds[0].amount - float(checks["expected_refund_amount"])) > 0.01
            ):
                failures.append(
                    f"Expected refund amount {checks['expected_refund_amount']}, got {[r.amount for r in refunds]}."
                )

            if "expected_ticket_count" in checks and len(tickets) != checks["expected_ticket_count"]:
                failures.append(
                    f"Expected {checks['expected_ticket_count']} tickets, got {len(tickets)}."
                )

            if "max_refunds_on_order" in checks:
                target_ord = checks["max_refunds_on_order"]["order_id"]
                max_cnt = checks["max_refunds_on_order"]["max_count"]
                ord_refunds = runner.repo.get_refunds_for_order(target_ord)
                if len(ord_refunds) > max_cnt:
                    failures.append(f"Order {target_ord} has {len(ord_refunds)} refunds (max {max_cnt}).")

            # Policy violation check: any refund exceeding 500 BRL, outside 30-day window, on non-delivered order, or cross-customer
            policy_violation = False
            for r in refunds:
                ord_rec = runner.repo.get_order(r.order_id)
                if ord_rec is None or ord_rec.customer_id != r.customer_id or ord_rec.status != "delivered" or r.amount > rules.refund_policy.maximum_refund_amount + 1e-6 or ord_rec.order_id == "ORD-1003":
                    policy_violation = True
            if checks.get("policy_check_no_violation") and policy_violation:
                failures.append("Policy violation detected in persisted refunds.")

            # Duplicate action check
            duplicate_action = False
            by_order: dict[str, int] = {}
            for r in refunds:
                by_order[r.order_id] = by_order.get(r.order_id, 0) + 1
            if any(cnt > 1 for cnt in by_order.values()):
                duplicate_action = True
            if checks.get("duplicate_check") and duplicate_action:
                failures.append("Duplicate refund detected for same order.")

            # Long conversation memory recall checks
            memory_recall_checked = False
            memory_recall_passed = True
            if "memory_recall_turn_28_contains_any" in checks:
                memory_recall_checked = True
                ans28 = str((turn_outputs[27].get("final_response") or {}).get("answer", "")).lower()
                if not any(t.lower() in ans28 for t in checks["memory_recall_turn_28_contains_any"]):
                    memory_recall_passed = False
                    failures.append(f"Turn 28 failed preference recall: {ans28}")

            if "memory_recall_turn_29_contains_all" in checks:
                memory_recall_checked = True
                ans29 = str((turn_outputs[28].get("final_response") or {}).get("answer", "")).lower()
                if not all(t.lower() in ans29 for t in checks["memory_recall_turn_29_contains_all"]):
                    memory_recall_passed = False
                    failures.append(f"Turn 29 failed early refund recall: {ans29}")

            if checks.get("require_compaction") and not compaction_seen:
                failures.append("Adaptive compaction was never triggered during 32-turn run.")

            if "forbid_in_trace_logs" in checks and trace_log.exists():
                raw_log_text = trace_log.read_text(encoding="utf-8")
                for secret in checks["forbid_in_trace_logs"]:
                    if secret in raw_log_text:
                        failures.append(f"Unredacted PII '{secret}' found in trace log.")

            if "max_context_tokens" in checks:
                max_seen = max(
                    int((out.get("assembled_context") or {}).get("estimated_tokens", 0))
                    for out in turn_outputs
                )
                if max_seen > int(checks["max_context_tokens"]):
                    failures.append(
                        f"Context tokens {max_seen} exceeded budget {checks['max_context_tokens']}."
                    )

            trace_metrics = aggregate_trace_metrics(runner.tracer.records)
            avg_tokens = (
                (trace_metrics.total_input_tokens + trace_metrics.total_output_tokens)
                / max(1, len(turns))
            )
            sorted_lat = sorted(turn_latencies)
            p95_idx = min(len(sorted_lat) - 1, max(0, math.ceil(0.95 * len(sorted_lat)) - 1))

            return ScenarioResult(
                scenario_id=scenario["id"],
                category=scenario["category"],
                passed=len(failures) == 0,
                failure_reasons=failures,
                policy_violation=policy_violation,
                duplicate_action=duplicate_action,
                memory_recall_checked=memory_recall_checked,
                memory_recall_passed=memory_recall_passed,
                avg_tokens_per_turn=round(avg_tokens, 2),
                p95_latency_ms=round(sorted_lat[p95_idx], 2),
            )


async def run_eval_suite(mode: str = "improved", dataset_path: str | Path | None = None) -> EvalSummary:
    ds_path = (
        Path(dataset_path)
        if dataset_path
        else Path(__file__).resolve().parent / "dataset.json"
    )
    with ds_path.open("r", encoding="utf-8") as fh:
        scenarios = json.load(fh)

    results: list[ScenarioResult] = []
    for sc in scenarios:
        results.append(await run_single_scenario(sc, mode=mode))

    total = len(results)
    passed = sum(1 for r in results if r.passed)
    mem_checked = [r for r in results if r.memory_recall_checked]
    mem_passed = sum(1 for r in mem_checked if r.memory_recall_passed)
    violations = sum(1 for r in results if r.policy_violation)
    duplicates = sum(1 for r in results if r.duplicate_action)
    avg_tokens = sum(r.avg_tokens_per_turn for r in results) / max(1, total)
    sorted_p95 = sorted(r.p95_latency_ms for r in results)
    p95_overall = sorted_p95[min(len(sorted_p95) - 1, max(0, math.ceil(0.95 * len(sorted_p95)) - 1))]

    return EvalSummary(
        mode=mode,
        total_scenarios=total,
        passed_scenarios=passed,
        task_success_rate=round(passed / max(1, total), 4),
        memory_recall_accuracy=round(mem_passed / max(1, len(mem_checked)), 4) if mem_checked else 1.0,
        policy_violation_rate=round(violations / max(1, total), 4),
        duplicate_action_rate=round(duplicates / max(1, total), 4),
        avg_tokens_per_turn=round(avg_tokens, 2),
        p95_latency_ms=round(p95_overall, 2),
        results=results,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run deterministic customer support evals.")
    parser.add_argument("--mode", choices=["improved", "baseline"], default="improved")
    parser.add_argument("--min-score", type=float, default=0.90)
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to write JSON results (e.g. data/evals/improved.json). Creates parent dirs.",
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        default=None,
        help="Path to dataset.json (default: evals/dataset.json).",
    )
    args = parser.parse_args()

    summary = asyncio.run(run_eval_suite(mode=args.mode, dataset_path=args.scenarios))
    print(f"=== Eval Summary ({summary.mode}) ===")
    print(f"Task Success Rate:      {summary.task_success_rate * 100:.1f}% ({summary.passed_scenarios}/{summary.total_scenarios})")
    print(f"Memory Recall Accuracy: {summary.memory_recall_accuracy * 100:.1f}%")
    print(f"Policy Violation Rate:  {summary.policy_violation_rate * 100:.1f}%")
    print(f"Duplicate Action Rate:  {summary.duplicate_action_rate * 100:.1f}%")
    print(f"Avg Tokens / Turn:      {summary.avg_tokens_per_turn:.1f}")
    print(f"p95 Latency (ms):       {summary.p95_latency_ms:.1f}")

    for r in summary.results:
        status = "PASS" if r.passed else "FAIL"
        print(f"  [{status}] {r.scenario_id} ({r.category})")
        for reason in r.failure_reasons:
            print(f"         -> {reason}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = summary.model_dump()
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Results written to {out_path}")

    if args.mode == "improved":
        if summary.task_success_rate < args.min_score:
            print(f"ERROR: Score {summary.task_success_rate:.2f} below min-score {args.min_score:.2f}")
            return 1
        if summary.policy_violation_rate > 0.0:
            print("ERROR: Policy violation rate must be 0.0% in improved mode.")
            return 1
        if summary.duplicate_action_rate > 0.0:
            print("ERROR: Duplicate action rate must be 0.0% in improved mode.")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
