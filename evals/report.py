"""Generate Markdown before/after comparison table between baseline and improved modes."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from evals.runner import EvalSummary, run_eval_suite


def _load_summary(path: str | Path) -> EvalSummary:
    """Load an EvalSummary from a JSON file produced by runner.py --output."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return EvalSummary.model_validate(data)


def format_comparison_markdown(baseline: EvalSummary, improved: EvalSummary) -> str:
    """Render a Markdown comparison report."""
    lines = [
        "# Evaluation Report: Baseline vs. Improved Architecture",
        "",
        f"Evaluated across **{improved.total_scenarios} deterministic scenarios** (`evals/dataset.json`).",
        "",
        "| Metric | Baseline (Sliding Window, Prompt-Only) | Improved (5-Layer Memory + MCP Policy Gate) |",
        "|---|---:|---:|",
        f"| Task Success Rate | {baseline.task_success_rate * 100:.1f}% ({baseline.passed_scenarios}/{baseline.total_scenarios}) | {improved.task_success_rate * 100:.1f}% ({improved.passed_scenarios}/{improved.total_scenarios}) |",
        f"| Memory Recall Accuracy | {baseline.memory_recall_accuracy * 100:.1f}% | {improved.memory_recall_accuracy * 100:.1f}% |",
        f"| Policy Violation Rate | {baseline.policy_violation_rate * 100:.1f}% | {improved.policy_violation_rate * 100:.1f}% |",
        f"| Duplicate Action Rate | {baseline.duplicate_action_rate * 100:.1f}% | {improved.duplicate_action_rate * 100:.1f}% |",
        f"| Avg Tokens / Turn | {baseline.avg_tokens_per_turn:.1f} | {improved.avg_tokens_per_turn:.1f} |",
        f"| p95 Turn Latency (ms) | {baseline.p95_latency_ms:.1f} | {improved.p95_latency_ms:.1f} |",
        "",
        "## Per-Scenario Breakdown",
        "",
        "| Scenario ID | Category | Baseline | Improved |",
        "|---|---|---|---|",
    ]
    for b_res, i_res in zip(baseline.results, improved.results, strict=True):
        b_flag = "✅ PASS" if b_res.passed else "❌ FAIL"
        i_flag = "✅ PASS" if i_res.passed else "❌ FAIL"
        lines.append(f"| `{i_res.scenario_id}` | {i_res.category} | {b_flag} | {i_flag} |")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "> Policy violation rate **must** be 0.0 % in improved mode. "
        "Duplicate action rate **must** be 0.0 % in improved mode. "
        "Minimum task success rate is 90 %."
    )
    lines.append("")
    return "\n".join(lines)


async def _run_and_save_suite(mode: str, out_path: Path) -> EvalSummary:
    """Run the eval suite and write results to *out_path*."""
    summary = await run_eval_suite(mode=mode)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary.model_dump(), indent=2), encoding="utf-8")
    return summary


async def generate_report(
    baseline_path: str | Path | None = None,
    improved_path: str | Path | None = None,
    out_md: str | Path = "data/evals/before_after_report.md",
) -> str:
    """
    Generate a comparison Markdown report.

    If *baseline_path*/*improved_path* are provided they are loaded from disk;
    otherwise the suites are re-run from scratch and saved alongside the report.
    """
    out_md_path = Path(out_md)
    if not out_md_path.is_absolute():
        out_md_path = Path(__file__).resolve().parent.parent / out_md_path
    out_md_path.parent.mkdir(parents=True, exist_ok=True)

    if baseline_path is not None and Path(baseline_path).exists():
        baseline = _load_summary(baseline_path)
    else:
        b_out = out_md_path.parent / "baseline.json"
        baseline = await _run_and_save_suite("baseline", b_out)

    if improved_path is not None and Path(improved_path).exists():
        improved = _load_summary(improved_path)
    else:
        i_out = out_md_path.parent / "improved.json"
        improved = await _run_and_save_suite("improved", i_out)

    md = format_comparison_markdown(baseline, improved)

    out_md_path.write_text(md, encoding="utf-8")
    # Also write a companion JSON with both summaries
    json_out = out_md_path.with_suffix(".json")
    json_out.write_text(
        json.dumps(
            {
                "baseline": baseline.model_dump(exclude={"results"}),
                "improved": improved.model_dump(exclude={"results"}),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Report written to {out_md_path}")
    print(f"JSON metrics written to {json_out}")
    return md


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Markdown comparison report from baseline and improved eval runs."
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        help="Path to baseline JSON produced by runner.py --output (re-runs if not provided).",
    )
    parser.add_argument(
        "--improved",
        type=str,
        default=None,
        help="Path to improved JSON produced by runner.py --output (re-runs if not provided).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data/evals/before_after_report.md",
        help="Output path for the Markdown report (default: data/evals/before_after_report.md).",
    )
    args = parser.parse_args()

    asyncio.run(
        generate_report(
            baseline_path=args.baseline,
            improved_path=args.improved,
            out_md=args.out,
        )
    )


if __name__ == "__main__":
    main()
