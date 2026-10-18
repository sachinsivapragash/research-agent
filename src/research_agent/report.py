"""Render an agent run as a Markdown report."""

from __future__ import annotations

from datetime import datetime, timezone

from research_agent.agent import AgentResult


def to_markdown(question: str, result: AgentResult, model: str, cost: float | None) -> str:
    when = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    status = "complete" if result.finished else "stopped at step limit"
    lines = [
        f"# {question}",
        "",
        result.answer,
        "",
        "---",
        "",
        "## Run details",
        "",
        "| | |",
        "|---|---|",
        f"| Run ID | `{result.run_id}` |",
        f"| Date | {when} |",
        f"| Model | `{model}` |",
        f"| Status | {status} |",
        f"| Steps | {result.steps} |",
        f"| Tool calls | {len(result.tool_runs)} |",
        f"| Time | {result.seconds:.1f} s |",
        f"| Tokens | {result.usage.input_tokens} in / {result.usage.output_tokens} out |",
    ]
    if cost is not None:
        lines.append(f"| Estimated cost | ${cost:.4f} |")
    c = result.citations
    if c.score is not None:
        lines.append(f"| Citations verified | {len(c.grounded)} of {len(c.cited)} |")
        if c.ungrounded:
            lines += ["", "## Unverified citations", ""]
            lines.append("These URLs appear in the answer but were never returned by a tool:")
            lines.append("")
            lines += [f"- {u}" for u in c.ungrounded]
    if result.tool_runs:
        lines += ["", "## Tool calls", ""]
        lines += ["| # | Tool | Input | Result | Time |", "|---|---|---|---|---|"]
        for i, run in enumerate(result.tool_runs, start=1):
            args = ", ".join(f"{k}={v!r}" for k, v in run.input.items()).replace("|", "\\|")
            outcome = "cached" if run.cached else ("ok" if run.ok else "error")
            lines.append(f"| {i} | `{run.name}` | {args} | {outcome} | {run.seconds:.2f} s |")
    return "\n".join(lines) + "\n"
