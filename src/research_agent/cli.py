"""Command line interface: `research-agent "your question"`."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from research_agent.agent import Agent, AgentResult
from research_agent.llm import (
    DEFAULT_MODEL,
    LLM,
    AnthropicLLM,
    LLMResponse,
    ScriptedLLM,
    ToolCall,
    Usage,
)
from research_agent.report import to_markdown
from research_agent.tools import Notes, default_registry
from research_agent.tracing import Tracer


def _demo_llm() -> ScriptedLLM:
    """A scripted run that shows parallel tools, notes and citation checking, fully offline."""
    return ScriptedLLM(
        [
            LLMResponse(
                text="Two independent calculations, so I will run them in parallel.",
                tool_calls=[
                    ToolCall("demo-1", "calculator", {"expression": "1000 * 1.07 ** 10"}),
                    ToolCall("demo-2", "calculator", {"expression": "1000 * 1.05 ** 10"}),
                ],
                usage=Usage(420, 60),
            ),
            LLMResponse(
                tool_calls=[
                    ToolCall(
                        "demo-3",
                        "save_note",
                        {"text": "10 years: 7% -> 1967.15, 5% -> 1628.89"},
                    )
                ],
                usage=Usage(530, 40),
            ),
            LLMResponse(
                text=(
                    "Answer: over 10 years, 1000 grows to about 1967.15 at 7% and about "
                    "1628.89 at 5%, so the extra 2% is worth roughly 338.\n\n"
                    "Sources: calculator tool (demo mode). "
                    "https://example.com/made-up-source"
                ),
                usage=Usage(610, 90),
            ),
        ]
    )


def _price(name: str) -> float | None:
    value = os.environ.get(name)
    try:
        return float(value) if value else None
    except ValueError:
        return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="research-agent",
        description="AI research agent with a tool-calling loop.",
    )
    parser.add_argument("question", nargs="?", help="The question to research")
    parser.add_argument("--max-steps", type=int, default=8, help="Maximum model turns (default 8)")
    parser.add_argument("--parallel", type=int, default=4, help="Max tools run at once (default 4)")
    parser.add_argument("--model", default=None, help=f"Model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--out", type=Path, default=None, help="Save a Markdown report here")
    parser.add_argument(
        "--trace", type=Path, default=None, help="Write a JSONL trace (default: runs/<time>.jsonl)"
    )
    parser.add_argument("--no-trace", action="store_true", help="Do not write a trace file")
    parser.add_argument("--no-cache", action="store_true", help="Disable the tool result cache")
    parser.add_argument("--demo", action="store_true", help="Run offline with a scripted model")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show each tool call")
    return parser


def _print_summary(result: AgentResult, cost: float | None, trace: Path | None) -> None:
    cached = sum(1 for r in result.tool_runs if r.cached)
    failed = sum(1 for r in result.tool_runs if not r.ok)
    parts = [
        f"{result.steps} steps",
        f"{len(result.tool_runs)} tool calls ({cached} cached, {failed} failed)",
        f"{result.usage.input_tokens + result.usage.output_tokens} tokens",
        f"{result.seconds:.1f}s",
    ]
    if cost is not None:
        parts.append(f"${cost:.4f}")
    print("\n[" + ", ".join(parts) + "]")

    c = result.citations
    if c.score is not None:
        print(f"[citations verified: {len(c.grounded)} of {len(c.cited)}]")
        for url in c.ungrounded:
            print(f"  WARNING: not seen in any tool result: {url}")
    if trace:
        print(f"[trace: {trace}]")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose or args.demo else logging.WARNING,
        format="%(message)s",
    )

    llm: LLM
    if args.demo:
        question = args.question or "How much more does 7% earn than 5% on 1000 over 10 years?"
        llm = _demo_llm()
        model = "demo (scripted)"
    else:
        if not args.question:
            print(
                "Give a question, or use --demo. Example:\n"
                '  research-agent "What are the main differences between MQTT and HTTP?"'
            )
            return 2
        question = args.question
        try:
            anthropic_llm = AnthropicLLM(model=args.model)
        except RuntimeError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        llm, model = anthropic_llm, anthropic_llm.model

    trace_path: Path | None = None
    if not args.no_trace and not args.demo:
        trace_path = args.trace or Path("runs") / f"{datetime.now():%Y%m%d-%H%M%S}.jsonl"
    elif args.trace:
        trace_path = args.trace

    agent = Agent(
        llm,
        default_registry(Notes(), cache=not args.no_cache),
        max_steps=args.max_steps,
        max_parallel=args.parallel,
        tracer=Tracer(trace_path),
    )
    result = agent.run(question)

    # Prices are per million tokens, set by you, e.g. RESEARCH_AGENT_PRICE_IN=3
    cost = result.usage.cost(_price("RESEARCH_AGENT_PRICE_IN"), _price("RESEARCH_AGENT_PRICE_OUT"))

    print(result.answer)
    _print_summary(result, cost, trace_path)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(to_markdown(question, result, model, cost), encoding="utf-8")
        print(f"[report: {args.out}]")

    return 0 if result.finished else 3


if __name__ == "__main__":
    raise SystemExit(main())
