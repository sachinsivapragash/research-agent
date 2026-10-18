"""Offline demo: a scripted model plus stand-in tools with the same names and schemas.

Used by the web UI's Demo mode so anyone can watch a full run with no API key and no
network. The script is deliberately realistic: parallel calls, a cache hit, page reads,
a note, and a final answer that includes one invented source so the citation check
has something to catch. All demo URLs use the reserved example.org domain on purpose.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from research_agent.llm import LLMResponse, ScriptedLLM, ToolCall, Usage
from research_agent.tools import Tool, ToolRegistry, default_registry

DEMO_QUESTION = "Do electric cars really produce less CO2 than petrol cars over their lifetime?"

LIFECYCLE = "https://example.org/demo/lifecycle-emissions-study"
GRID = "https://example.org/demo/grid-mix-report"
BATTERY = "https://example.org/demo/battery-manufacturing"
WIKI = "https://en.wikipedia.org/wiki/Electric_vehicle"
FAKE = "https://example.org/demo/2031-ev-mega-study"

_SEARCH = f"""1. Lifecycle emissions of electric and petrol cars (demo)
   {LIFECYCLE}
   Compares manufacturing, fuel and electricity, and end-of-life emissions.
2. How the electricity grid mix changes EV emissions (demo)
   {GRID}
   Emissions depend strongly on how the local electricity is generated.
3. Battery manufacturing and its carbon cost (demo)
   {BATTERY}
   Battery production adds emissions before the car is first driven."""

_PAGES = {
    LIFECYCLE: (
        "Demo page. Lifecycle studies count emissions from making the car, producing its "
        "energy, driving it and disposing of it. Electric cars usually start with a higher "
        "manufacturing footprint because of the battery, then make it up over time because "
        "driving them emits less, so their lifetime total is typically lower."
    ),
    GRID: (
        "Demo page. The benefit of an electric car depends on the electricity it uses. On "
        "grids with more renewable or nuclear power the gap is large; on coal-heavy grids "
        "it is much smaller."
    ),
    BATTERY: "Demo page. Battery production is the main extra emission source for EVs.",
}


def _slow(seconds: float, func: Callable[..., str]) -> Callable[..., str]:
    def wrapper(**kwargs: Any) -> str:
        if seconds:
            time.sleep(seconds)
        return func(**kwargs)

    return wrapper


def demo_registry(delay: float = 0.9) -> ToolRegistry:
    """Same tool names and schemas as the real registry, with offline answers."""
    stubs: dict[str, Callable[..., str]] = {
        "web_search": lambda query, max_results=5: _SEARCH,
        "fetch_page": lambda url: _PAGES.get(url, "Demo page not found."),
        "wikipedia_summary": lambda title: (
            "Electric vehicle: a vehicle that uses one or more electric motors for "
            f"propulsion.\nSource: {WIKI}"
        ),
        "github_repo_info": lambda repo: "Not used in the demo.",
        "arxiv_search": lambda query, max_results=3: "Not used in the demo.",
        "weather": lambda city: "Not used in the demo.",
    }
    real = default_registry()
    registry = ToolRegistry()
    for tool in real.tools():
        func = stubs.get(tool.name, tool.func)
        speed = delay if tool.name in stubs else delay * 0.3
        registry.register(
            Tool(
                tool.name,
                tool.description,
                tool.input_schema,
                _slow(speed, func),
                cacheable=tool.cacheable,
                untrusted=tool.untrusted,
            )
        )
    return registry


class _PacedLLM(ScriptedLLM):
    def __init__(self, responses: list[LLMResponse], delay: float) -> None:
        super().__init__(responses)
        self.delay = delay

    def complete(
        self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> LLMResponse:
        if self.delay:
            time.sleep(self.delay)
        return super().complete(system, messages, tools)


def demo_llm(delay: float = 1.1) -> ScriptedLLM:
    search = {"query": "electric vs petrol car lifetime CO2 emissions"}
    return _PacedLLM(
        [
            LLMResponse(
                text="Plan: search for lifecycle studies and get background, in parallel.",
                tool_calls=[
                    ToolCall("d1", "web_search", search),
                    ToolCall("d2", "wikipedia_summary", {"title": "Electric vehicle"}),
                ],
                usage=Usage(1840, 96),
            ),
            LLMResponse(
                text="Reading the two most relevant pages. Repeating the search hits the cache.",
                tool_calls=[
                    ToolCall("d3", "fetch_page", {"url": LIFECYCLE}),
                    ToolCall("d4", "fetch_page", {"url": GRID}),
                    ToolCall("d5", "web_search", search),
                ],
                usage=Usage(2710, 120),
            ),
            LLMResponse(
                tool_calls=[
                    ToolCall(
                        "d6",
                        "save_note",
                        {"text": f"EVs: higher build emissions, lower lifetime CO2 ({LIFECYCLE})"},
                    )
                ],
                usage=Usage(3390, 64),
            ),
            LLMResponse(
                text=(
                    "Usually yes. Over a full lifetime, an electric car typically emits less CO2 "
                    "than a comparable petrol car.\n\n"
                    "- Building an EV emits more at first, mainly because of the battery.\n"
                    "- It makes that up while driving, because running on electricity emits "
                    "less than burning petrol.\n"
                    "- How big the advantage is depends on the local electricity grid: large on "
                    "clean grids, much smaller on coal-heavy ones.\n\n"
                    "Sources:\n"
                    f"- {LIFECYCLE}\n"
                    f"- {GRID}\n"
                    f"- {WIKI}\n"
                    f"- {FAKE}"
                ),
                usage=Usage(3580, 210),
            ),
        ],
        delay,
    )
