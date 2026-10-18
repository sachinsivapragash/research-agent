"""research-agent: a small, testable AI agent with a tool-calling loop."""

from research_agent.agent import Agent, AgentResult, ToolRun
from research_agent.citations import CitationReport, check_citations
from research_agent.llm import AnthropicLLM, LLMResponse, ScriptedLLM, ToolCall, Usage
from research_agent.tools import Tool, ToolRegistry, default_registry
from research_agent.tracing import Tracer

__all__ = [
    "Agent",
    "AgentResult",
    "AnthropicLLM",
    "CitationReport",
    "LLMResponse",
    "ScriptedLLM",
    "Tool",
    "ToolCall",
    "ToolRegistry",
    "ToolRun",
    "Tracer",
    "Usage",
    "check_citations",
    "default_registry",
]

__version__ = "0.4.0"
