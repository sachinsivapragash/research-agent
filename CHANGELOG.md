# Changelog

## 0.4.0

### Added
- OpenAI-compatible model client, so Live mode also works with free options such as Gemini's free tier or Ollama. New `--provider` flag and `RESEARCH_AGENT_BASE_URL`, `RESEARCH_AGENT_API_KEY` settings.

## 0.3.0

### Added
- 3D evidence lab web UI (`research-agent-ui`): live tool calls, evidence sphere, citation verdict, custom cursor and transitions. No JavaScript libraries.
- Server-Sent Events stream of agent events from a standard-library web server.
- `tool_start` events, source URLs on `tool_result` and full citation lists on `run_end`.
- Static GitHub Pages build (`scripts/build_pages.py`): the UI replays a recorded run, so it works with no server.
- Offline demo with stand-in tools, a cache hit and one deliberately invented citation.

## 0.2.0

### Added
- Citation grounding check: every URL in the answer is compared with what the tools returned, and invented links are flagged.
- Parallel tool execution, with results kept in the original order.
- Tool result cache, so repeated identical calls are not run twice.
- HTTP retries with exponential backoff and `Retry-After` support.
- Prompt-injection guard: web content is wrapped as untrusted before the model sees it.
- Tool argument validation (missing or unknown arguments are rejected).
- JSONL run tracing with timings and token counts.
- Token usage and optional cost estimate.
- Markdown report export with `--out`.
- Dockerfile, type checking, pre-commit hooks, Dependabot, issue and PR templates.

## 0.1.0

### Added
- Tool-calling agent loop with step limit and error handling.
- Tools: web search, page reader, calculator, notes.
- Integrations: Wikipedia, GitHub, arXiv, Open-Meteo weather.
- Offline demo mode, unit tests and CI.
