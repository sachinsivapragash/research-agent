# Contributing

## Set up

```cmd
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
pre-commit install
```

## Before you open a pull request

```cmd
ruff check .
ruff format .
mypy src
pytest
```

## Adding a tool

1. Write a function that takes plain arguments and returns a string. Raise an exception on failure; the agent reports it to the model.
2. Register it as a `Tool` in `src/research_agent/tools.py`. Set `untrusted=True` if its output comes from the internet, and `cacheable=False` if calling it twice can give different results.
3. If it calls a web API, go through `research_agent.http.get_with_retry`, and in tests replace the network call so tests never go online.
4. Add tests.

## Commit messages

Use short prefixes: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`.
