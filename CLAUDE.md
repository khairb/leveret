# CLAUDE.md

## Project overview

Scout is an AI agent that generates standalone Playwright (Patchright) web scraping scripts. Generate once, run forever without AI.

## Architecture

- `src/scout/scraper.py` -- Main `Scraper` class and `ScraperResult`. Public API: `run()`, `async_run()`, `regenerate()`, `export()`. Handles script caching, validation, and auto-fix orchestration.
- `src/scout/agent/` -- AI agent: LLM orchestration loop, prompt engineering, tool definitions (`python`, `last_resort_antibot_escape`), planning, show-page context management, and the Patchright API guide.
- `src/scout/autofix/` -- Auto-regeneration system. Diagnoses failures (stale selectors vs. anti-bot vs. site down), decides whether to regenerate, and detects bot-blocking pages.
- `src/scout/page/` -- HTML processing pipeline: sanitization (unstable CSS class removal, structural cleanup, repeating element truncation), HTML-to-text conversion (preserving interactive elements), sectioning (semantic blocks with IDs and roles), and interactive element detection (5-layer browser-side strategy).
- `src/scout/schema/` -- Schema system: parsing user-defined schemas (`Field`, `Items`), compilation, validation against scraped data, tolerance levels.
- `src/scout/runtime/` -- Script execution: stateful Python REPL, browser lifecycle, execution history, sandbox (RestrictedPython).
- `src/scout/browser.py` -- Browser management via Patchright. Stealth defaults, `LaunchOptions` TypedDict.
- `src/scout/errors.py` -- Error hierarchy: `Error` base, `GenerationError`, `ScriptRuntimeError`, `ValidationError`, `AutoRegenerateError`, `SandboxViolationError`, etc.
- `src/scout/inputs.py` -- Dynamic input handling for parameterized scraper runs.

## Development commands

```bash
uv sync                                                # install deps
uv run pytest tests/ -x -q --ignore=tests/integration  # run tests
uv run ruff check . && uv run ruff format .            # lint and format
```

## Key design decisions

- Generated scripts are plain Python functions with no Scout imports -- they work standalone. You can uninstall Scout and your scrapers keep working.
- Schema validation happens at the boundary (inside `Scraper.run()`), not inside generated code.
- Auto-regeneration diagnoses failures before deciding to regenerate. It distinguishes stale selectors from anti-bot blocks, site outages, and intermittent errors -- not a blind retry.
- RestrictedPython sandbox is opt-in via `sandboxed=True`.
- The page representation uses three levels: text view (overview), sections (semantic navigation), and zoom (full sanitized HTML for writing selectors).
- Unstable CSS-in-JS class names are stripped before the agent sees HTML, preventing reliance on fragile selectors.
- Patchright (patched Playwright fork) is used over vanilla Playwright for better anti-bot stealth.

## Code style

- Formatter/linter: ruff
- Line length: 100
- Python: 3.10+
- LLM framework: Pydantic AI (supports Anthropic, OpenAI, Google, Groq, Mistral, DeepSeek)
