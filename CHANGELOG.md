# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] - 2026-05-17

Initial public release.

### Core

- `Scraper` class — the public API. Accepts a URL, a natural-language task,
  and a schema. On first `run()`, an AI agent generates a Playwright scraping
  script. On every subsequent `run()`, the saved script is loaded from disk
  with no AI involved and no API key required.
- `ScraperResult` dataclass with `data`, `url`, `timestamp`,
  `script_generated`, `script_path`, and `auto_regenerated` fields.
- Sync (`run`, `regenerate`) and async (`async_run`, `async_regenerate`)
  execution paths.
- `Scraper.export()` to save a standalone copy of the generated script.
- `Scraper.has_script` property to check if a cached script exists.
- Browser session sharing via context manager (`with scraper:` /
  `async with scraper:`) for running the same script across multiple URLs
  without restarting the browser.

### Agent

- Multi-turn agentic loop that explores the page, reasons about the DOM,
  and writes a complete `async def scrape(page, start_url, checkpoint)`
  function.
- **Exploration planner** — decomposes the user's task into investigation
  steps before the agent touches the page.
- **Three-level page representation** — raw HTML is sanitized, converted to
  a compact text format (preserving interactive elements as HTML tags), and
  split into addressable sections. The agent sees a structured page view and
  can zoom into specific sections.
- **Show-page context management** — page similarity detection decides
  between full re-analysis (new page) and incremental updates (same page,
  new state). Old page views and zoom results are automatically stubbed
  after a fixed number of turns to maximize prompt-cache hit rates.
- **History compression** — when the conversation approaches the model's
  context limit, older exploration messages are compressed into a dense
  sequential summary via a separate LLM call. Previous summaries and the
  first user message are never re-compressed.
- **Checkpoint system** — the generated script can call `checkpoint(label,
  data_preview)` during execution. On rejection, the agent can expand any
  checkpoint to inspect what the page looked like at that point.
- **AI reviewer** — a separate validation agent inspects script output and
  decides whether it satisfies the original task, providing structured
  feedback on rejection.
- **Timeout prediction** — static AST analysis of the generated script
  predicts execution time from Playwright API calls, loops, and wait
  patterns. Two independent signals (AST scorer and await counter) are
  combined to set subprocess timeouts.
- **Selector extraction** — three-stage pipeline parses the generated code
  to extract DOM selectors, classify actions, and detect loop/navigation
  boundaries.
- **Interactive element detection** — browser-side JavaScript stamps every
  interactive element with `data-iid` attributes and hidden subtree roots
  with `data-hidden`, enabling the text converter to preserve actionable
  elements while stripping everything else.
- **Demo overlay** — when `headless=False`, opens a separate browser window
  that visualizes agent actions (thinking, tool calls, results) in real time.
- Tools: `python` (execute generated script), `last_resort_antibot_escape`
  (detect and report blocking).

### Schema

- `Field` class with constraints: `min`, `max`, `min_length`, `max_length`,
  `enum`, `optional`, and `description`.
- `Items` wrapper (aliased as `List`) for list-of-objects schemas with
  `min` and `max` item count constraints.
- Nested schema support — dicts, lists, and `Items` can be arbitrarily
  nested.
- Schema compiler that transforms Python-native schema definitions into
  an internal node tree for validation and prompt generation.
- Grouped error feedback during generation — validation failures are
  deduplicated and grouped by path pattern, so the agent sees
  `"[*].price — missing required field (4 of 25 items)"` instead of 25
  individual errors.
- `Tolerance` levels (`strict`, `balanced`, `lenient`) controlling how
  strictly results must match the schema.

### Auto-regeneration

- `auto_regenerate` parameter with configurable modes: `cautious`,
  `balanced`, `eager`, and `always`.
- `RegenerateMode` enum for type-safe mode selection.
- **Diagnosis loop** — runs the cached script up to 3 times, collecting
  error fingerprints and page-level signals on each failure before deciding
  whether to regenerate.
- **Error classification** — categorizes failures into 7 groups (syntax,
  runtime, network, timeout, DOM interaction, browser crash, schema
  violation) with category-specific regeneration rules.
- **Error fingerprinting** — extracts structured fingerprints (category,
  error type, method, target) for cross-attempt stability comparison.
- **Stability assessment** — compares fingerprints across attempts at three
  specificity levels to distinguish consistent failures (stale script) from
  intermittent ones (environmental).
- **Anti-bot detection** — identifies Cloudflare, Akamai, DataDome,
  Imperva, Kasada, and PerimeterX blocking pages from HTML content and
  HTTP headers. Raises immediately instead of wasting a regeneration
  attempt.
- **Non-content page detection** — identifies login walls, maintenance
  pages, rate limiting, and geographic restrictions on HTTP 200 responses.
- `protect_manual_edits` option to prevent regeneration of manually edited
  scripts.

### Browser

- Stealth defaults via Patchright (real Chrome channel, anti-fingerprint
  flags, US locale/timezone).
- `LaunchOptions` TypedDict for typed browser configuration — proxy,
  locale, timezone, viewport, storage state, geolocation, and any
  Playwright `launch_persistent_context()` option.
- `headless` parameter (default `True`). Set to `False` to watch the
  browser work.

### Other

- Multi-model support via Pydantic AI — Anthropic (default), OpenAI,
  Google, Groq, Mistral, DeepSeek, and any provider Pydantic AI supports.
  Uses `provider:model` format.
- `Input` class for dynamic URL parameters — same script, different search
  queries, locations, or filters on each run.
- RestrictedPython sandbox (`sandboxed=True`) for safe execution of
  AI-generated code.
- CLI (`python -m scout`) for one-shot script generation.
- Docker support with Xvfb for headful mode in containers.
- `run_timeout` and `generation_timeout` parameters for controlling
  execution and generation time limits.
