# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] - 2026-05-17

### Added

- AI agent that generates Playwright-compatible scraping scripts from natural-language task descriptions.
- Schema validation system (`Field`, `Items`) with constraints (`min`, `max`, `min_length`, `enum`, `optional`) and grouped error feedback during generation.
- Tolerance levels for schema validation strictness.
- Auto-regeneration (`auto_fix`) with configurable modes: `conservative`, `balanced`, `aggressive`, and `always`.
- Anti-bot detection and diagnosis — identifies Cloudflare, Akamai, DataDome, and similar blocking systems and raises immediately instead of wasting regeneration attempts.
- RestrictedPython sandbox (`sandbox=True`) for safe execution of AI-generated code.
- Multi-model support via Pydantic AI — Anthropic (default), OpenAI, Google, Groq, Mistral, DeepSeek, and any other provider Pydantic AI supports. Uses `provider:model` format.
- Browser session sharing via context manager (`with scraper:` / `async with scraper:`), allowing the same script to run across multiple pages without restarting the browser.
- Script persistence — generated scripts are plain Python files saved to disk. Subsequent runs load from disk with no AI involvement and no API key required.
- Dynamic URL parameter detection (`Input`) for parameterized scraping runs.
- `Scraper.export()` to save standalone copies of generated scripts.
- `ScraperResult` with `data`, `url`, `cached`, `auto_fixed`, `timestamp`, and `script_path` fields.
- CLI (`python -m scout`) for one-shot script generation.
- `LaunchOptions` TypedDict for typed browser configuration, forwarded to Patchright's `launch_persistent_context()`.
- Stealth browser defaults via Patchright (real Chrome channel, anti-bot flags, US locale/timezone).
- `protect_script` option to prevent regeneration of manually edited scripts.
- Docker support.
