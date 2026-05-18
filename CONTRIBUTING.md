# Contributing to Scout

Thanks for your interest in contributing to Scout! This guide will get you up and running.

## Development Setup

1. **Clone the repo:**

   ```bash
   git clone https://github.com/anthropics/scout.git
   cd scout
   ```

2. **Install [uv](https://docs.astral.sh/uv/getting-started/installation/) (if you don't have it):**

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. **Install dependencies:**

   ```bash
   uv sync
   ```

4. **Install browser:**

   ```bash
   patchright install chromium
   ```

## Running Tests

```bash
uv run pytest tests/ -x -q
```

> **Note:** Tests in `tests/integration/` require API keys and are skipped in CI.
> You can run unit tests without any credentials.

## Code Style

We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting. Before submitting a PR, run:

```bash
uv run ruff check .
uv run ruff format .
```

## Submitting Changes

1. Fork the repo and create a branch from `main`.
2. Make your changes.
3. Ensure tests pass and ruff is clean.
4. Submit a pull request with a clear description of what you changed and why.

Keep PRs focused -- one feature or fix per PR is ideal.

## What to Work On

- Check out issues labeled [**good first issue**](https://github.com/anthropics/scout/labels/good%20first%20issue) for a place to start.
- The most valuable contribution is **trying Scout on real websites** and opening issues when something breaks. Include the URL, the task you tried, and what happened. This helps us find edge cases no test suite can cover.

## Bug Reports

When filing a bug, please include:

- **URL** you were scraping
- **Task description** (what you asked Scout to do)
- **Schema** (if you used one)
- **Error output** or unexpected behavior
- **Scout version** (`uv pip show scout`)

The more detail you provide, the faster we can fix it.

## Questions?

Open a [discussion](https://github.com/anthropics/scout/discussions) or ask in an issue. We're happy to help.
