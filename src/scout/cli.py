"""CLI entry point for the scraping agent.

Usage::

    python -m scout.cli "https://example.com/products" "Extract all product names and prices"
    python -m scout.cli "https://news.ycombinator.com" "Extract top 10 story titles and URLs" --headless
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from .agent.llm import LLMConfig
from .agent.loop import AgentLoop


def main() -> None:
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Run the AI scraping agent to produce a Patchright script.",
    )
    parser.add_argument("url", help="Target URL to scrape")
    parser.add_argument("task", help="What to extract (natural language)")
    parser.add_argument(
        "--model",
        default="claude-haiku-4-5",
        help="Anthropic model ID (default: claude-haiku-4-5)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run the browser in headless mode",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=60,
        help="Maximum total tool calls (default: 50)",
    )
    parser.add_argument(
        "--max-python-steps",
        type=int,
        default=50,
        help="Maximum code execution steps (default: 30)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Write the final script to this file path",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Anthropic API key (default: ANTHROPIC_API_KEY env var)",
    )
    parser.add_argument(
        "--trace-dir",
        type=str,
        default="./traces",
        help="Directory for trace files (default: ./traces)",
    )
    parser.add_argument(
        "--script-timeout",
        type=int,
        default=600,
        help="Timeout in seconds for the final script execution (default: 600)",
    )
    parser.add_argument(
        "--approve-mode",
        choices=["human", "auto"],
        default="human",
        help="Approval mode: 'human' for manual review, 'auto' for validator agent (default: human)",
    )
    parser.add_argument(
        "--validator-model",
        type=str,
        default="claude-haiku-4-5",
        help="Model for the validation agent (default: claude-haiku-4-5)",
    )
    args = parser.parse_args()

    config = LLMConfig(
        model=args.model,
        api_key=args.api_key,
    )
    validator_config = LLMConfig(
        model=args.validator_model,
        api_key=args.api_key,
    )

    agent = AgentLoop(
        llm_config=config,
        max_steps=args.max_steps,
        max_python_steps=args.max_python_steps,
        headless=args.headless,
        trace_dir=args.trace_dir,
        script_timeout=args.script_timeout,
        approval_mode=args.approve_mode,
        validator_config=validator_config,
        max_script_attempts=6,
    )

    # The loop now handles all live output via console.py.
    result = asyncio.run(agent.run(args.url, args.task))

    if not result.success:
        sys.exit(1)

    # Determine the run directory (created by the tracer).
    run_dir = Path(result.run_dir) if result.run_dir else Path(args.trace_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save the script into the run directory.
    if args.output:
        script_path = Path(args.output)
    else:
        script_path = run_dir / "script.py"

    script_path.write_text(result.final_script, encoding="utf-8")
    print(f"  Script saved to: {script_path}")

    # Script is already executed and approved inside the agent loop.
    # No need to re-run it here.


if __name__ == "__main__":
    main()
