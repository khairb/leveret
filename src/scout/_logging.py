"""Scout logging configuration.

All console output goes through ``logging.getLogger("scout")``.
Messages write to stderr so ``python scrape.py > data.json`` produces
clean JSON without ``[scout]`` lines mixed in.

A default handler is installed at import time only if the logger has
no handlers — this respects any configuration the user has already set up.
"""

import logging
import sys

logger = logging.getLogger("scout")


def _setup_default_handler() -> None:
    """Add a stderr handler if none exists. Called at import time."""
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("[scout] %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    # Silence noisy third-party debug logs (anthropic SDK, httpx, httpcore).
    for name in ("anthropic", "httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)


_setup_default_handler()
