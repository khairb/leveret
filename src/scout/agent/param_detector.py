"""URL Parameter Detector — detects when form fill values appear as URL query parameters.

Pure-function module.  No browser/runtime dependencies.

Pipeline:

  1. ``extract_fill_values(code)``     → values the AI filled into forms
  2. ``detect_url_params(before, after, fills)`` → matched params or None
  3. ``format_hint(result)``            → hint string for the AI

Usage::

    from scout.agent.param_detector import (
        extract_fill_values, detect_url_params, format_hint,
    )

    fills = extract_fill_values(code_string)
    result = detect_url_params(url_before, url_after, fills)
    if result:
        hint = format_hint(result)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from urllib.parse import parse_qs, unquote_plus, urlparse

logger = logging.getLogger("scout")

# ═══════════════════════════════════════════════════════════════════════
#  Output models
# ═══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class FillValue:
    """A value the AI filled into a form element."""

    action: str  # fill, type, select_option, press_sequentially, keyboard.type
    value: str   # the literal string value
    line: int    # 1-based line number in the code


@dataclass(frozen=True)
class ParamMatch:
    """A URL query parameter whose value matches a form fill."""

    param_name: str   # e.g. "ss"
    param_value: str  # raw value from the URL, e.g. "Berlin"
    fill: FillValue   # the FillValue it matched


@dataclass(frozen=True)
class DetectionResult:
    """Detection result — clean param matches and/or complex-encoded params."""

    url: str
    matches: tuple[ParamMatch, ...]       # Tier 1: clean key=value matches
    all_new_params: dict[str, list[str]]   # clean params only
    complex_params: dict[str, list[str]] = None  # Tier 2: JSON/encoded params


# ═══════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════


def _pos_to_line(code: str, pos: int) -> int:
    """Convert a character position to a 1-based line number."""
    return code[:pos].count("\n") + 1


def _is_comment_line(code: str, pos: int) -> bool:
    """Check if the match at *pos* sits on a ``#``-commented line."""
    line_start = code.rfind("\n", 0, pos) + 1
    prefix = code[line_start:pos].lstrip()
    return prefix.startswith("#")


# ═══════════════════════════════════════════════════════════════════════
#  Stage 1 — Extract fill values from AI-generated Python code
# ═══════════════════════════════════════════════════════════════════════

# --- Methods whose second argument is a user-typed value ---------------
_DIRECT_METHODS: set[str] = {"fill", "type", "select_option"}

_DIRECT_ALT = "|".join(sorted(_DIRECT_METHODS, key=len, reverse=True))

# page.fill("selector", "value")  —  captures value as groups 2/3
_RE_DIRECT_FILL = re.compile(
    rf"page\.({_DIRECT_ALT})\(\s*"
    rf"(?:'[^']*'|\"[^\"]*\")"        # first arg  (selector — skip)
    rf"\s*,\s*"                        # comma
    rf"(?:'([^']*)'|\"([^\"]*)\")",    # second arg (value  — capture)
)

# --- Chain / locator / variable terminal calls -------------------------
_CHAIN_METHODS: set[str] = {"fill", "type", "select_option", "press_sequentially"}

_CHAIN_ALT = "|".join(sorted(_CHAIN_METHODS, key=len, reverse=True))

# .fill("value")  on a locator, variable, or chained expression.
# The value is the *first* (and only) string argument.
_RE_CHAIN_FILL = re.compile(
    rf"\.({_CHAIN_ALT})\(\s*(?:'([^']*)'|\"([^\"]*)\")",
)

# --- select_option with keyword argument: .select_option(value="x") ---
_RE_SELECT_KW = re.compile(
    r"\.select_option\(\s*(?:value|label)\s*=\s*(?:'([^']*)'|\"([^\"]*)\")",
)

# --- page.keyboard.type("value") — global keyboard input --------------
_RE_KEYBOARD_TYPE = re.compile(
    r"page\.keyboard\.type\(\s*(?:'([^']*)'|\"([^\"]*)\")",
)

# --- page.keyboard.press("KeyA") — skip single-key presses, but ------
#     catch press_sequentially which types text character by character.
#     (Already handled by _RE_CHAIN_FILL for locator.press_sequentially)

# --- inputs["key"] / inputs['key'] references as fill values ----------
# Matches:  page.fill("sel", inputs["key"])
#           page.fill("sel", inputs['key'])
#           locator.fill(inputs["key"])
#           page.fill("sel", str(inputs["key"]))
_FILL_ACTIONS_ALT = "|".join(sorted(
    _DIRECT_METHODS | _CHAIN_METHODS, key=len, reverse=True,
))
_RE_INPUTS_REF = re.compile(
    rf"\.({_FILL_ACTIONS_ALT})\("
    rf"[^)]*?"                             # anything before the inputs ref
    rf"(?:str\()?\s*inputs\[(?:'([^']*)'|\"([^\"]*)\")\]",
)


def extract_fill_values(
    code: str,
    inputs: dict[str, object] | None = None,
) -> list[FillValue]:
    """Extract all literal string values the AI filled into form elements.

    Handles:
        - ``page.fill("sel", "val")``
        - ``page.type("sel", "val")``
        - ``page.select_option("sel", "val")``
        - ``page.locator("sel").fill("val")``
        - ``page.get_by_*(...).fill("val")``
        - ``var.fill("val")`` (variable holding a locator)
        - ``page.locator("sel").select_option(value="val")``
        - ``page.locator("sel").select_option(label="val")``
        - ``page.keyboard.type("val")``
        - ``page.locator("sel").press_sequentially("val")``

    Skips f-strings and variable arguments (no false positives from
    values that can't be resolved statically).

    Returns a list of :class:`FillValue`, sorted by line number.
    """
    results: list[FillValue] = []
    # Track match spans from direct page calls so we can skip them
    # in the chain pass (avoids extracting the selector as a value).
    direct_spans: list[tuple[int, int]] = []

    # ── Pass 1: page.fill/type/select_option("sel", "val") ───────
    for m in _RE_DIRECT_FILL.finditer(code):
        if _is_comment_line(code, m.start()):
            continue
        value = m.group(2) or m.group(3)
        if not value:
            continue
        results.append(FillValue(
            action=m.group(1),
            value=value,
            line=_pos_to_line(code, m.start()),
        ))
        direct_spans.append((m.start(), m.end()))

    # ── Pass 2: .fill/type/select_option/press_sequentially("val")
    for m in _RE_CHAIN_FILL.finditer(code):
        if _is_comment_line(code, m.start()):
            continue
        # Skip if this match overlaps with a direct page call
        # (for direct calls the first string arg is the selector,
        # not the value — Pass 1 already extracted the real value).
        if any(s <= m.start() < e for s, e in direct_spans):
            continue
        # Skip if preceded by "page" — this is a page.METHOD() call
        # where the second arg was a variable/f-string (Pass 1
        # didn't match, but the first arg is the selector, not a
        # fill value).
        dot_pos = m.start()
        if dot_pos >= 4 and code[dot_pos - 4:dot_pos] == "page":
            continue
        # Skip if preceded by "keyboard" — handled by Pass 4.
        if dot_pos >= 8 and code[dot_pos - 8:dot_pos] == "keyboard":
            continue
        value = m.group(2) or m.group(3)
        if not value:
            continue
        results.append(FillValue(
            action=m.group(1),
            value=value,
            line=_pos_to_line(code, m.start()),
        ))

    # ── Pass 3: .select_option(value="x") / .select_option(label="x")
    for m in _RE_SELECT_KW.finditer(code):
        if _is_comment_line(code, m.start()):
            continue
        value = m.group(1) or m.group(2)
        if not value:
            continue
        # Dedup: if Pass 2 already captured the same value on the
        # same line, skip (shouldn't happen — KW args don't match
        # the positional regex, but guard just in case).
        line = _pos_to_line(code, m.start())
        if any(r.value == value and r.line == line for r in results):
            continue
        results.append(FillValue(
            action="select_option",
            value=value,
            line=line,
        ))

    # ── Pass 4: page.keyboard.type("val") ────────────────────────
    for m in _RE_KEYBOARD_TYPE.finditer(code):
        if _is_comment_line(code, m.start()):
            continue
        value = m.group(1) or m.group(2)
        if not value:
            continue
        results.append(FillValue(
            action="keyboard.type",
            value=value,
            line=_pos_to_line(code, m.start()),
        ))

    # ── Pass 5: inputs["key"] references (resolved dynamically) ──
    # When the AI uses inputs["destination"] instead of a literal
    # "Berlin", we resolve the key against the provided inputs dict.
    if inputs:
        for m in _RE_INPUTS_REF.finditer(code):
            if _is_comment_line(code, m.start()):
                continue
            action = m.group(1)
            key = m.group(2) or m.group(3)
            if not key or key not in inputs:
                continue
            resolved = str(inputs[key])
            if not resolved:
                continue
            line = _pos_to_line(code, m.start())
            # Dedup: skip if a literal extraction already got this
            if any(r.value == resolved and r.line == line for r in results):
                continue
            results.append(FillValue(
                action=action,
                value=resolved,
                line=line,
            ))
            logger.info(
                "[param_detector]   resolved inputs[\"%s\"] → \"%s\" (line %d)",
                key, resolved, line,
            )

    results.sort(key=lambda r: r.line)
    if results:
        logger.info(
            "[param_detector] extract_fill_values: %d fill(s) found", len(results),
        )
        for fv in results:
            logger.info(
                "[param_detector]   line %d: %s(\"%s\")", fv.line, fv.action, fv.value,
            )
    return results


# ═══════════════════════════════════════════════════════════════════════
#  Stage 2 — Detect URL parameters that match fill values
# ═══════════════════════════════════════════════════════════════════════

# Parameters that are tracking / session / cache-buster noise.
_NOISE_PARAMS: set[str] = {
    # Analytics / ad tracking
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "fbclid", "gclid", "gclsrc", "dclid", "msclkid", "ttclid",
    # Referral / attribution
    "ref", "referer", "referrer", "source",
    # Session / auth tokens
    "sid", "session", "session_id", "sessionid", "ssid",
    "token", "csrf", "csrf_token", "csrftoken", "nonce",
    "auth", "auth_token",
    # Cache busters / timestamps
    "_", "timestamp", "ts", "cb", "t", "v", "ver",
}

_NOISE_PREFIXES: tuple[str, ...] = ("utm_", "fb_", "_ga", "__", "pk_")


def _classify_param(name: str, values: list[str]) -> str:
    """Classify a URL parameter as ``"clean"``, ``"complex"``, or ``"noise"``.

    - ``"clean"``   — normal key=value, suitable for Tier 1 matching.
    - ``"complex"`` — JSON blob, base64, or other structured encoding
                      (useful for Tier 2 observational hint).
    - ``"noise"``   — tracking / session junk (discard entirely).
    """
    lower = name.lower()
    if lower in _NOISE_PARAMS:
        return "noise"
    if any(lower.startswith(p) for p in _NOISE_PREFIXES):
        return "noise"
    for v in values:
        decoded = unquote_plus(v).strip()
        # JSON-encoded blobs (e.g. Zillow's searchQueryState).
        if decoded.startswith(("{", "[")) and len(decoded) > 50:
            return "complex"
        # High-entropy random strings (session tokens, nonces).
        if len(v) > 24:
            digits = sum(c.isdigit() for c in v)
            letters = sum(c.isalpha() for c in v)
            if digits and letters and (digits + letters) / len(v) > 0.85:
                return "noise"
    return "clean"


def _parse_query_params(url: str) -> dict[str, list[str]]:
    """Parse query parameters from both the query string and fragment.

    Handles SPA-style fragment routing (e.g. ``/#/search?q=Berlin``).
    """
    parsed = urlparse(url)
    params = parse_qs(parsed.query, keep_blank_values=False)

    # SPA fragment routing: /#/search?q=Berlin
    if "?" in (parsed.fragment or ""):
        frag_query = parsed.fragment.split("?", 1)[1]
        for k, v in parse_qs(frag_query, keep_blank_values=False).items():
            params.setdefault(k, []).extend(v)

    return params


def detect_url_params(
    url_before: str | None,
    url_after: str | None,
    fill_values: list[FillValue],
    inputs: dict[str, object] | None = None,
) -> DetectionResult | None:
    """Detect URL query parameters that match form fill values.

    Matching strategy (applied in order of confidence):
      1. **Exact match** — fill value equals URL param value.
      2. **Contains match** — fill value (≥4 chars) is a substring of
         the URL param value (handles autocomplete expansion, e.g.
         "Barcelona, Spain" → "Barcelona, Catalonia (...), Spain").
      3. **Inputs fallback** — if an ``inputs`` dict is provided, its
         values are also checked against URL params.  This catches
         values the AI set via date pickers or other non-fill UI
         (the value never appeared in a ``fill()`` call).

    Args:
        url_before: Page URL before code execution (may be None).
        url_after:  Page URL after code execution (may be None).
        fill_values: Output of :func:`extract_fill_values`.
        inputs: Optional raw inputs dict (e.g. from ``runtime.repl``).

    Returns:
        A :class:`DetectionResult` if at least one fill value matched
        a new/changed query parameter, otherwise ``None``.
    """
    if not url_after or (not fill_values and not inputs):
        logger.info("[param_detector] detect_url_params: skipped (no url_after or no fills)")
        return None

    params_after = _parse_query_params(url_after)
    if not params_after:
        logger.info("[param_detector] detect_url_params: no query params in url_after")
        return None

    params_before = _parse_query_params(url_before) if url_before else {}

    # Identify new or changed parameters.
    new_params: dict[str, list[str]] = {}
    for key, values in params_after.items():
        if key not in params_before or params_before[key] != values:
            new_params[key] = values

    if not new_params:
        logger.info("[param_detector] detect_url_params: no new/changed params")
        return None

    logger.info(
        "[param_detector] detect_url_params: %d new param(s) before noise filter",
        len(new_params),
    )
    for k, v in new_params.items():
        logger.info("[param_detector]   %s=%s", k, v)

    # Classify each parameter.
    clean_params: dict[str, list[str]] = {}
    complex_params: dict[str, list[str]] = {}
    noise_filtered: dict[str, list[str]] = {}
    for k, v in new_params.items():
        cat = _classify_param(k, v)
        if cat == "clean":
            clean_params[k] = v
        elif cat == "complex":
            complex_params[k] = v
        else:
            noise_filtered[k] = v

    if noise_filtered:
        logger.info(
            "[param_detector] filtered %d noise param(s): %s",
            len(noise_filtered),
            ", ".join(noise_filtered),
        )
    if complex_params:
        logger.info(
            "[param_detector] %d complex param(s): %s",
            len(complex_params),
            ", ".join(complex_params),
        )

    new_params = clean_params
    if not new_params and not complex_params:
        logger.info("[param_detector] all params were noise — no detection")
        return None

    # Build a normalised lookup of fill values.
    # Key: lowercased stripped value → FillValue object.
    fill_lookup: dict[str, FillValue] = {}
    for fv in fill_values:
        normalised = fv.value.strip().lower()
        if normalised:
            # First fill wins (preserves earliest occurrence).
            fill_lookup.setdefault(normalised, fv)

    # Also build a lookup from the raw inputs dict (fallback for values
    # set via date pickers, dropdowns etc. that bypass fill() calls).
    input_lookup: dict[str, tuple[str, str]] = {}  # normalised → (key, original)
    if inputs:
        for key, val in inputs.items():
            s = str(val).strip().lower()
            if s:
                input_lookup.setdefault(s, (key, str(val)))

    # Match parameter values against fill values, then inputs.
    matches: list[ParamMatch] = []
    for param_name, param_values in new_params.items():
        matched = False
        for pv in param_values:
            decoded = unquote_plus(pv).strip().lower()

            # Strategy 1: exact match against fill values
            if decoded in fill_lookup:
                fv = fill_lookup[decoded]
                matches.append(ParamMatch(
                    param_name=param_name,
                    param_value=pv,
                    fill=fv,
                ))
                logger.info(
                    "[param_detector]   MATCH (exact): %s=%s ← %s(\"%s\")",
                    param_name, pv, fv.action, fv.value,
                )
                matched = True
                break

            # Strategy 2: contains match (fill value is substring of
            # param value).  Handles autocomplete expansion like
            # "Barcelona, Spain" → "Barcelona, Catalonia ..., Spain".
            # Require ≥4 chars to avoid spurious short-string matches.
            for norm_fill, fv in fill_lookup.items():
                if len(norm_fill) >= 4 and norm_fill in decoded:
                    matches.append(ParamMatch(
                        param_name=param_name,
                        param_value=pv,
                        fill=fv,
                    ))
                    logger.info(
                        "[param_detector]   MATCH (contains): %s=%s ← %s(\"%s\")",
                        param_name, pv, fv.action, fv.value,
                    )
                    matched = True
                    break
            if matched:
                break

            # Strategy 3: exact match against raw inputs dict
            if decoded in input_lookup:
                inp_key, inp_val = input_lookup[decoded]
                # Create a synthetic FillValue for the match.
                fv = FillValue(
                    action=f"input:{inp_key}",
                    value=inp_val,
                    line=0,
                )
                matches.append(ParamMatch(
                    param_name=param_name,
                    param_value=pv,
                    fill=fv,
                ))
                logger.info(
                    "[param_detector]   MATCH (input): %s=%s ← inputs[\"%s\"]",
                    param_name, pv, inp_key,
                )
                matched = True
                break

        if not matched:
            logger.info(
                "[param_detector]   no match: %s=%s",
                param_name, param_values,
            )

    if not matches and not complex_params:
        logger.info("[param_detector] no fill↔param matches found")
        return None

    if matches:
        logger.info(
            "[param_detector] detected %d URL param(s) matching form fills",
            len(matches),
        )
    elif complex_params:
        logger.info(
            "[param_detector] no clean matches, but %d complex param(s) found (Tier 2)",
            len(complex_params),
        )

    return DetectionResult(
        url=url_after,
        matches=tuple(sorted(matches, key=lambda m: m.fill.line)),
        all_new_params=new_params,
        complex_params=complex_params or None,
    )


# ═══════════════════════════════════════════════════════════════════════
#  Stage 3 — Format the hint for the AI
# ═══════════════════════════════════════════════════════════════════════


def _format_match_line(m: ParamMatch) -> str:
    """Format a single match as a readable mapping line.

    Shows ``inputs["key"]  →  param_name`` when the match came from
    the inputs dict, or ``"literal value"  →  param_name`` when it
    came from a string literal in the code.
    """
    if m.fill.action.startswith("input:"):
        # Matched via the inputs dict — show the key reference.
        inp_key = m.fill.action.removeprefix("input:")
        left = f'inputs["{inp_key}"]'
    else:
        # Matched via a literal fill value — show the quoted value.
        left = f'"{m.fill.value}"'
    return f"  {left}  \u2192  {m.param_name}"


def format_hint(result: DetectionResult) -> str:
    """Format a detection result as a concise hint for the AI.

    Only called when ``detect_url_params`` returned a non-None result,
    so there is always at least one match.
    """
    lines: list[str] = [
        "\n[URL PARAMETER DETECTION HINT] "
        "After your form submission, the page navigated to:",
        f"  {result.url}",
        "",
        "We checked this URL and found that some of your "
        "form inputs appear as query parameters:",
    ]
    for m in result.matches:
        lines.append(_format_match_line(m))

    # Show a few unmatched params as context (with "etc." to hint
    # there may be others without implying a specific count).
    matched_names = {m.param_name for m in result.matches}
    unmatched = {
        k: v for k, v in result.all_new_params.items()
        if k not in matched_names
    }
    if unmatched:
        preview = ", ".join(
            f"{k}={vs[0]}" for k, vs in list(unmatched.items())[:5]
        )
        lines.append(f"  ({preview}, etc.)")

    lines.append(
        "\nIf possible, use page.goto() with these parameters in your "
        "final script instead of filling forms \u2014 it is faster and more "
        "reliable. Only use parameter names you are sure of from the URLs "
        "you have seen. If you don't know the parameter name for an input, "
        "don't guess \u2014 fill the form field, select the filter, or click "
        "the option in the UI, then check the resulting URL to see how "
        "the site encodes it."
    )
    return "\n".join(lines)


def format_hint_complex(result: DetectionResult) -> str:
    """Format a Tier 2 hint for complex-encoded URL parameters.

    Used when no clean key=value matches were found but the URL
    contains structured encodings (JSON blobs, base64, etc.) that
    embed the user's form inputs.
    """
    lines: list[str] = [
        "\n[URL PARAMETER DETECTION HINT] "
        "After your form submission, the page navigated to:",
        f"  {result.url}",
        "",
        "Your form inputs appear to be encoded in this URL. You may be "
        "able to construct this URL directly in your final script instead "
        "of filling forms. To understand the full URL format, fill each "
        "form field and apply each filter during exploration, then observe "
        "how the URL changes \u2014 this will show you exactly how the site "
        "encodes each parameter.",
    ]
    return "\n".join(lines)
