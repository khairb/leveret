"""
Unstable CSS Class Name Detector

Classifies CSS class names as stable (human-written, semantic) or unstable
(auto-generated, hashed, framework artifacts) using multiple heuristic signals.

Usage:
    from unstable_class_detector import is_unstable_class, clean_html

    is_unstable_class("sc-bdVTJa")      # True
    is_unstable_class("product-card")    # False

    cleaned = clean_html(raw_html)       # HTML with unstable classes removed
"""

import math
import re
from functools import lru_cache

# ---------------------------------------------------------------------------
# 1. Compact English word set (common words found in CSS class names)
# ---------------------------------------------------------------------------
_CSS_WORDS = frozenset(
    # Layout & structure
    "container wrapper inner outer main content body header footer sidebar"
    " section article nav navbar navigation menu menubar toolbar topbar"
    " bottombar left right center middle top bottom row col column grid"
    " flex block inline stack frame panel pane box card tile cell group"
    " area zone region layer overlay backdrop modal dialog popup drawer"
    " dropdown tooltip popover accordion tab tabs carousel slider"
    " layout page pages site wrap root base app shell scaffold"
    # Components & UI
    " button btn link anchor icon image img avatar thumbnail logo badge"
    " tag chip label caption title subtitle heading subheading text"
    " paragraph description summary detail details note alert warning"
    " error success info notification toast banner message callout hint"
    " input textarea select option checkbox radio toggle switch range"
    " form field control search filter sort pagination breadcrumb"
    " stepper progress spinner loader skeleton placeholder divider"
    " separator spacer gap border outline shadow embed widget component"
    " figure figcaption picture video audio source track canvas"
    # States & modifiers
    " active inactive disabled enabled selected checked focused"
    " hovered pressed open closed expanded collapsed visible hidden"
    " show hide shown loading loaded empty full ready pending"
    " primary secondary tertiary accent muted subtle bold light dark"
    " small medium large extra mini micro compact wide narrow"
    " rounded circle square flat raised elevated floating fixed sticky"
    " responsive fluid static relative absolute current"
    # Semantic / content
    " product item list table thead tbody tfoot tr th td"
    " price cost amount total discount sale offer deal coupon"
    " name email phone address date time year month day hour"
    " user profile account settings preferences dashboard home"
    " post blog feed story comment reply share like save bookmark"
    " rating star review score vote count number index rank"
    " category categories type status role level tier"
    " feature hero cta action submit cancel confirm delete edit"
    " create add new update remove clear reset close back next prev"
    " first last step steps stage phase intro outro preview"
    " cover background foreground highlight focus spotlight"
    # Common prefixes/suffixes used as words
    " wrap sub info meta data src ref alt opt max min"
    " no yes on off in out up down is has not with for from"
    " mobile desktop tablet portrait landscape print screen"
    " sm md lg xl xxl xs"
    # Colors
    " red blue green yellow orange purple pink white black gray grey"
    " teal cyan indigo violet brown"
    # Misc common
    " animation transition transform fade slide zoom scale rotate"
    " scroll overflow truncate ellipsis nowrap break"
    " align justify start end between around evenly stretch"
    " order grow shrink basis auto none inherit initial"
    " font size weight style variant decoration spacing tracking"
    " leading line height width depth opacity cursor pointer"
    " margin padding radius"
    " test debug dev prod staging"
    " view display render template slot"
    " social map chart graph diagram table"
    " file doc document folder upload download attachment"
    " notification bell inbox mail"
    " arrow caret chevron plus minus check cross times"
    " handle grip drag drop resize move"
    " col cols row rows span offset"
    " theme mode color scheme".split()
)

# ---------------------------------------------------------------------------
# 2. Known CSS-in-JS / build-tool prefixes (high-confidence UNSTABLE)
#    These are framework internals, not component library classes.
# ---------------------------------------------------------------------------
_FRAMEWORK_PREFIX_RE = re.compile(
    r"^("
    r"css-[a-zA-Z0-9]|"  # Emotion / generic CSS-in-JS: css-1a2b3c
    r"sc-[a-zA-Z]{3,}|"  # Styled Components: sc-bdVTJa (NOT sc-sm etc.)
    r"emotion-[0-9]|"  # Emotion: emotion-0, emotion-12
    r"jss\d|"  # JSS: jss123, jss45
    r"styled-[a-zA-Z0-9]{4,}|"  # styled-hR83ls
    r"__next|"  # Next.js internals
    r"svelte-[a-z0-9]{4,}|"  # Svelte scoped: svelte-1hjk32
    r"astro-[a-zA-Z0-9]{4,}|"  # Astro scoped
    r"v-[a-f0-9]{6,}"  # Vue scoped (v- + hex hash, not v-if etc.)
    r")"
)

# ---------------------------------------------------------------------------
# 3. Patterns
# ---------------------------------------------------------------------------

# CSS Modules: component__element--hashOfFiveOrMore
_CSS_MODULES_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9]*__[a-zA-Z][a-zA-Z0-9]*--[a-zA-Z0-9]{5,}$")

# Pure hash: entirely random-looking, no delimiters, mixed case+digits
# Must contain BOTH letters and digits to avoid matching real words
_PURE_HASH_RE = re.compile(r"^(?=.*[a-zA-Z])(?=.*[0-9])[a-zA-Z0-9]{6,}$")

# Hash suffix: real-name followed by a delimiter and a hash
# e.g., "styles-a8f2d1", "component_3kx8fj"
_HASH_SUFFIX_RE = re.compile(r"^.+[_-](?=.*[0-9])[a-zA-Z0-9]{5,}$")

# Underscore-prefixed hash (CSS Modules default): _3xk2F, _a8Bz2K
_UNDERSCORE_HASH_RE = re.compile(
    r"^_(?=.*[0-9])[a-zA-Z0-9]{4,}$|^_(?=.*[A-Z])(?=.*[a-z])[a-zA-Z]{5,}$"
)

# Tailwind-style utilities — these are STABLE, protect them
_TAILWIND_RE = re.compile(
    r"^-?"
    r"("
    # Variant prefixes
    r"(hover|focus|focus-within|focus-visible|active|visited|group-hover|"
    r"dark|sm|md|lg|xl|2xl|first|last|odd|even|disabled|placeholder|"
    r"before|after|peer|motion-safe|motion-reduce):"
    r")?"
    r"("
    # Spacing & sizing
    r"[mp][trblxyse]?-[0-9]|"
    r"w-|h-|min-w-|min-h-|max-w-|max-h-|size-|"
    # Typography
    r"text-|font-|leading-|tracking-|"
    # Backgrounds & borders
    r"bg-|border-|rounded|shadow|ring-|outline-|opacity-|"
    # Layout
    r"flex|grid|block|inline|hidden|table|contents|"
    r"justify-|items-|self-|place-|content-|"
    r"gap-|space-[xy]-|"
    # Positioning
    r"z-|top-|right-|bottom-|left-|inset-|"
    r"overflow-|"
    # Transitions & transforms
    r"transition|duration-|ease-|delay-|"
    r"translate-|rotate-|skew-|scale-|origin-|"
    # Interactivity
    r"cursor-|pointer-events-|select-|touch-|"
    r"sr-only|not-sr-only|"
    # Other
    r"aspect-|columns-|break-|object-|"
    r"decoration-|underline|overline|line-through|no-underline|"
    r"list-|whitespace-|"
    r"accent-|caret-|scroll-|snap-|will-change-"
    r")",
)

# BEM splitter
_BEM_SPLIT_RE = re.compile(r"[_\-]{1,2}")

# CamelCase splitter
_CAMEL_RE = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


# ---------------------------------------------------------------------------
# 4. Scoring helpers
# ---------------------------------------------------------------------------
def _shannon_entropy(s: str) -> float:
    """Shannon entropy of a string (bits per character)."""
    if not s:
        return 0.0
    freq: dict[str, int] = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    n = len(s)
    return -sum((count / n) * math.log2(count / n) for count in freq.values())


def _split_into_segments(class_name: str) -> list[str]:
    """Split a class name into word-like segments."""
    parts = _BEM_SPLIT_RE.split(class_name)
    segments = []
    for part in parts:
        segments.extend(_CAMEL_RE.split(part))
    return [s.lower() for s in segments if s and len(s) > 0]


def _word_coverage(segments: list[str]) -> float:
    """Fraction of segments that are known English/CSS words."""
    if not segments:
        return 0.0
    matches = sum(1 for s in segments if s.lower() in _CSS_WORDS)
    return matches / len(segments)


def _digit_ratio(s: str) -> float:
    """Fraction of characters that are digits."""
    if not s:
        return 0.0
    return sum(1 for c in s if c.isdigit()) / len(s)


def _has_mixed_case_no_camel(s: str) -> bool:
    """
    Check for suspicious mixed case that isn't standard camelCase or PascalCase.
    e.g., 'bdVTJa' has unusual uppercase patterns unlike 'productCard'.
    """
    alpha = re.sub(r"[^a-zA-Z]", "", s)
    if len(alpha) < 4:
        return False
    # Count case transitions
    transitions = sum(
        1 for i in range(1, len(alpha)) if alpha[i].isupper() != alpha[i - 1].isupper()
    )
    # High transition rate relative to length → random-looking
    return transitions / len(alpha) > 0.4


# ---------------------------------------------------------------------------
# 5. Main classifier
# ---------------------------------------------------------------------------
@lru_cache(maxsize=8192)
def is_unstable_class(class_name: str, threshold: float = 0.45) -> bool:
    """
    Determine whether a CSS class name is likely unstable / auto-generated.

    Args:
        class_name: A single CSS class name (no spaces).
        threshold: Confidence threshold (0-1). Classes scoring above this
                   are considered unstable. Default 0.45 balances precision
                   and recall well for most sites.

    Returns:
        True if the class is likely unstable and should be removed.
    """
    name = class_name.strip()
    if not name:
        return False

    # ---- Fast-path: protect known stable patterns ----

    # Very short classes (1-2 chars) are usually intentional
    if len(name) <= 2:
        return False

    # Tailwind utilities are stable
    if _TAILWIND_RE.match(name):
        return False

    # data-* attributes sometimes appear as classes; leave them
    if name.startswith("data-"):
        return False

    # ---- Fast-path: high-confidence unstable patterns ----

    # Known CSS-in-JS framework prefixes
    if _FRAMEWORK_PREFIX_RE.match(name):
        return True

    # CSS Modules hash pattern: component__element--a8f2d
    if _CSS_MODULES_RE.match(name):
        return True

    # Underscore-prefixed hashes: _3xk2F
    if _UNDERSCORE_HASH_RE.match(name):
        return True

    # ---- Scoring-based classification ----
    score = 0.0
    segments = _split_into_segments(name)
    word_cov = _word_coverage(segments)
    alphanum = re.sub(r"[^a-zA-Z0-9]", "", name)

    # Signal 1: word coverage (strongest protective signal)
    if word_cov >= 1.0 and len(segments) >= 2:
        return False  # all segments are real words → definitely stable
    if word_cov >= 0.8:
        score -= 0.4
    elif word_cov >= 0.5:
        score -= 0.2
    elif word_cov == 0.0 and len(segments) >= 2:
        score += 0.25

    # Signal 2: entropy of the alphanumeric content
    entropy = _shannon_entropy(alphanum)
    if len(alphanum) >= 6:  # entropy is unreliable on very short strings
        if entropy > 4.0:
            score += 0.2
        elif entropy > 3.5:
            score += 0.1
        elif entropy < 2.5:
            score -= 0.1

    # Signal 3: digit ratio
    dr = _digit_ratio(alphanum)
    if dr > 0.4:
        score += 0.2
    elif dr > 0.25:
        score += 0.1

    # Signal 4: pure hash (letters + digits, no delimiters, 6+ chars)
    if _PURE_HASH_RE.match(name):
        score += 0.3

    # Signal 5: hash suffix on otherwise normal name
    # Fire if the suffix part itself looks like a hash (has digits mixed with letters)
    if _HASH_SUFFIX_RE.match(name):
        score += 0.2

    # Signal 6: suspicious mixed case (not clean camelCase)
    if _has_mixed_case_no_camel(alphanum):
        score += 0.15

    # Signal 7: any segment looks like a hash (has digits + letters, not a word)
    for seg in segments:
        if (
            len(seg) >= 5
            and seg not in _CSS_WORDS
            and any(c.isdigit() for c in seg)
            and any(c.isalpha() for c in seg)
        ):
            score += 0.35
            break

    # Signal 8: 5+ consecutive consonants (rare in real words)
    if re.search(r"[bcdfghjklmnpqrstvwxyz]{5,}", alphanum.lower()):
        score += 0.1

    return score >= threshold


# ---------------------------------------------------------------------------
# 6. HTML cleaner — removes unstable classes from HTML string
# ---------------------------------------------------------------------------
_CLASS_ATTR_RE = re.compile(
    r"""(class\s*=\s*)(["'])([^"']*?)\2""",
    re.IGNORECASE,
)


def _clean_class_attr(match: re.Match) -> str:
    prefix = match.group(1)
    quote = match.group(2)
    classes = match.group(3)
    kept = [c for c in classes.split() if not is_unstable_class(c)]
    if not kept:
        return ""  # remove entire class attribute if empty
    return f"{prefix}{quote}{' '.join(kept)}{quote}"


def clean_html(html: str) -> str:
    """
    Remove unstable CSS classes from an HTML string.

    Keeps the HTML structure intact, only modifying class attribute values.
    Classes identified as unstable are removed; if all classes on an element
    are unstable, the entire class attribute is removed.

    Args:
        html: Raw HTML string.

    Returns:
        Cleaned HTML with unstable classes stripped.
    """
    return _CLASS_ATTR_RE.sub(_clean_class_attr, html)


# ---------------------------------------------------------------------------
# 7. CLI / demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_cases = [
        # (class_name, expected_unstable)
        # ---- STABLE classes (should return False) ----
        ("product-card", False),
        ("main-nav", False),
        ("sidebar-menu", False),
        ("btn-primary", False),
        ("header__logo", False),  # BEM without hash
        ("text-lg", False),
        ("flex", False),
        ("mt-4", False),
        ("hover:bg-blue-500", False),
        ("container", False),
        ("dark:text-white", False),
        ("items-center", False),
        ("w-full", False),
        ("sr-only", False),
        ("nav-container", False),
        ("sidebar_content", False),
        ("user_profile", False),
        ("ProductCard", False),  # PascalCase component
        ("is-active", False),  # state class (semantic)
        ("has-error", False),
        ("page-title", False),
        ("col-md-6", False),  # Bootstrap grid
        ("ant-btn", False),  # Ant Design (stable component lib)
        ("ant-modal-header", False),
        ("el-input", False),  # Element UI
        ("mat-button", False),  # Angular Material
        ("form-control", False),  # Bootstrap
        ("list-group-item", False),
        ("d-flex", False),  # Bootstrap utility
        ("text-center", False),
        ("bg-primary", False),
        ("alert-danger", False),
        ("card-body", False),
        ("table-striped", False),
        ("scraper", False),
        ("reminder", False),
        ("clock", False),
        ("timer", False),
        ("stopwatch", False),
        ("countdown", False),
        ("timer-container", False),
        ("my-life", False),
        ("bird", False),
        ("animal", False),
        ("plant", False),
        ("tree", False),
        ("flower", False),
        ("fruit", False),
        ("vegetable", False),
        ("animal", False),
        ("plant", False),
        ("tree", False),
        ("flower", False),
        ("fruit", False),
        ("vegetable", False),
        ("animal", False),
        ("plant", False),
        ("tree", False),
        ("flower", False),
        ("fruit", False),
        # ---- UNSTABLE classes (should return True) ----
        ("css-1a2b3c", True),
        ("sc-bdVTJa", True),
        ("emotion-0", True),
        ("jss123", True),
        ("styled-hR83ls", True),
        ("_3xk2F", True),
        ("_next-data", False),  # Actually this is an attr, not class
        ("__next", True),
        ("MuiButton-root", True),
        ("styles__container--a8f2d", True),
        ("svelte-1hjk32", True),
        ("v-4af2c3", True),
        ("css-14el2xx", True),
        ("sc-AxjAm", True),
        ("kJHsdf8", True),  # pure hash with mixed case + digit
        ("a3kF8x2Z", True),  # pure hash
        ("hashed_8faj3k2lx", True),
        ("component_a3b8c2d1f", True),
        ("emotion-12", True),
        ("jss45", True),
        ("svelte-ab12cd", True),
        ("astro-J7PkDi3T", True),
        ("v-a1b2c3d4", True),  # Vue scoped hash
    ]

    print(f"{'Class Name':<40} {'Expected':<10} {'Got':<10} {'OK?'}")
    print("-" * 75)
    correct = 0
    failures = []
    for name, expected in test_cases:
        result = is_unstable_class(name)
        ok = result == expected
        correct += ok
        status = "✓" if ok else "✗ WRONG"
        if not ok:
            failures.append((name, expected, result))
        print(f"{name:<40} {str(expected):<10} {str(result):<10} {status}")

    print(f"\n{correct}/{len(test_cases)} correct")
    if failures:
        print("\nFailed cases:")
        for name, expected, got in failures:
            print(f"  {name}: expected {expected}, got {got}")

    # Demo: clean HTML
    sample_html = """
    <div class="css-1a2b3c container flex items-center">
        <nav class="sc-bdVTJa main-nav">
            <a class='emotion-0 btn-primary hover:bg-blue-500' href="/">Home</a>
        </nav>
        <div class="styles__sidebar--f8a2d sidebar-menu">
            <span class="product-card _3xk2F">Item</span>
        </div>
        <button class="ant-btn ant-btn-primary MuiButton-root">Click</button>
    </div>
    """
    print("\n\n--- HTML Cleaning Demo ---")
    print("BEFORE:")
    print(sample_html)
    print("AFTER:")
    print(clean_html(sample_html))
