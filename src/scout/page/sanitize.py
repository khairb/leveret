"""
HTML Sanitizer for Web Scraping Agents

This module provides functionality to minimize HTML content by removing elements
that are not relevant for scraping, while preserving the essential structure
needed for query selection and data extraction.
"""

import re
import time
from bs4 import BeautifulSoup, Comment, NavigableString

import logging
from .classes import clean_html as clean_unstable_classes

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# Elements to completely remove (including their contents)
ELEMENTS_TO_REMOVE = {
    "script",
    "style",
    "noscript",
    "link",
    "meta",
    "iframe",
    "template",
    "source",
    "track",
    "embed",
    "object",
    "param",
    "map",
    "area",
}

# Elements where we keep the tag but remove inner content (like SVG paths)
ELEMENTS_TO_EMPTY = {
    "svg",
    "math",
    "canvas",
}

# Attributes to always remove
ATTRIBUTES_TO_REMOVE = {
    # Style and visual
    "style",
    "width",
    "height",
    "border",
    "align",
    "valign",
    "bgcolor",
    "color",
    "face",
    "size",
    "cellpadding",
    "cellspacing",
    # Event handlers
    "onclick",
    "ondblclick",
    "onmousedown",
    "onmouseup",
    "onmouseover",
    "onmousemove",
    "onmouseout",
    "onmouseenter",
    "onmouseleave",
    "onkeydown",
    "onkeypress",
    "onkeyup",
    "onfocus",
    "onblur",
    "onchange",
    "onsubmit",
    "onreset",
    "onselect",
    "oninput",
    "onload",
    "onunload",
    "onerror",
    "onscroll",
    "onresize",
    "ontouchstart",
    "ontouchmove",
    "ontouchend",
    "ondrag",
    "ondragstart",
    "ondragend",
    "ondragover",
    "ondragenter",
    "ondragleave",
    "ondrop",
    # Form validation (usually not needed for scraping)
    "required",
    "pattern",
    "minlength",
    "maxlength",
    "min",
    "max",
    "step",
    "autocomplete",
    "autofocus",
    "novalidate",
    "formnovalidate",
    # Loading/performance
    "loading",
    "decoding",
    "fetchpriority",
    "crossorigin",
    "referrerpolicy",
    "integrity",
    "async",
    "defer",
    # Accessibility (usually not needed, but can be kept if required)
    "tabindex",
    "accesskey",
    "contenteditable",
    "draggable",
    "spellcheck",
    "translate",
    # Media-specific
    "sizes",
    "autoplay",
    "controls",
    "loop",
    "muted",
    "preload",
    "poster",
    "playsinline",
}

# Attribute prefixes to remove (regex patterns)
ATTRIBUTE_PREFIXES_TO_REMOVE = [
    r"^on[a-z]+$",  # All event handlers
    r"^data-gtm",  # Google Tag Manager
    r"^data-ga",  # Google Analytics
    r"^data-analytics",  # Generic analytics
    r"^data-tracking",  # Tracking
    r"^data-ad",  # Advertising
    r"^data-react",  # React internals
    r"^data-reactid",  # React ID
    r"^_ngcontent",  # Angular
    r"^_nghost",  # Angular
    r"^ng-",  # Angular directives
    r"^v-",  # Vue directives (but not v-if which affects rendering)
    r"^x-",  # Alpine.js
    r"^wire:",  # Livewire
    r"^hx-",  # HTMX (usually not needed)
    r"^:class$",  # Vue dynamic binding
    r"^:style$",  # Vue dynamic binding
    r"^@",  # Vue event handlers
]

# Compiled regex patterns
ATTRIBUTE_PREFIX_PATTERNS = [re.compile(p, re.IGNORECASE) for p in ATTRIBUTE_PREFIXES_TO_REMOVE]

# Attributes to truncate if their value exceeds MAX_ATTR_LENGTH characters.
# These are not CSS selectors, so truncating is safe, but they may still
# contain partial data worth keeping.
MAX_ATTR_LENGTH = 80

ATTRIBUTES_TO_TRUNCATE = {
    # URL attributes — tracking params, base64 data URIs, redirect chains
    "src",
    "href",
    "action",
    "formaction",
    "background",
    "srcset",
    "ping",
    # Citation / structured-data URLs
    "cite",
    "itemid",
    # Text content attributes
    "alt",
    "title",
    "aria-label",
    "placeholder",
    # Value / content — tokens, schema.org descriptions
    "value",
    "content",
}

# Attributes to keep (whitelist for important scraping attributes)
ATTRIBUTES_TO_KEEP = {
    "id",
    "class",
    "name",
    "alt",
    "type",
    "value",
    "placeholder",
    "action",
    "method",
    "for",
    "role",
    "aria-label",
    "aria-labelledby",
    "aria-describedby",
    "itemtype",
    "itemprop",
    "itemscope",
    "content",
    "property",
    "rel",
    "target",
    "colspan",
    "rowspan",
    "headers",
    "scope",
    "datetime",
    "cite",
    "lang",
    "dir",
}

# Common boilerplate section identifiers (classes and IDs)
BOILERPLATE_PATTERNS = [
    r"(?:^|[-_])nav(?:[-_]|$|igation)",
    r"(?:^|[-_])header(?:[-_]|$)",
    r"(?:^|[-_])footer(?:[-_]|$)",
    r"(?:^|[-_])sidebar(?:[-_]|$)",
    r"(?:^|[-_])menu(?:[-_]|$)",
    r"(?:^|[-_])cookie",
    r"(?:^|[-_])gdpr",
    r"(?:^|[-_])consent",
    r"(?:^|[-_])banner(?:[-_]|$)",
    r"(?:^|[-_])modal(?:[-_]|$)",
    r"(?:^|[-_])popup(?:[-_]|$)",
    r"(?:^|[-_])overlay(?:[-_]|$)",
    r"(?:^|[-_])ad(?:s|vert)?(?:[-_]|$)",
    r"(?:^|[-_])sponsor",
    r"(?:^|[-_])promo(?:tion)?(?:[-_]|$)",
    r"(?:^|[-_])social[-_]?(?:share|links|media)",
    r"(?:^|[-_])share[-_]?(?:button|link)",
    r"(?:^|[-_])comment(?:s)?(?:[-_]|$)",
    r"(?:^|[-_])related[-_]?(?:post|article)",
    r"(?:^|[-_])recommend",
    r"(?:^|[-_])newsletter",
    r"(?:^|[-_])subscribe",
    r"(?:^|[-_])signup",
    r"(?:^|[-_])login",
    r"(?:^|[-_])breadcrumb",
    r"(?:^|[-_])pagination",
    r"(?:^|[-_])skip[-_]?(?:link|nav)",
    r"(?:^|[-_])search(?:[-_]box|[-_]form)?$",
]

BOILERPLATE_COMPILED = [re.compile(p, re.IGNORECASE) for p in BOILERPLATE_PATTERNS]

# ---------------------------------------------------------------------------
# Repeating Elements Detection
# ---------------------------------------------------------------------------

# Semantic list containers and their expected children
SEMANTIC_LIST_CONTAINERS = {
    "ul": {"li"},
    "ol": {"li"},
    "tbody": {"tr"},
    "thead": {"tr"},
    "tfoot": {"tr"},
    "table": {"tr"},  # Direct tr children (no tbody)
    "dl": {"dt", "dd"},
    "select": {"option"},
    "datalist": {"option"},
    "optgroup": {"option"},
}

# Tags that commonly contain repeated items
COMMON_LIST_ITEM_TAGS = {"li", "tr", "option", "dt", "dd", "article", "section"}

# Parent class patterns that suggest list containers
LIST_CONTAINER_CLASS_PATTERNS = [
    r"(?:^|[-_])list(?:[-_]|$|s)",
    r"(?:^|[-_])grid(?:[-_]|$)",
    r"(?:^|[-_])items(?:[-_]|$)",
    r"(?:^|[-_])results(?:[-_]|$)",
    r"(?:^|[-_])cards(?:[-_]|$)",
    r"(?:^|[-_])entries(?:[-_]|$)",
    r"(?:^|[-_])rows(?:[-_]|$)",
]

LIST_CONTAINER_PATTERNS_COMPILED = [
    re.compile(p, re.IGNORECASE) for p in LIST_CONTAINER_CLASS_PATTERNS
]

# Default truncation settings
DEFAULT_MIN_REPEATING_ITEMS = 6
DEFAULT_KEEP_FIRST = 3
DEFAULT_KEEP_LAST = 3


def _should_remove_attribute(attr_name: str) -> bool:
    """Check if an attribute should be removed."""
    attr_lower = attr_name.lower()
    
    # Always keep whitelisted attributes
    if attr_lower in ATTRIBUTES_TO_KEEP:
        return False
    
    # Keep data-* attributes by default (they're often useful for scraping)
    # unless they match a specific pattern to remove
    if attr_lower.startswith("data-"):
        for pattern in ATTRIBUTE_PREFIX_PATTERNS:
            if pattern.match(attr_lower):
                return True
        return False  # Keep other data-* attributes
    
    # Remove if in the removal set
    if attr_lower in ATTRIBUTES_TO_REMOVE:
        return True
    
    # Check prefix patterns
    for pattern in ATTRIBUTE_PREFIX_PATTERNS:
        if pattern.match(attr_lower):
            return True
    
    return False


def _is_boilerplate_element(element, use_boilerplate_patterns: bool = True) -> bool:
    """Check if an element appears to be boilerplate (nav, footer, etc.)."""
    # Check ID

    element_id = element.get("id", "")
    if element_id and use_boilerplate_patterns:
        for pattern in BOILERPLATE_COMPILED:
            if pattern.search(element_id):
                return True
    
    # Check classes
    classes = element.get("class", [])
    if isinstance(classes, str):
        classes = classes.split()
    if use_boilerplate_patterns:
        for cls in classes:
            for pattern in BOILERPLATE_COMPILED:
                if pattern.search(cls):
                    return True
    
    # Check semantic elements
    if element.name in {"nav", "aside", "footer", "header"}:
        return True
    
    # Check role attribute
    role = element.get("role", "").lower()
    if role in {"navigation", "banner", "contentinfo", "complementary", "search"}:
        return True
    
    return False


def _is_empty_element(element) -> bool:
    """Check if an element is effectively empty and can be removed."""

    # Self-closing elements that might be meaningful
    if element.name in {"img", "input", "br", "hr", "source", "track", "embed", "area"}:
        return False
    
    # Check if element has any non-whitespace text content
    text = element.get_text(strip=True)
    if text:
        return False
    
    # Check if element has any meaningful attributes
    meaningful_attrs = {"id", "class", "name", "data-", "role", "aria-"}
    for attr in element.attrs:
        if attr in meaningful_attrs or attr.startswith("data-"):
            return False
    
    # Check if it has non-empty children
    for child in element.children:
        if isinstance(child, NavigableString):
            if child.strip():
                return False
        elif hasattr(child, "name") and child.name:
            if not _is_empty_element(child):
                return False
    
    return True


def _truncate_long_attributes(element) -> None:
    """Truncate attribute values that exceed MAX_ATTR_LENGTH characters.

    Named attributes in ATTRIBUTES_TO_TRUNCATE and all data-* attributes are
    candidates. Only values longer than MAX_ATTR_LENGTH are touched; shorter
    values are left as-is.
    """
    for attr, value in list(element.attrs.items()):
        if isinstance(value, list):
            # BeautifulSoup stores multi-value attrs (e.g. class) as lists — skip.
            continue
        attr_lower = attr.lower()
        is_candidate = (
            attr_lower in ATTRIBUTES_TO_TRUNCATE
            or attr_lower.startswith("data-")
        )
        if is_candidate and len(value) > MAX_ATTR_LENGTH:
            element[attr] = value[:MAX_ATTR_LENGTH] + "...(omitted)"


def _truncate_text(element, max_length: int = 200):
    """Truncate long text content within an element."""
    for text_node in element.find_all(string=True):
        if isinstance(text_node, NavigableString) and not isinstance(text_node, Comment):
            text = str(text_node).strip()
            if len(text) > max_length:
                truncated = text[:max_length] + "..."
                text_node.replace_with(truncated)


def _limit_class_count(element, max_classes: int = 10):
    """Limit the number of classes on an element."""
    classes = element.get("class", [])
    if isinstance(classes, list) and len(classes) > max_classes:
        element["class"] = classes[:max_classes]


# ---------------------------------------------------------------------------
# Repeating Elements Detection & Truncation (Sliding Window Approach)
# ---------------------------------------------------------------------------


def _is_similar(el1, el2, class_threshold: float = 0.5) -> bool:
    """
    Check if two elements are structurally similar using sliding window comparison.

    This approach handles "drift" in long lists where element #1 might differ
    from element #100, but each element is similar to its neighbors.

    Args:
        el1: First BeautifulSoup element
        el2: Second BeautifulSoup element
        class_threshold: Minimum Jaccard similarity for class overlap (0.0-1.0)

    Returns:
        True if elements are structurally similar
    """
    # Must have same tag name
    if el1.name != el2.name:
        return False

    # Get class sets
    c1 = el1.get("class", [])
    c2 = el2.get("class", [])
    if isinstance(c1, str):
        c1 = c1.split()
    if isinstance(c2, str):
        c2 = c2.split()
    c1_set = set(c1)
    c2_set = set(c2)

    # Check class overlap (Jaccard similarity)
    if c1_set and c2_set:
        intersection = len(c1_set & c2_set)
        union = len(c1_set | c2_set)
        if union > 0 and (intersection / union) < class_threshold:
            return False
    elif c1_set != c2_set:
        # One has classes, other doesn't - consider dissimilar
        return False

    # Check child tag structure (same tags in same order)
    children1 = [c.name for c in el1.children if hasattr(c, "name") and c.name]
    children2 = [c.name for c in el2.children if hasattr(c, "name") and c.name]

    return children1 == children2


# ---------------------------------------------------------------------------
# Dominant Element Detection & Truncation (Option C)
# ---------------------------------------------------------------------------


def _group_children_by_similarity(
    children: list,
    class_threshold: float = 0.5,
) -> list[list]:
    """
    Group children into clusters of similar elements.

    Uses the existing _is_similar() function to determine similarity.
    Each group contains elements that are structurally similar to each other.

    Args:
        children: List of BeautifulSoup elements (direct children of a parent)
        class_threshold: Minimum Jaccard similarity for class overlap

    Returns:
        List of groups, where each group is a list of similar elements
    """
    if not children:
        return []

    groups: list[list] = []

    for child in children:
        # Try to find an existing group this element belongs to
        found_group = False
        for group in groups:
            # Compare with the first element of the group (representative)
            if _is_similar(child, group[0], class_threshold):
                group.append(child)
                found_group = True
                break

        if not found_group:
            # Create a new group with this element
            groups.append([child])

    return groups


def _find_dominant_group(
    groups: list[list],
    min_items: int = DEFAULT_MIN_REPEATING_ITEMS,
    dominance_ratio: float = 1.5,
) -> list | None:
    """
    Find the dominant (most frequent) element group.

    Args:
        groups: List of element groups from _group_children_by_similarity()
        min_items: Minimum items required for a group to be considered dominant
        dominance_ratio: Dominant group must have this ratio more items than
                        the second-most-common group (e.g., 1.5 = 50% more)

    Returns:
        The dominant group (list of elements), or None if no clear dominant group
    """
    if not groups:
        return None

    # Sort groups by size (descending)
    sorted_groups = sorted(groups, key=len, reverse=True)
    dominant_group = sorted_groups[0]

    # Safeguard 1: Must have at least min_items
    if len(dominant_group) < min_items:
        return None

    # Safeguard 2: Must be significantly more common than second-most-common
    if len(sorted_groups) > 1:
        second_count = len(sorted_groups[1])
        if second_count > 0 and len(dominant_group) < second_count * dominance_ratio:
            # Not clearly dominant - two similarly-sized groups
            return None

    return dominant_group


def _get_indices_to_preserve(
    all_children: list,
    dominant_elements: list,
    keep_first: int = DEFAULT_KEEP_FIRST,
    keep_last: int = DEFAULT_KEEP_LAST,
) -> set[int]:
    """
    Determine which dominant element indices should be preserved.

    Preserves:
    1. First N elements of the dominant type
    2. Last N elements of the dominant type
    3. Elements adjacent to non-dominant elements (widgets/interstitials)

    Args:
        all_children: All children of the parent element
        dominant_elements: Elements belonging to the dominant group
        keep_first: Number of dominant elements to keep at the start
        keep_last: Number of dominant elements to keep at the end

    Returns:
        Set of indices (within dominant_elements list) to preserve
    """
    if not dominant_elements:
        return set()

    # Create a set of dominant element ids for fast lookup
    dominant_ids = {id(el) for el in dominant_elements}

    # Map each dominant element to its position in all_children
    dominant_to_all_idx: dict[int, int] = {}
    for all_idx, child in enumerate(all_children):
        if id(child) in dominant_ids:
            dominant_to_all_idx[id(child)] = all_idx

    # Create reverse mapping: dominant element index -> element
    dominant_idx_to_element = {i: el for i, el in enumerate(dominant_elements)}

    # Start with first N and last N
    preserve_indices: set[int] = set()
    preserve_indices.update(range(min(keep_first, len(dominant_elements))))
    preserve_indices.update(range(max(0, len(dominant_elements) - keep_last), len(dominant_elements)))

    # Find dominant elements adjacent to non-dominant elements
    for all_idx, child in enumerate(all_children):
        if id(child) not in dominant_ids:
            # This is a non-dominant element (widget/interstitial)
            # Find dominant elements immediately before and after

            # Check element before
            if all_idx > 0:
                prev_child = all_children[all_idx - 1]
                if id(prev_child) in dominant_ids:
                    # Find this element's index in dominant_elements
                    for dom_idx, dom_el in enumerate(dominant_elements):
                        if id(dom_el) == id(prev_child):
                            preserve_indices.add(dom_idx)
                            break

            # Check element after
            if all_idx < len(all_children) - 1:
                next_child = all_children[all_idx + 1]
                if id(next_child) in dominant_ids:
                    # Find this element's index in dominant_elements
                    for dom_idx, dom_el in enumerate(dominant_elements):
                        if id(dom_el) == id(next_child):
                            preserve_indices.add(dom_idx)
                            break

    return preserve_indices


def _truncate_dominant_with_adjacency(
    parent,
    min_items: int = DEFAULT_MIN_REPEATING_ITEMS,
    keep_first: int = DEFAULT_KEEP_FIRST,
    keep_last: int = DEFAULT_KEEP_LAST,
    class_threshold: float = 0.5,
    dominance_ratio: float = 1.5,
) -> dict:
    """
    Truncate repeating elements using dominant element detection.

    This approach:
    1. Groups children by structural similarity
    2. Identifies the dominant (most frequent) element type
    3. Truncates only the dominant elements, preserving:
       - First N and last N of the dominant type
       - Elements adjacent to non-dominant elements (widgets)
    4. Non-dominant elements are never touched

    Args:
        parent: Parent BeautifulSoup element to process
        min_items: Minimum items for a group to be truncated
        keep_first: Dominant elements to keep at start
        keep_last: Dominant elements to keep at end
        class_threshold: Similarity threshold for grouping
        dominance_ratio: Required ratio of dominant vs second-most-common

    Returns:
        Dict with statistics: {
            "dominant_found": bool,
            "dominant_count": int,
            "elements_removed": int,
            "preserved_adjacent": int,
        }
    """
    stats = {
        "dominant_found": False,
        "dominant_count": 0,
        "elements_removed": 0,
        "preserved_adjacent": 0,
    }

    # Get all direct element children (skip text nodes)
    all_children = [c for c in parent.children if hasattr(c, "name") and c.name]

    if len(all_children) < min_items:
        return stats

    # Step 1: Group children by similarity
    groups = _group_children_by_similarity(all_children, class_threshold)

    # Step 2: Find the dominant group
    dominant_group = _find_dominant_group(groups, min_items, dominance_ratio)

    if dominant_group is None:
        return stats

    stats["dominant_found"] = True
    stats["dominant_count"] = len(dominant_group)

    # Step 3: Determine which elements to preserve
    preserve_indices = _get_indices_to_preserve(
        all_children, dominant_group, keep_first, keep_last
    )

    # Calculate how many were preserved due to adjacency
    # (beyond the basic first N and last N)
    basic_preserve_count = min(keep_first, len(dominant_group)) + min(keep_last, max(0, len(dominant_group) - keep_first))
    stats["preserved_adjacent"] = len(preserve_indices) - min(basic_preserve_count, len(dominant_group))

    # Step 4: Determine elements to remove
    # (dominant elements that are NOT in preserve set)
    to_remove = [
        el for i, el in enumerate(dominant_group)
        if i not in preserve_indices
    ]

    if not to_remove:
        return stats

    # Step 5: Create placeholder comment and remove elements
    placeholder_text = (
        f" [TRUNCATED: {len(to_remove)} similar items omitted. "
        f"Showing {len(preserve_indices)} of {len(dominant_group)} items "
        f"(preserved {stats['preserved_adjacent']} adjacent to widgets).] "
    )
    placeholder = Comment(placeholder_text)

    # Insert placeholder before first item to remove
    to_remove[0].insert_before(placeholder)

    # Remove the elements
    for item in to_remove:
        item.decompose()

    stats["elements_removed"] = len(to_remove)

    return stats


def _find_repeating_runs(
    parent,
    min_run: int = DEFAULT_MIN_REPEATING_ITEMS,
    lookback: int = 3,
    class_threshold: float = 0.5,
) -> list[list]:
    """
    Find runs of similar consecutive sibling elements using sliding window.

    Each element is compared to the previous `lookback` elements in the current run.
    If similar to at least 2 of them (or all if fewer), it continues the run.
    This handles drift naturally - element #100 only needs to match #97, #98, #99.

    Args:
        parent: Parent BeautifulSoup element to search within
        min_run: Minimum run length to report
        lookback: Number of previous elements to compare against
        class_threshold: Passed to _is_similar()

    Returns:
        List of runs, where each run is a list of similar consecutive elements
    """
    # Get direct element children (skip text nodes)
    children = [c for c in parent.children if hasattr(c, "name") and c.name]

    if len(children) < min_run:
        return []

    runs = []
    current_run = [children[0]]

    for i, element in enumerate(children[1:], 1):
        # Get previous elements from current run (up to lookback)
        prev_elements = current_run[-lookback:]

        # Count how many previous elements this one is similar to
        similar_count = sum(
            1 for prev in prev_elements
            if _is_similar(element, prev, class_threshold)
        )

        # Threshold: similar to at least 2 of last 3, or all if fewer
        threshold = min(2, len(prev_elements))

        if similar_count >= threshold:
            current_run.append(element)
        else:
            # End current run, start new one
            if len(current_run) >= min_run:
                runs.append(current_run)
            current_run = [element]

    # Don't forget the last run
    if len(current_run) >= min_run:
        runs.append(current_run)

    return runs


def _find_repeating_runs_semantic(
    parent,
    min_run: int = DEFAULT_MIN_REPEATING_ITEMS,
) -> list[list]:
    """
    Find repeating runs in semantic HTML containers (ul, ol, table, etc.).

    For semantic containers, we can be more confident about what constitutes
    a list item, so we use simpler tag-based detection.

    Args:
        parent: Parent BeautifulSoup element (should be in SEMANTIC_LIST_CONTAINERS)
        min_run: Minimum run length to report

    Returns:
        List containing a single run if found, empty list otherwise
    """
    if parent.name not in SEMANTIC_LIST_CONTAINERS:
        return []

    expected_tags = SEMANTIC_LIST_CONTAINERS[parent.name]
    children = [c for c in parent.children if hasattr(c, "name") and c.name]

    # Get all children matching expected tags
    matching = [c for c in children if c.name in expected_tags]

    if len(matching) >= min_run:
        return [matching]

    return []


def _truncate_repeating_group(
    group: list,
    keep_first: int = DEFAULT_KEEP_FIRST,
    keep_last: int = DEFAULT_KEEP_LAST,
) -> int:
    """
    Truncate a repeating group, keeping first and last N items.

    Returns number of items removed.
    """
    if len(group) <= keep_first + keep_last:
        return 0

    to_remove = group[keep_first:-keep_last] if keep_last > 0 else group[keep_first:]

    if not to_remove:
        return 0

    # Create placeholder comment - clear for scraping agents
    placeholder_text = (
        f" [TRUNCATED: {len(to_remove)} similar items omitted. "
        f"Showing first {keep_first} and last {keep_last} of {len(group)} total items.] "
    )
    placeholder = Comment(placeholder_text)

    # Insert placeholder before first item to remove
    to_remove[0].insert_before(placeholder)

    # Remove middle items
    for item in to_remove:
        item.decompose()

    return len(to_remove)


def _truncate_all_repeating(
    soup,
    min_items: int = DEFAULT_MIN_REPEATING_ITEMS,
    keep_first: int = DEFAULT_KEEP_FIRST,
    keep_last: int = DEFAULT_KEEP_LAST,
    lookback: int = 3,
    class_threshold: float = 0.5,
    dominance_ratio: float = 1.5,
) -> dict:
    """
    Find and truncate all repeating element groups in the document.

    Uses a multi-strategy approach:
    1. Dominant element detection (handles mixed content with widgets/interstitials)
    2. Semantic detection for HTML containers (ul, ol, table, etc.)
    3. Sliding window for consecutive similar elements

    Processes bottom-up to handle nested lists correctly.

    Args:
        soup: BeautifulSoup document
        min_items: Minimum run length to collapse
        keep_first: Elements to preserve at run start
        keep_last: Elements to preserve at run end
        lookback: Sliding window size for similarity comparison
        class_threshold: Class similarity threshold for _is_similar()
        dominance_ratio: Required ratio for dominant element detection

    Returns:
        Dict with statistics: {
            "runs_found": int,
            "runs_collapsed": int,
            "elements_removed": int,
            "dominant_truncations": int,
            "preserved_adjacent": int,
        }
    """
    stats = {
        "runs_found": 0,
        "runs_collapsed": 0,
        "elements_removed": 0,
        "dominant_truncations": 0,
        "preserved_adjacent": 0,
    }

    # Get all elements, sorted by depth (deepest first for bottom-up processing)
    all_elements = list(soup.find_all(True))
    all_elements.sort(key=lambda e: -len(list(e.parents)))

    processed_parents: set[int] = set()

    for element in all_elements:
        # Skip if already removed or already processed
        if element.parent is None:
            continue
        if id(element) in processed_parents:
            continue

        # Strategy 1: Try dominant element detection first
        # This handles mixed content with widgets/interstitials interspersed
        dom_result = _truncate_dominant_with_adjacency(
            element,
            min_items=min_items,
            keep_first=keep_first,
            keep_last=keep_last,
            class_threshold=class_threshold,
            dominance_ratio=dominance_ratio,
        )

        if dom_result["dominant_found"] and dom_result["elements_removed"] > 0:
            stats["runs_found"] += 1
            stats["runs_collapsed"] += 1
            stats["elements_removed"] += dom_result["elements_removed"]
            stats["dominant_truncations"] += 1
            stats["preserved_adjacent"] += dom_result["preserved_adjacent"]
            processed_parents.add(id(element))
            logger.debug(
                f"Truncated {dom_result['elements_removed']} items from <{element.name}> "
                f"(method: dominant, total: {dom_result['dominant_count']}, "
                f"preserved_adjacent: {dom_result['preserved_adjacent']})"
            )
            continue

        # Strategy 2: Try semantic detection for HTML containers (ul, ol, table, etc.)
        runs = _find_repeating_runs_semantic(element, min_items)
        method = "semantic"

        # Strategy 3: Fall back to sliding window detection for generic containers
        if not runs:
            runs = _find_repeating_runs(
                element, min_items, lookback, class_threshold
            )
            method = "sliding_window"

        # Process each run found
        for run in runs:
            stats["runs_found"] += 1

            if len(run) > keep_first + keep_last:
                removed = _truncate_repeating_group(run, keep_first, keep_last)
                if removed > 0:
                    stats["runs_collapsed"] += 1
                    stats["elements_removed"] += removed
                    processed_parents.add(id(element))
                    logger.debug(
                        f"Truncated {removed} items from <{element.name}> "
                        f"(method: {method}, total: {len(run)}, "
                        f"kept: {keep_first}+{keep_last})"
                    )

    return stats


def _format_html(
    html: str,
    remove_boilerplate: bool = False,
    truncate_text: bool = False,
    max_text_length: int = 200,
    limit_classes: bool = False,
    max_classes: int = 10,
    remove_empty: bool = False,
    remove_head: bool = True,
    prettify: bool = True,
    use_boilerplate_patterns: bool = False,
    truncate_repeating: bool = True,  # Always enabled by default
    min_repeating_items: int = DEFAULT_MIN_REPEATING_ITEMS,
    keep_first_items: int = DEFAULT_KEEP_FIRST,
    keep_last_items: int = DEFAULT_KEEP_LAST,
) -> str:
    """
    Sanitize HTML for web scraping by removing unnecessary elements and attributes.

    Args:
        html: The raw HTML string to process.
        remove_boilerplate: If True, removes common boilerplate sections
                           (nav, footer, sidebar, ads, etc.). Use with caution
                           as it may remove desired content.
        truncate_text: If True, truncates long text content to max_text_length.
        max_text_length: Maximum length for text content when truncate_text is True.
        limit_classes: If True, limits the number of classes per element.
        max_classes: Maximum number of classes per element when limit_classes is True.
        remove_empty: If True, removes elements that are empty and have no
                     meaningful attributes.
        remove_head: If True, removes the entire <head> section.
        prettify: If True, returns prettified HTML; otherwise returns minified.
        use_boilerplate_patterns: If True, uses regex patterns to identify boilerplate.
        truncate_repeating: If True, truncates repeating elements (lists, tables, etc.)
                           keeping only the first and last N items.
        min_repeating_items: Minimum items needed before truncation applies.
        keep_first_items: Number of items to keep at the start of a repeating group.
        keep_last_items: Number of items to keep at the end of a repeating group.

    Returns:
        Sanitized HTML string.
    """
    start_time = time.perf_counter()
    original_len = len(html)

    logger.debug(
        f"Starting HTML sanitization: {original_len:,} chars",
        extra={
            "original_length": original_len,
            "remove_boilerplate": remove_boilerplate,
            "truncate_text": truncate_text,
            "remove_empty": remove_empty,
        },
    )

    soup = BeautifulSoup(html, "lxml")
    
    # Remove comments
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()
    
    # Remove head section if requested
    if remove_head:
        head = soup.find("head")
        if head:
            head.decompose()
    
    # Remove unwanted elements
    for tag_name in ELEMENTS_TO_REMOVE:
        for element in soup.find_all(tag_name):
            element.decompose()

    # Empty elements that should keep their shell but lose contents
    for tag_name in ELEMENTS_TO_EMPTY:
        for element in soup.find_all(tag_name):
            # Keep the element and its attributes, but clear contents
            element.clear()
    
    # Remove boilerplate sections if requested
    if remove_boilerplate:
        # We need to collect elements first, then remove them
        # to avoid modifying the tree while iterating
        to_remove = []
        for element in soup.find_all(True):  # All tags
            # check if it has class "adx-disclosure-card"
           
            if _is_boilerplate_element(element, use_boilerplate_patterns=use_boilerplate_patterns):
                to_remove.append(element)
        
        for element in to_remove:
            # Only decompose if it hasn't been removed already (nested boilerplate)
            if element.parent is not None:
                element.decompose()

    # Truncate repeating elements if requested
    if truncate_repeating:
        size_before_truncation = len(str(soup))
        truncate_stats = _truncate_all_repeating(
            soup,
            min_items=min_repeating_items,
            keep_first=keep_first_items,
            keep_last=keep_last_items,
        )
        if truncate_stats["elements_removed"] > 0:
            size_after_truncation = len(str(soup))
            reduction = size_before_truncation - size_after_truncation
            reduction_pct = (reduction / size_before_truncation) * 100 if size_before_truncation > 0 else 0
            logger.info(
                f"Repeating elements truncation: "
                f"{truncate_stats['runs_collapsed']}/{truncate_stats['runs_found']} runs collapsed, "
                f"{truncate_stats['elements_removed']} elements removed, "
                f"{size_before_truncation:,} -> {size_after_truncation:,} chars "
                f"({reduction_pct:.1f}% reduction)"
            )

    # Process all remaining elements
    for element in soup.find_all(True):
        # Remove unwanted attributes
        attrs_to_delete = []
        for attr in list(element.attrs.keys()):
            if _should_remove_attribute(attr):
                attrs_to_delete.append(attr)
        
        for attr in attrs_to_delete:
            del element[attr]

        # Truncate long attribute values
        _truncate_long_attributes(element)

        # Limit classes if requested
        if limit_classes:
            _limit_class_count(element, max_classes)
        
        # Truncate text if requested
        if truncate_text:
            _truncate_text(element, max_text_length)
    
    # Remove empty elements if requested
    if remove_empty:
        # Multiple passes may be needed as removing children may make parents empty
        changed = True
        max_passes = 5  # Prevent infinite loops
        passes = 0
        
        while changed and passes < max_passes:
            changed = False
            passes += 1
            
            for element in soup.find_all(True):
                if element.name not in {"html", "body"} and _is_empty_element(element):
                    element.decompose()
                    changed = True
    
    # Return result
    if prettify:
        result = soup.prettify()
    else:
        # Minified: collapse whitespace
        html_str = str(soup)
        # Collapse multiple whitespace into single space
        html_str = re.sub(r"\s+", " ", html_str)
        # Remove space around tags
        html_str = re.sub(r">\s+<", "><", html_str)
        result = html_str.strip()

    # Clean unstable CSS classes as final step
    size_before_class_clean = len(result)
    result = clean_unstable_classes(result)
    size_after_class_clean = len(result)

    if size_before_class_clean != size_after_class_clean:
        class_clean_reduction = size_before_class_clean - size_after_class_clean
        class_clean_pct = (class_clean_reduction / size_before_class_clean) * 100 if size_before_class_clean > 0 else 0
        logger.debug(
            f"Class cleaning: {size_before_class_clean:,} -> {size_after_class_clean:,} chars "
            f"({class_clean_pct:.1f}% reduction)"
        )

    duration_ms = (time.perf_counter() - start_time) * 1000
    result_len = len(result)
    reduction_pct = (1 - result_len / original_len) * 100 if original_len > 0 else 0

    logger.debug(
        f"HTML sanitization complete: {original_len:,} -> {result_len:,} chars "
        f"({reduction_pct:.1f}% reduction, {duration_ms:.1f}ms)",
        extra={
            "original_length": original_len,
            "result_length": result_len,
            "reduction_percent": reduction_pct,
            "duration_ms": duration_ms,
        },
    )

    return result


def format_html_aggressive(html: str) -> str:
    """
    Aggressively sanitize HTML, removing as much as possible while preserving
    scraping capability. Use this for very large documents.
    """
    logger.info(f"Aggressive HTML sanitization: {len(html):,} chars input")
    return _format_html(
        html,
        remove_boilerplate=True,
        truncate_text=True,
        max_text_length=100,
        limit_classes=True,
        max_classes=5,
        remove_empty=True,
        remove_head=True,
        prettify=False,
        use_boilerplate_patterns=False,
        # Truncate repeating elements (lists, tables, etc.)
        truncate_repeating=True,
        min_repeating_items=6,
        keep_first_items=3,
        keep_last_items=3,
    )


def format_html_conservative(html: str, truncate_repeating=True) -> str:
    """
    Conservatively sanitize HTML, removing only clearly unnecessary content.
    Use this when you're unsure what content is needed.
    """
    logger.info(f"Conservative HTML sanitization: {len(html):,} chars input")
    return _format_html(
        html,
        remove_boilerplate=False,
        truncate_text=False,
        limit_classes=False,
        remove_empty=False,
        remove_head=True,
        prettify=True,
        # Truncate repeating elements (lists, tables, etc.)
        truncate_repeating=truncate_repeating,
        min_repeating_items=6,
        keep_first_items=3,
        keep_last_items=3,
    )


# Example usage and testing
if __name__ == "__main__":
    import sys

    # Test with inline HTML - simple case (no widgets)
    test_html_simple = """
    <html>
    <body>
        <div class="product-list">
            <div class="product-card"><h3>Product 1</h3><p>$10</p></div>
            <div class="product-card"><h3>Product 2</h3><p>$20</p></div>
            <div class="product-card"><h3>Product 3</h3><p>$30</p></div>
            <div class="product-card"><h3>Product 4</h3><p>$40</p></div>
            <div class="product-card"><h3>Product 5</h3><p>$50</p></div>
            <div class="product-card"><h3>Product 6</h3><p>$60</p></div>
            <div class="product-card"><h3>Product 7</h3><p>$70</p></div>
            <div class="product-card"><h3>Product 8</h3><p>$80</p></div>
            <div class="product-card"><h3>Product 9</h3><p>$90</p></div>
            <div class="product-card"><h3>Product 10</h3><p>$100</p></div>
        </div>
    </body>
    </html>
    """

    # Test with mixed content - products with widgets interspersed (Amazon-like)
    test_html_mixed = """
    <html>
    <body>
        <div class="search-results">
            <div class="header-widget"><span>Showing results for "headphones"</span></div>
            <div class="product-card s-asin"><h3>Product 1</h3><p>$10</p><span>Rating: 4.5</span></div>
            <div class="product-card s-asin"><h3>Product 2</h3><p>$20</p><span>Rating: 4.2</span></div>
            <div class="product-card s-asin"><h3>Product 3</h3><p>$30</p><span>Rating: 4.8</span></div>
            <div class="product-card s-asin"><h3>Product 4</h3><p>$40</p><span>Rating: 4.1</span></div>
            <div class="product-card s-asin"><h3>Product 5</h3><p>$50</p><span>Rating: 4.7</span></div>
            <div class="promo-widget carousel"><h2>Recently Viewed</h2><ul><li>Item A</li><li>Item B</li></ul></div>
            <div class="product-card s-asin"><h3>Product 6</h3><p>$60</p><span>Rating: 4.3</span></div>
            <div class="product-card s-asin"><h3>Product 7</h3><p>$70</p><span>Rating: 4.6</span></div>
            <div class="product-card s-asin"><h3>Product 8</h3><p>$80</p><span>Rating: 4.4</span></div>
            <div class="ad-widget sponsored"><span>Sponsored Ad</span></div>
            <div class="product-card s-asin"><h3>Product 9</h3><p>$90</p><span>Rating: 4.9</span></div>
            <div class="product-card s-asin"><h3>Product 10</h3><p>$100</p><span>Rating: 4.0</span></div>
            <div class="product-card s-asin"><h3>Product 11</h3><p>$110</p><span>Rating: 4.5</span></div>
            <div class="product-card s-asin"><h3>Product 12</h3><p>$120</p><span>Rating: 4.2</span></div>
        </div>
    </body>
    </html>
    """

    # Try to load file, fall back to test HTML
    try:
        with open("app/domains/scraping/agent/generated_scrapers/view.html", "r") as f:
            sample_html = f.read()
        print(f"Loaded HTML file: {len(sample_html):,} characters")
    except FileNotFoundError:
        sample_html = test_html_simple
        print("Using inline test HTML (simple)")

    print("\n" + "=" * 60)
    print("TESTING DOMINANT ELEMENT DETECTION WITH MIXED CONTENT")
    print("=" * 60)

    # Test the dominant element detection with mixed content
    soup_mixed = BeautifulSoup(test_html_mixed, "lxml")
    search_results = soup_mixed.find("div", class_="search-results")

    if search_results:
        print("\nBefore truncation:")
        children = [c for c in search_results.children if hasattr(c, "name") and c.name]
        for i, child in enumerate(children):
            classes = " ".join(child.get("class", []))
            text = child.get_text(strip=True)[:50]
            print(f"  [{i}] <{child.name} class='{classes}'> {text}...")

        # Test grouping
        groups = _group_children_by_similarity(children)
        print(f"\nGrouped into {len(groups)} groups:")
        for i, group in enumerate(sorted(groups, key=len, reverse=True)):
            classes = " ".join(group[0].get("class", []))
            print(f"  Group {i}: {len(group)} elements - <{group[0].name} class='{classes}'>")

        # Test dominant detection
        dominant = _find_dominant_group(groups, min_items=6)
        if dominant:
            print(f"\nDominant group: {len(dominant)} elements")

            # Test adjacency preservation
            preserve = _get_indices_to_preserve(children, dominant, keep_first=3, keep_last=3)
            print(f"Indices to preserve: {sorted(preserve)}")

            # Test full truncation
            result = _truncate_dominant_with_adjacency(search_results, min_items=6)
            print(f"\nTruncation result: {result}")

        print("\nAfter truncation:")
        children_after = [c for c in search_results.children if hasattr(c, "name") and c.name]
        for i, child in enumerate(children_after):
            classes = " ".join(child.get("class", []))
            text = child.get_text(strip=True)[:50] if hasattr(child, "get_text") else str(child)[:50]
            print(f"  [{i}] <{child.name} class='{classes}'> {text}...")

    print("\n" + "=" * 60)
    print("TESTING SLIDING WINDOW REPEATING ELEMENT DETECTION")
    print("=" * 60)

    # Test the sliding window detection directly
    soup = BeautifulSoup(sample_html, "lxml")

    # Find all parents and test detection
    for parent in soup.find_all(["div", "ul", "ol", "tbody", "table"]):
        runs = _find_repeating_runs(parent, min_run=5)
        if runs:
            for run in runs:
                print(f"Found run in <{parent.name}>: {len(run)} similar elements")
                print(f"  First element: <{run[0].name} class='{run[0].get('class', [])}'>")

    print("\n" + "=" * 60)
    print("CONSERVATIVE SANITIZATION (with truncation)")
    print("=" * 60)
    sanitized_html = format_html_conservative(sample_html)

    print("\n" + "=" * 60)
    print("AGGRESSIVE SANITIZATION (with truncation)")
    print("=" * 60)
    sanitized_html_aggressive = format_html_aggressive(sample_html)

    print("\n" + "=" * 60)
    print("LENGTH COMPARISON")
    print("=" * 60)
    print(f"Original:     {len(sample_html):,} characters")
    print(f"Conservative: {len(sanitized_html):,} characters ({len(sanitized_html) / len(sample_html) * 100:.1f}% of original)")
    print(f"Aggressive:   {len(sanitized_html_aggressive):,} characters ({len(sanitized_html_aggressive) / len(sample_html) * 100:.1f}% of original)")

    # Show truncated result for visual inspection
    print("\n" + "=" * 60)
    print("TRUNCATED OUTPUT (first 2000 chars)")
    print("=" * 60)
    print(sanitized_html_aggressive[:2000])




"""

## Smart HTML Sanitization Pipeline

We want to add two preprocessing steps before the main analyzer. The goal is to reduce the HTML to only the elements that contain the target data and their functionally related elements.

**Why this is needed:** Even after applying the HTML sanitizer, the document may still be too large. However, using aggressive sanitization risks removing data we need. These two smart sanitization steps solve this by intelligently identifying and extracting only the relevant portions of the HTML.

### Step 1: Sanitization Analyzer

The Sanitization Analyzer performs the following tasks:

1. **Locate the target data** — Identify which elements contain the data we need to scrape.

2. **Identify the container** — Find the parent element that encompasses all the target data elements AND the related elements to the data if available (e.g. pagination, filters, etc.).

3. **Include** — The reduction should not be limited to just the data elements themselves. It must also include functionally related elements, such as:
   - Pagination controls (e.g., "Next Page" buttons)
   - Filters or sorting options
   - Any other UI elements that affect or relate to the data

4. **Document the findings** — The analyzer describes and explains which elements should be preserved. What is the container that includes all the data elements and their related elements? It does not write code.

### Step 2: Sanitization Code Generator

The Sanitization Code Generator takes the analyzer's description and writes BeautifulSoup code that:

1. Extracts the container element identified by the analyzer
2. Returns it for further processing (as html string)

Finally, the next steps (analyzer and code generator) will use the html string returned by the code generator to analyze the structure and generate the code.

### Important Constraints

- **Do not modify the HTML** — No alterations to the existing structure or content.
- **Do not add new elements** — The output must contain only elements from the original HTML.
- **Extract, don't transform** — Simply isolate the container element that holds all important content and its related elements, then return it as-is for the subsequent pipeline steps (the main analyzer and code generator).


"""