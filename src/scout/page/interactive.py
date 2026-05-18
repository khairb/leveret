"""Interactive Element Detection

Detects all interactive elements on a live web page controlled by
Patchright (undetectable Playwright fork). Stamps each detected element in the browser DOM
with a ``data-iid`` attribute and hidden subtree roots with ``data-hidden``.

Usage::

    elements = await detect_interactive_elements(page)
    html = await page.content()          # HTML now contains data-iid markers
    # ... pass html + elements downstream ...
    clean = cleanup_markers(html)         # strip markers when needed

Detection layers (applied in order):

    1. Native HTML interactive elements  (a[href], button, input, …)
    2. ARIA roles and attributes         (role="button", aria-expanded, …)
    4. Playwright accessibility tree     (catch-all for browser-inferred roles)
    5. CSS ``cursor: pointer`` heuristic
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Data Structures
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class InteractiveElement:
    """Metadata for a single interactive element detected on the page.

    The *iid* field matches the ``data-iid`` attribute stamped in the DOM,
    providing a reliable bridge between browser state and any HTML parser.
    """

    iid: int  # matches data-iid="N" in the DOM
    tag: str  # "a", "button", "input", …
    attributes: dict[str, str]  # relevant attrs (href, name, role, …)
    text: str  # visible text content (≤ 500 chars)
    selector: str  # CSS selector to re-locate the element
    bounding_box: dict[str, float]  # {x, y, width, height}
    detected_by: list[str] = field(default_factory=list)


# Roles the accessibility-tree layer (4) considers interactive.
_AX_INTERACTIVE_ROLES = frozenset(
    {
        "button",
        "link",
        "tab",
        "menuitem",
        "option",
        "checkbox",
        "radio",
        "slider",
        "switch",
        "combobox",
        "searchbox",
        "textbox",
        "treeitem",
        "menuitemcheckbox",
        "menuitemradio",
        "spinbutton",
    }
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  JavaScript — runs in the browser via page.evaluate()
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


_CLEAR_MARKERS_JS = """() => {
    for (const el of document.querySelectorAll('[data-iid]'))
        el.removeAttribute('data-iid');
    for (const el of document.querySelectorAll('[data-hidden]'))
        el.removeAttribute('data-hidden');
}"""


_DETECTION_JS = """(stampHidden) => {
    // ── Configuration ──────────────────────────────────────────────

    const NATIVE_SELECTORS = [
        'a[href]', 'button', 'input', 'select', 'textarea',
        '[contenteditable="true"]', 'details', 'summary', 'label'
    ];

    const ARIA_ROLES = new Set([
        'button', 'link', 'tab', 'menuitem', 'option', 'checkbox',
        'radio', 'slider', 'switch', 'combobox', 'searchbox',
        'textbox', 'treeitem'
    ]);

    const ARIA_ATTRS = [
        'aria-expanded', 'aria-haspopup', 'aria-pressed', 'aria-checked'
    ];

    const SKIP_ATTRS = new Set(['class', 'style', 'data-iid', 'data-hidden']);
    const SKIP_TAGS  = new Set(['script', 'style', 'noscript', 'template']);

    // ── State ──────────────────────────────────────────────────────

    const seen    = new Set();   // elements already registered
    const results = [];
    let   nextIid = 1;

    // ── Helpers ────────────────────────────────────────────────────

    /** True when the element is visible to the user. */
    function isVisible(el) {
        try {
            // checkVisibility accounts for inherited display:none / visibility:hidden
            if (typeof el.checkVisibility === 'function') {
                if (!el.checkVisibility({checkVisibilityCSS: true})) return false;
            } else {
                const s = getComputedStyle(el);
                if (s.display === 'none' || s.visibility === 'hidden') return false;
            }
            const r = el.getBoundingClientRect();
            if (r.width === 0 && r.height === 0) return false;
            // Generous off-screen margin (2× viewport) — scrollable elements are kept
            const m = Math.max(window.innerWidth, window.innerHeight) * 2;
            if (r.bottom < -m || r.top  > window.innerHeight + m) return false;
            if (r.right  < -m || r.left > window.innerWidth  + m) return false;
            return true;
        } catch (_) {
            return false;
        }
    }

    function isDisabled(el) {
        return el.disabled === true
            || el.getAttribute('aria-disabled') === 'true';
    }

    /** Collect all attributes except presentational / internal ones. */
    function collectAttrs(el) {
        const out = {};
        for (const a of el.attributes) {
            if (!SKIP_ATTRS.has(a.name)) out[a.name] = a.value;
        }
        return out;
    }

    function getText(el) {
        return (el.textContent || '').trim().substring(0, 500);
    }

    /**
     * Build a CSS selector that uniquely identifies *el* on the page.
     * Priority: id → data-testid → name → tag+attr → positional path.
     */
    function makeSelector(el) {
        // 1. id
        if (el.id) {
            const s = '#' + CSS.escape(el.id);
            try { if (document.querySelectorAll(s).length === 1) return s; }
            catch (_) { /* malformed id — fall through */ }
        }
        // 2. data-testid
        const tid = el.getAttribute('data-testid');
        if (tid) {
            const s = '[data-testid="' + CSS.escape(tid) + '"]';
            try { if (document.querySelectorAll(s).length === 1) return s; }
            catch (_) {}
        }
        // 3. name attribute (scoped to tag)
        const nameAttr = el.getAttribute('name');
        if (nameAttr) {
            const tag = el.tagName.toLowerCase();
            const s = tag + '[name="' + CSS.escape(nameAttr) + '"]';
            try { if (document.querySelectorAll(s).length === 1) return s; }
            catch (_) {}
        }
        // 4. tag + a single distinctive attribute
        const tag = el.tagName.toLowerCase();
        for (const attr of ['href','type','role','aria-label','placeholder','action']) {
            const v = el.getAttribute(attr);
            if (v) {
                const s = tag + '[' + attr + '="' + CSS.escape(v) + '"]';
                try { if (document.querySelectorAll(s).length === 1) return s; }
                catch (_) {}
            }
        }
        // 5. Positional path from <body>
        const parts = [];
        let cur = el;
        while (cur && cur !== document.body && cur !== document.documentElement) {
            const p = cur.parentElement;
            if (!p) break;
            const t = cur.tagName.toLowerCase();
            const sibs = Array.from(p.children).filter(c => c.tagName === cur.tagName);
            parts.unshift(
                sibs.length === 1
                    ? t
                    : t + ':nth-of-type(' + (sibs.indexOf(cur) + 1) + ')'
            );
            cur = p;
        }
        return parts.join(' > ') || tag;
    }

    // ── Core: register an interactive element ──────────────────────

    function register(el, layer) {
        if (seen.has(el)) {
            // Already found by an earlier layer — record the extra source
            const iid = parseInt(el.getAttribute('data-iid'));
            const r = results.find(r => r.iid === iid);
            if (r && !r.detected_by.includes(layer)) r.detected_by.push(layer);
            return;
        }
        if (!isVisible(el) || isDisabled(el)) return;

        const iid = nextIid++;
        seen.add(el);
        el.setAttribute('data-iid', String(iid));

        const rect = el.getBoundingClientRect();
        results.push({
            iid,
            tag:          el.tagName.toLowerCase(),
            attributes:   collectAttrs(el),
            text:         getText(el),
            selector:     makeSelector(el),
            bounding_box: {
                x:      Math.round(rect.x),
                y:      Math.round(rect.y),
                width:  Math.round(rect.width),
                height: Math.round(rect.height)
            },
            detected_by: [layer]
        });
    }

    // ── Layer 1 — Native HTML interactive elements ─────────────────

    for (const sel of NATIVE_SELECTORS) {
        for (const el of document.querySelectorAll(sel))
            register(el, 'layer1_native');
    }
    // tabindex ≥ 0 makes any element focusable / interactive
    for (const el of document.querySelectorAll('[tabindex]')) {
        const v = parseInt(el.getAttribute('tabindex'));
        if (!isNaN(v) && v >= 0) register(el, 'layer1_native');
    }

    // ── Layer 2 — ARIA roles and attributes ────────────────────────

    for (const el of document.querySelectorAll('[role]')) {
        if (ARIA_ROLES.has(el.getAttribute('role')))
            register(el, 'layer2_aria');
    }
    const ariaSelector = ARIA_ATTRS.map(a => '[' + a + ']').join(',');
    for (const el of document.querySelectorAll(ariaSelector)) {
        register(el, 'layer2_aria');
    }

    // ── Layer 5 — cursor:pointer heuristic ─────────────────────────

    if (document.body) {
        for (const el of document.body.querySelectorAll('*')) {
            if (seen.has(el)) continue;
            if (SKIP_TAGS.has(el.tagName.toLowerCase())) continue;
            try {
                if (getComputedStyle(el).cursor === 'pointer')
                    register(el, 'layer5_cursor');
            } catch (_) {}
        }
    }

    // ── Stamp hidden subtree roots ─────────────────────────────────
    //
    // Mark the *topmost* element that causes hiding with
    // data-hidden="true".  Downstream converters can skip the entire
    // subtree when they encounter this attribute.
    //
    // Gated by the stampHidden parameter — many modern SPAs (e.g.
    // Airbnb) use CSS visibility tricks that cause false positives.

    if (stampHidden && document.body) {
        for (const el of document.body.querySelectorAll('*')) {
            if (SKIP_TAGS.has(el.tagName.toLowerCase())) continue;

            // Is this element hidden?
            let hidden;
            if (typeof el.checkVisibility === 'function') {
                hidden = !el.checkVisibility({checkVisibilityCSS: true});
            } else {
                const s = getComputedStyle(el);
                hidden = s.display === 'none' || s.visibility === 'hidden';
            }
            if (!hidden) continue;

            // Only stamp if the *parent* is visible (= this is the hide-root)
            const parent = el.parentElement;
            if (parent && parent !== document.body
                       && parent !== document.documentElement) {
                let parentHidden;
                if (typeof parent.checkVisibility === 'function') {
                    parentHidden = !parent.checkVisibility({checkVisibilityCSS: true});
                } else {
                    const ps = getComputedStyle(parent);
                    parentHidden = ps.display === 'none' || ps.visibility === 'hidden';
                }
                if (parentHidden) continue;   // parent already hidden — not the root
            }

            el.setAttribute('data-hidden', 'true');
        }
    }

    return {results, nextIid};
}"""


# JS cleanup pass: remove data-iid from children of already-interactive
# parents when the child has no independent interactivity.
# Returns list of iids that were removed so Python can drop them from results.
_CLEANUP_REDUNDANT_JS = """(layerInfo) => {
    // layerInfo is {iid: {detected_by: [...], tag: "..."}, ...}
    // Natively interactive tags that are independently actionable even inside
    // an interactive parent.
    const NATIVE_INTERACTIVE = new Set([
        'a', 'button', 'input', 'select', 'textarea', 'details', 'summary'
    ]);

    // Layers that indicate independent semantic interactivity
    const SEMANTIC_LAYERS = new Set(['layer1_native', 'layer2_aria']);

    const removed = [];
    const stamped = document.querySelectorAll('[data-iid]');

    for (const el of stamped) {
        const iid = el.getAttribute('data-iid');
        const info = layerInfo[iid];
        if (!info) continue;

        // Check if any ancestor is also stamped
        let ancestor = el.parentElement;
        let hasStampedAncestor = false;
        while (ancestor) {
            if (ancestor.getAttribute('data-iid') != null) {
                hasStampedAncestor = true;
                break;
            }
            ancestor = ancestor.parentElement;
        }
        if (!hasStampedAncestor) continue;

        // This element is inside an already-interactive parent.
        // Keep it only if it has independent interactivity:
        // 1. It's a natively interactive tag
        const tag = el.tagName.toLowerCase();
        if (NATIVE_INTERACTIVE.has(tag)) continue;

        // 2. It has an explicit interactive ARIA role
        const role = el.getAttribute('role');
        if (role && ['button','link','tab','menuitem','option','checkbox',
                     'radio','slider','switch','combobox','searchbox',
                     'textbox','treeitem','spinbutton',
                     'menuitemcheckbox','menuitemradio'].includes(role)) continue;

        // 3. It was detected by a semantic layer (layer1 or layer2)
        const layers = info.detected_by;
        const hasSemantic = layers.some(l => SEMANTIC_LAYERS.has(l));
        if (hasSemantic) continue;

        // Not independently interactive — remove the stamp
        el.removeAttribute('data-iid');
        removed.push(parseInt(iid));
    }
    return removed;
}"""


# JS passed to locator.evaluate_all() for Layer 4.  Receives all elements
# matched by get_by_role() for a single role as an array.  Filters out
# already-stamped elements, applies visibility/disabled checks, stamps
# new ones, and returns their info.  This is the same logic as the old
# per-element _STAMP_SINGLE_JS, but batched over all elements at once.
_BATCH_STAMP_JS = """(elements, args) => {
    const nextIid = args.nextIid;
    const maxNew  = args.maxNew;

    const SKIP_ATTRS = new Set(['class', 'style', 'data-iid', 'data-hidden']);
    const results = [];
    let currentIid = nextIid;

    for (const el of elements) {
        if (results.length >= maxNew) break;

        try {
            // Already stamped by an earlier layer or earlier role iteration
            if (el.getAttribute('data-iid') != null) continue;

            // Visibility: CSS visibility + zero-size only (no position filter).
            // This matches what the accessibility tree includes.
            if (typeof el.checkVisibility === 'function') {
                if (!el.checkVisibility({checkVisibilityCSS: true})) continue;
            }
            const r = el.getBoundingClientRect();
            if (r.width === 0 && r.height === 0) continue;

            // Disabled check
            if (el.disabled === true
                || el.getAttribute('aria-disabled') === 'true') continue;

            // Stamp
            el.setAttribute('data-iid', String(currentIid));

            const rect = el.getBoundingClientRect();
            const attrs = {};
            for (const a of el.attributes) {
                if (!SKIP_ATTRS.has(a.name)) attrs[a.name] = a.value;
            }

            // Unique selector
            let selector = el.tagName.toLowerCase();
            if (el.id) {
                const s = '#' + CSS.escape(el.id);
                try { if (document.querySelectorAll(s).length === 1) selector = s; }
                catch(_) {}
            } else {
                for (const attr of ['name','href','type','role','aria-label',
                                     'placeholder','data-testid']) {
                    const v = el.getAttribute(attr);
                    if (v) {
                        const s = el.tagName.toLowerCase()
                                  + '[' + attr + '="' + CSS.escape(v) + '"]';
                        try {
                            if (document.querySelectorAll(s).length === 1) {
                                selector = s;
                                break;
                            }
                        } catch(_) {}
                    }
                }
            }

            results.push({
                iid:          currentIid,
                tag:          el.tagName.toLowerCase(),
                attributes:   attrs,
                text:         (el.textContent || '').trim().substring(0, 500),
                selector,
                bounding_box: {
                    x:      Math.round(rect.x),
                    y:      Math.round(rect.y),
                    width:  Math.round(rect.width),
                    height: Math.round(rect.height)
                }
            });

            currentIid++;
        } catch (_) {
            continue;
        }
    }

    return {results, nextIid: currentIid};
}"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Public API
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def detect_interactive_elements(
    page,
    *,
    stamp_hidden: bool = False,
) -> list[InteractiveElement]:
    """Detect all interactive elements on the current page.

    **Side-effects on the live DOM:**

    * ``data-iid="N"`` stamped on each interactive element.
    * ``data-hidden="true"`` stamped on the root of every hidden subtree
      (only when *stamp_hidden* is ``True``).

    After calling this function, use ``await page.content()`` to obtain
    the HTML string with markers baked in.

    Args:
        page: A Patchright *Page* object that has already been navigated
              to the target URL.
        stamp_hidden: Whether to detect and stamp hidden subtree roots
            with ``data-hidden="true"``.  Defaults to ``False`` because
            many modern SPAs use CSS visibility tricks that cause false
            positives (e.g. Airbnb marks most content as hidden).

    Returns:
        A list of :class:`InteractiveElement` dataclasses — one per
        detected element, ordered by ``iid`` (which equals DOM order).
    """
    # ── 0. Clear markers from any previous detection run ────────────
    await page.evaluate(_CLEAR_MARKERS_JS, isolated_context=True)

    # ── 1. Layers 1, 2, 5 + hidden stamping  (single browser call) ─
    js_result = await page.evaluate(_DETECTION_JS, stamp_hidden, isolated_context=True)
    raw_elements: list[dict] = js_result["results"]
    next_iid: int = js_result["nextIid"]

    # ── 2. Layer 4 — accessibility tree catch-all ───────────────────
    ax_elements, next_iid = await _detect_via_accessibility_tree(page, next_iid)
    raw_elements.extend(ax_elements)

    # ── 3. Remove redundant children of interactive parents ─────────
    #
    # Elements detected only by Layer 4/5 that sit inside an already-
    # interactive ancestor are not independently actionable.  Strip
    # their data-iid and drop them from results.
    layer_info = {
        str(r["iid"]): {"detected_by": r["detected_by"], "tag": r["tag"]} for r in raw_elements
    }
    removed_iids = await page.evaluate(_CLEANUP_REDUNDANT_JS, layer_info, isolated_context=True)
    removed_set = set(removed_iids) if removed_iids else set()

    # ── 4. Build typed result list ──────────────────────────────────
    return [
        InteractiveElement(
            iid=r["iid"],
            tag=r["tag"],
            attributes=r["attributes"],
            text=r["text"],
            selector=r["selector"],
            bounding_box=r["bounding_box"],
            detected_by=r["detected_by"],
        )
        for r in raw_elements
        if r["iid"] not in removed_set
    ]


def cleanup_markers(html: str) -> str:
    """Remove ``data-iid`` and ``data-hidden`` marker attributes from *html*.

    This is a standalone utility for the orchestrator to call whenever it
    needs a clean copy of the HTML without detection markers.
    """
    html = re.sub(r'\s+data-iid="[^"]*"', "", html)
    html = re.sub(r'\s+data-hidden="[^"]*"', "", html)
    return html


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Layer 4 — Accessibility Roles via get_by_role() (batched)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Roles to scan with Playwright's get_by_role().  These are the ARIA roles
# that indicate interactive elements.  We iterate each role and use
# evaluate_all() to process all matched elements in a single CDP call
# per role (14 calls total instead of 500+).
_ROLES_TO_SCAN = [
    "button",
    "link",
    "tab",
    "menuitem",
    "option",
    "checkbox",
    "radio",
    "slider",
    "switch",
    "combobox",
    "searchbox",
    "textbox",
    "treeitem",
    "spinbutton",
]


async def _detect_via_accessibility_tree(
    page,
    next_iid: int,
    *,
    max_new: int = 50,
) -> tuple[list[dict], int]:
    """Layer 4: find interactive elements the earlier layers missed.

    Uses ``page.get_by_role()`` for each known interactive ARIA role —
    the same browser accessibility tree as before — but batches all
    matched elements per role into a single ``locator.evaluate_all()``
    call.  This reduces CDP round-trips from ~500+ (one per element) to
    14 (one per role).

    Elements already stamped by layers 1/2/5 are skipped.  Caps at
    *max_new* newly stamped elements.
    """
    new_elements: list[dict] = []

    for role in _ROLES_TO_SCAN:
        if len(new_elements) >= max_new:
            break

        try:
            locator = page.get_by_role(role)
            remaining = max_new - len(new_elements)

            result = await locator.evaluate_all(
                _BATCH_STAMP_JS,
                {"nextIid": next_iid, "maxNew": remaining},
                isolated_context=True,
            )

            for info in result["results"]:
                info["detected_by"] = ["layer4_accessibility"]
                new_elements.append(info)

            next_iid = result["nextIid"]

        except Exception:
            continue

    return new_elements, next_iid
