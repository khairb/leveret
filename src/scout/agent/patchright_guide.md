
## Patchright API Quick Reference

Patchright's API is identical to Playwright's async Python API, with one addition:
`isolated_context=True` (default) in `evaluate()` — prevents Runtime.enable detection.
Import from `patchright.async_api`, not `playwright.async_api`.
`launch_persistent_context()` returns a **BrowserContext** — close the **context**, not a browser.

### Navigation

```python
await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
# wait_until options: "load", "domcontentloaded", "networkidle", "commit"
# "networkidle" waits until no network activity for 500ms — useful for SPAs,
# but can hang on sites with persistent connections (analytics, websockets).
# Always pair with a timeout; fall back to waiting for specific elements.
await page.wait_for_load_state("domcontentloaded")
await page.reload()
page.url  # current URL (property, no await)
await page.wait_for_url("**/results**")  # wait for URL to match pattern
```

### JavaScript Evaluation (primary extraction method)

```python
# Return serializable data (strings, numbers, lists, dicts)
results = await page.evaluate("""
    () => Array.from(document.querySelectorAll('.item')).map(el => ({
        text: el.querySelector('h3')?.innerText?.trim() || '',
        href: el.querySelector('a')?.href || '',
        img: el.querySelector('img')?.src || '',
        data: el.dataset.id || '',
    }))
""", isolated_context=True)

# Pass an argument
val = await page.evaluate("(sel) => document.querySelector(sel)?.innerText", ".price", isolated_context=True)
```

Useful JS inside evaluate: `document.querySelector(css)`, `document.querySelectorAll(css)`,
`el.innerText`, `el.textContent`, `el.innerHTML`, `el.getAttribute(name)`, `el.href`, `el.src`,
`el.dataset.x`, `el.closest(css)`, `el.querySelector(css)`, `Array.from(nodeList)`,
`JSON.parse(el.textContent)` (for script tags), `window.scrollTo(0, document.body.scrollHeight)`.

### Waiting

```python
await page.wait_for_selector('.items', state='visible', timeout=30_000)
# state options: "visible", "hidden", "attached", "detached"

await page.wait_for_function("document.querySelectorAll('.item').length > 0", timeout=30_000)
# Pass arguments and control polling interval:
await page.wait_for_function(
    "(n) => document.querySelectorAll('.item').length >= n", arg=10, polling=500
)
await page.wait_for_timeout(5_000)  # hard wait, use sparingly
```

### Content Extraction (convenience methods)

```python
text = await page.inner_text('.selector')             # visible text
attr = await page.get_attribute('a.link', 'href')     # attribute value
txt  = await page.text_content('.selector')            # all text including hidden
```

### Locators

```python
loc = page.locator('.card')                           # CSS or XPath
loc = page.get_by_text("Load More")                   # by visible text
loc = page.get_by_role("button", name="Submit")       # by ARIA role

await loc.count()                                     # number of matches
await loc.all()                                       # list of individual locators
await loc.inner_text()                                # text (single element)
await loc.all_inner_texts()                           # list of texts (all matches)
await loc.get_attribute('href')                       # attribute
await loc.click()                                     # click
loc.first / loc.last / loc.nth(2)                     # pick specific match
loc.locator('.child')                                 # scoped sub-search
loc.filter(has_text="Sale")                           # filter matches
loc.or_(page.locator('.alt'))                         # match either locator

# Wait for a locator's element to reach a state (essential for SPA transitions)
await loc.wait_for(state="visible", timeout=15_000)
await page.locator(".spinner").wait_for(state="hidden")   # wait for loading to finish
# states: "attached", "detached", "visible", "hidden"

# Run JS on all matching elements at once
data = await page.locator('.card').evaluate_all("""
    els => els.map(el => ({ name: el.querySelector('h3')?.innerText || '' }))
""")
```

### ElementHandle (query_selector)

```python
el = await page.query_selector('.item')               # single element or None
els = await page.query_selector_all('.item')           # list of elements
# On an element handle:
await el.inner_text()
await el.get_attribute('href')
child = await el.query_selector('.sub')                # scoped search
```

### User Interaction

```python
await page.click('button.load-more')
await page.fill('input[name="q"]', 'search term')
await page.keyboard.press('Enter')
await page.select_option('select#sort', value='price')
await page.hover('.menu-trigger')
await page.mouse.wheel(0, 3000)                       # scroll via mouse wheel
await page.evaluate("window.scrollTo(0, document.body.scrollHeight)", isolated_context=True)
```

### Network — Capture API Responses

```python
# Passive listener — capture JSON from background XHR/fetch
captured = []
async def on_response(response):
    if "/api/products" in response.url and response.ok:
        captured.append(await response.json())
page.on("response", on_response)
await page.goto(url, wait_until="domcontentloaded")

# Wait for a specific response after an action
async with page.expect_response(lambda r: "/api/data" in r.url) as resp_info:
    await page.click('#load-btn')
response = await resp_info.value
data = await response.json()

# Wait for navigation triggered by a click (prevents race conditions)
async with page.expect_navigation(wait_until="domcontentloaded"):
    await page.click("a.next-page")

# Response object: response.url, response.status, response.ok,
#   await response.json(), await response.text(), await response.body()
```

### Network — Block Resources (speed up scraping)

```python
await page.route("**/*.{png,jpg,jpeg,gif,svg,css,woff,woff2}", lambda route: route.abort())
# Or by resource type:
async def block(route):
    if route.request.resource_type in ("image", "stylesheet", "font", "media"):
        await route.abort()
    else:
        await route.continue_()
await page.route("**/*", block)
```

### Overlay & Popup Handling

```python
# Auto-dismiss overlays (cookie banners, popups) whenever they appear.
# Registers a handler that triggers any time the element is detected.
await page.add_locator_handler(
    page.get_by_role("button", name="Accept Cookies"),
    handler=lambda loc: loc.click(),
    times=1  # trigger once (None = unlimited)
)

# Handle JS dialogs (alert/confirm/prompt) — unhandled dialogs freeze the page
page.on("dialog", lambda dialog: dialog.accept())
```

### Frames (iframes)

Consent banners and cookie dialogs are often inside iframes.

```python
frame = page.frame_locator('iframe#content')
text = await frame.locator('.data').inner_text()

# Consent banner in an iframe
consent = page.frame_locator("iframe[src*='consent']")
await consent.get_by_role("button", name="Accept").click()

# Bridge: locator → frame content
frame = page.locator("iframe#consent").content_frame
await frame.get_by_role("button", name="Accept").click()
```

### Common Patterns

**Scroll to load all items before extracting:**
```python
# For lists/grids: scroll first so all lazy-loaded items appear, then extract.
# For pagination: scroll → extract → next page → scroll → extract → repeat.
await scroll_to_bottom(page, max_scrolls=50)
# Now extract - all items are loaded
```

**Pagination via button:**
```python
while True:
    # ... extract current page ...
    nxt = page.locator('a.next')
    if await nxt.count() == 0: break
    await nxt.click()
    await page.wait_for_load_state("domcontentloaded")
```

**SPA pagination (page doesn't reload):**
```python
while True:
    await page.locator('.listing').first.wait_for(state="visible")
    # ... extract current page ...
    nxt = page.locator('a[aria-label="Next"]')
    if await nxt.count() == 0: break
    old_text = await page.locator('.listing').first.inner_text()
    await nxt.click()
    # Wait for content to actually change (old element replaced)
    await page.wait_for_function(
        f"document.querySelector('.listing')?.innerText !== `{old_text}`",
        timeout=15_000
    )
```

**JSON-LD from script tags:**
```python
data = await page.evaluate("""
    () => Array.from(document.querySelectorAll('script[type="application/ld+json"]'))
           .map(s => JSON.parse(s.textContent))
""", isolated_context=True)
```
