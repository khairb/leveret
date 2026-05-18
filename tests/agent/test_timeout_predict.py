"""Tests for the static timeout prediction algorithm.

Each test documents *why* a particular code pattern should produce
a certain timeout range, making the heuristics auditable.
"""

from __future__ import annotations

import textwrap
import time

import pytest

from scout.agent.timeout_predict import (
    BASELINE,
    MAX_TIMEOUT,
    predict_timeout,
)

# ── Helpers ──────────────────────────────────────────────────────


def _t(code: str) -> float:
    """Shortcut: dedent + predict."""
    return predict_timeout(textwrap.dedent(code).strip())


# ═════════════════════════════════════════════════════════════════
#  1. Baseline / edge cases
# ═════════════════════════════════════════════════════════════════


class TestBaseline:
    """The algorithm should never go below BASELINE."""

    def test_empty_code(self):
        assert predict_timeout("") == BASELINE

    def test_syntax_error(self):
        assert predict_timeout("if if if ???") == BASELINE

    def test_pure_computation(self):
        """No Patchright calls → baseline."""
        assert _t("x = 1 + 2\nprint(x)") == BASELINE

    def test_print_only(self):
        assert _t('print("hello")') == BASELINE

    def test_variable_assignment(self):
        assert _t("items = []\nitems.append(1)") == BASELINE

    def test_comment_only(self):
        assert predict_timeout("# just a comment") == BASELINE


# ═════════════════════════════════════════════════════════════════
#  2. Single Patchright calls
# ═════════════════════════════════════════════════════════════════


class TestSingleCalls:
    def test_goto(self):
        t = _t('await page.goto("https://example.com")')
        # goto(10) * 1.5 = 15 → clamped to BASELINE
        assert t == BASELINE

    def test_evaluate(self):
        t = _t('result = await page.evaluate("() => 42")')
        # evaluate(3) * 1.5 = 4.5 → BASELINE
        assert t == BASELINE

    def test_click(self):
        t = _t("await page.click('button.submit')")
        assert t == BASELINE

    def test_show_page(self):
        """show_page is costed at 25s (observed 20-25s on heavy pages)."""
        t = _t("await show_page(page)")
        # 25 * 1.5 = 37.5 → above BASELINE
        assert t == pytest.approx(37.5)

    def test_wait_for_selector_default(self):
        t = _t("await page.wait_for_selector('.item')")
        # 5s default * 1.5 = 7.5 → BASELINE
        assert t == BASELINE

    def test_wait_for_selector_with_timeout(self):
        """When the agent specifies a large timeout kwarg, respect it."""
        t = _t("await page.wait_for_selector('.item', timeout=60_000)")
        # 60s * 1.5 = 90
        assert t > BASELINE
        assert t == pytest.approx(90.0)

    def test_wait_for_timeout_explicit(self):
        """page.wait_for_timeout(ms) is a hard sleep."""
        t = _t("await page.wait_for_timeout(10_000)")
        # 10s * 1.5 = 15 → BASELINE
        assert t == BASELINE

    def test_asyncio_sleep(self):
        t = _t("await asyncio.sleep(2)")
        assert t == BASELINE


# ═════════════════════════════════════════════════════════════════
#  3. Multiple sequential calls
# ═════════════════════════════════════════════════════════════════


class TestSequentialCalls:
    def test_navigate_wait_extract(self):
        code = """
        await page.goto("https://example.com")
        await page.wait_for_selector('.items', state='visible', timeout=15_000)
        items = await page.evaluate("() => []", isolated_context=True)
        """
        t = _t(code)
        # goto(10) + wait_for_selector(15) + evaluate(3) = 28 * 1.5 = 42
        assert t == pytest.approx(42.0)

    def test_goto_show_page(self):
        code = """
        await page.goto("https://example.com")
        await show_page(page)
        """
        t = _t(code)
        # 10 + 25 = 35 * 1.5 = 52.5
        assert t == pytest.approx(52.5)

    def test_many_operations_exceeds_baseline(self):
        """Enough operations should push well past BASELINE."""
        code = """
        await page.goto("https://example.com")
        await page.wait_for_selector('.a', timeout=10_000)
        await show_page(page)
        await page.click('.btn')
        await page.wait_for_load_state("domcontentloaded")
        await show_page(page)
        """
        t = _t(code)
        # goto(10) + wait(10) + show(25) + click(5) + wait_load(8) + show(25) = 83 * 1.5 = 124.5
        assert t > BASELINE
        assert t == pytest.approx(124.5)

    def test_navigation_chain(self):
        """Multiple navigations add up."""
        code = """
        await page.goto("https://a.com")
        await page.goto("https://b.com")
        await page.goto("https://c.com")
        await page.goto("https://d.com")
        await page.goto("https://e.com")
        await page.goto("https://f.com")
        """
        t = _t(code)
        # 6 * 10 = 60 * 1.5 = 90
        assert t == pytest.approx(90.0)


# ═════════════════════════════════════════════════════════════════
#  4. Loops
# ═════════════════════════════════════════════════════════════════


class TestLoops:
    def test_for_range_literal(self):
        """for i in range(N) — we extract N."""
        code = """
        for i in range(10):
            await page.click('.next')
            await page.wait_for_load_state("domcontentloaded")
        """
        t = _t(code)
        # body = click(5) + wait_load(8) = 13, 10 * 13 = 130 * 1.5 = 195
        assert t == pytest.approx(195.0)

    def test_for_range_start_stop(self):
        """range(2, 8) → 6 iterations."""
        code = """
        for i in range(2, 8):
            await page.click('.item')
        """
        t = _t(code)
        # 6 * click(5) = 30, but await_floor: 1*6*6 = 36 → max(30,36) = 36 * 1.5 = 54
        assert t == pytest.approx(54.0)

    def test_for_unknown_iterable(self):
        """for item in items — defaults to 10 iterations."""
        code = """
        for item in items:
            await page.goto(item)
            await show_page(page)
        """
        t = _t(code)
        # body = goto(10) + show(25) = 35, 10 * 35 = 350 → capped 225 * 1.5 = 337.5
        assert t == pytest.approx(337.5)

    def test_for_range_capped(self):
        """range(1000) is capped at 50 iterations."""
        code = """
        for i in range(1000):
            await page.click('.x')
        """
        t = _t(code)
        # AST: 50 * click(5) = 250 → capped 225.
        # Await: 1*6*50 = 300. max(225, 300) = 300 * 1.5 = 450
        assert t == pytest.approx(450.0)

    def test_while_true_pagination(self):
        """Classic pagination pattern."""
        code = """
        all_items = []
        while True:
            items = await page.evaluate("() => []", isolated_context=True)
            all_items.extend(items)
            nxt = page.locator('a.next')
            if await nxt.count() == 0:
                break
            await nxt.click()
            await page.wait_for_load_state("domcontentloaded")
        """
        t = _t(code)
        # body: evaluate(3) + count(1) + click(5) + wait_load(8) = 17
        # while: 10 * 17 = 170.  await_floor: 4 awaits * 6 * 10 = 240.
        # max(170, 240) = 240 * 1.5 = 360
        assert t == pytest.approx(360.0)

    def test_while_condition(self):
        """while <condition> — same heuristic as while True."""
        code = """
        page_num = 1
        while page_num <= 20:
            await page.click('.next')
            await page.wait_for_load_state("domcontentloaded")
            page_num += 1
        """
        t = _t(code)
        # body: click(5) + wait_load(8) = 13, 10 * 13 = 130 * 1.5 = 195
        assert t == pytest.approx(195.0)

    def test_loop_body_cap(self):
        """Loop contribution is capped at 225s with heavy body."""
        code = """
        while True:
            await page.goto("https://example.com")
            await show_page(page)
            await page.wait_for_selector('.data', timeout=30_000)
        """
        t = _t(code)
        # body: goto(10) + show(25) + wait(30) = 65
        # 10 * 65 = 650 → capped at 225.
        # await_floor: 3 * 6 * 10 = 180. max(225, 180) = 225 * 1.5 = 337.5
        assert t == pytest.approx(337.5)

    def test_nested_loop(self):
        """Nested loop — inner scores multiply with outer iterations."""
        code = """
        for i in range(3):
            for j in range(4):
                await page.click('.item')
        """
        t = _t(code)
        # AST: inner: 4 * click(5) = 20, outer: 3 * 20 = 60
        # Await: 1 await * 6 * 3 * 4 = 72
        # max(60, 72) = 72 * 1.5 = 108
        assert t == pytest.approx(108.0)

    def test_async_for(self):
        """async for — uses default 10 iterations."""
        code = """
        async for item in some_async_generator():
            await page.goto(item.url)
        """
        t = _t(code)
        # AST: 10 * goto(10) = 100. Await: 1*6*10 = 60. max(100,60) = 100 * 1.5 = 150
        assert t == pytest.approx(150.0)


# ═════════════════════════════════════════════════════════════════
#  5. Context managers
# ═════════════════════════════════════════════════════════════════


class TestContextManagers:
    def test_expect_navigation(self):
        code = """
        async with page.expect_navigation(wait_until="domcontentloaded"):
            await page.click("a.next-page")
        """
        t = _t(code)
        # expect_navigation(10) + click(5) = 15 * 1.5 = 22.5 → BASELINE
        assert t == BASELINE

    def test_expect_response(self):
        code = """
        async with page.expect_response(lambda r: "/api/data" in r.url) as resp_info:
            await page.click('#load-btn')
        response = await resp_info.value
        """
        t = _t(code)
        assert t == BASELINE


# ═════════════════════════════════════════════════════════════════
#  6. If/else branches (take the max)
# ═════════════════════════════════════════════════════════════════


class TestBranching:
    def test_if_else_takes_max(self):
        code = """
        if condition:
            await page.goto("https://a.com")
        else:
            await page.goto("https://b.com")
            await show_page(page)
            await page.wait_for_selector('.x', timeout=30_000)
        """
        t = _t(code)
        # else-branch: goto(10) + show(25) + wait(30) = 65
        # max(10, 65) = 65 * 1.5 = 97.5
        assert t == pytest.approx(97.5)


# ═════════════════════════════════════════════════════════════════
#  7. Try/except
# ═════════════════════════════════════════════════════════════════


class TestTryExcept:
    def test_try_except(self):
        code = """
        try:
            await page.goto("https://example.com")
            await page.wait_for_selector('.data', timeout=20_000)
        except Exception:
            await page.reload()
        """
        t = _t(code)
        # try: goto(10) + wait(20) = 30, except: reload(10)
        # max(30, 10) = 30 * 1.5 = 45
        assert t == pytest.approx(45.0)


# ═════════════════════════════════════════════════════════════════
#  8. scroll_to_bottom helper
# ═════════════════════════════════════════════════════════════════


class TestScrollHelper:
    def test_scroll_with_max_scrolls(self):
        code = """
        await scroll_to_bottom(page, max_scrolls=30)
        data = await page.evaluate("() => []", isolated_context=True)
        """
        t = _t(code)
        # scroll(30 * 1 = 30) + evaluate(3) = 33 * 1.5 = 49.5
        assert t == pytest.approx(49.5)

    def test_scroll_default(self):
        code = "await scroll_to_bottom(page)"
        t = _t(code)
        # 15 * 1.5 = 22.5 → BASELINE
        assert t == BASELINE

    def test_scroll_capped(self):
        code = "await scroll_to_bottom(page, max_scrolls=200)"
        t = _t(code)
        # 200 capped to 50 → 50 * 1.5 = 75
        assert t == pytest.approx(75.0)


# ═════════════════════════════════════════════════════════════════
#  9. Timeout kwarg extraction
# ═════════════════════════════════════════════════════════════════


class TestTimeoutKwarg:
    def test_wait_for_function_with_timeout(self):
        code = "await page.wait_for_function('() => true', timeout=45_000)"
        t = _t(code)
        # 45s * 1.5 = 67.5
        assert t == pytest.approx(67.5)

    def test_locator_wait_for_with_timeout(self):
        code = 'await loc.wait_for(state="visible", timeout=20_000)'
        t = _t(code)
        # 20s * 1.5 = 30 → BASELINE
        assert t == BASELINE

    def test_wait_for_selector_small_timeout(self):
        """Small timeout kwarg → uses that value, not the default."""
        code = "await page.wait_for_selector('.x', timeout=2_000)"
        t = _t(code)
        # 2s * 1.5 = 3 → BASELINE
        assert t == BASELINE


# ═════════════════════════════════════════════════════════════════
#  10. Duration argument extraction
# ═════════════════════════════════════════════════════════════════


class TestDurationArg:
    def test_wait_for_timeout_ms(self):
        code = "await page.wait_for_timeout(25_000)"
        t = _t(code)
        # 25s * 1.5 = 37.5
        assert t == pytest.approx(37.5)

    def test_asyncio_sleep_float(self):
        code = "await asyncio.sleep(2.5)"
        t = _t(code)
        assert t == BASELINE

    def test_large_sleep_in_loop(self):
        code = """
        for i in range(5):
            await asyncio.sleep(3)
            await page.click('.next')
        """
        t = _t(code)
        # AST: body: sleep(3) + click(5) = 8, 5 * 8 = 40
        # Await: 2 * 6 * 5 = 60
        # max(40, 60) = 60 * 1.5 = 90
        assert t == pytest.approx(90.0)

    def test_sleep_unresolvable(self):
        """If sleep arg is a variable, use conservative default."""
        code = "await asyncio.sleep(delay)"
        t = _t(code)
        assert t == BASELINE


# ═════════════════════════════════════════════════════════════════
#  11. Function definitions
# ═════════════════════════════════════════════════════════════════


class TestFunctionDefs:
    def test_function_body_await_floor(self):
        """Defining a function — AST scorer skips body, but await
        counter sees the awaits inside and provides a floor."""
        code = """
        async def do_heavy_stuff():
            await page.goto("https://a.com")
            await page.goto("https://b.com")
            await page.goto("https://c.com")
            await show_page(page)

        print("defined")
        """
        t = _t(code)
        # AST scorer: function def = 0, print = 0 → 0
        # Await counter: 4 awaits * 6 = 24
        # max(0, 24) = 24 * 1.5 = 36
        assert t == pytest.approx(36.0)

    def test_function_def_then_call(self):
        """If the function is called, await counter counts body + call."""
        code = """
        async def helper():
            await page.goto("https://a.com")

        await helper()
        """
        t = _t(code)
        # AST scorer: function def = 0, helper() not in cost table → 0
        # Await counter: 1 (inside def) + 1 (the call) = 2 * 6 = 12
        # max(0, 12) = 12 * 1.5 = 18 → BASELINE
        assert t == BASELINE


# ═════════════════════════════════════════════════════════════════
#  12. Cross-block function context
# ═════════════════════════════════════════════════════════════════


class TestFunctionContext:
    """When function_sources is passed, predict_timeout sees bodies
    of functions defined in previous REPL steps."""

    def test_cross_block_simple(self):
        """Function defined in step N, called in step N+1."""
        fn_sources = {
            "scrape": textwrap.dedent("""
                async def scrape(page, url):
                    await page.goto(url)
                    data = await page.evaluate("() => []", isolated_context=True)
                    await show_page(page)
                    return data
            """).strip(),
        }
        code = 'result = await scrape(page, "https://example.com")'
        t_without = predict_timeout(code)
        t_with = predict_timeout(code, function_sources=fn_sources)
        # Without context: just 1 await → BASELINE
        assert t_without == BASELINE
        # With context: function body has goto(10)+evaluate(3)+show(25) = 38
        # Plus call await = 38 * 1.5 = 57 → above BASELINE
        assert t_with > BASELINE

    def test_cross_block_heavy_loop(self):
        """Heavy function with loop, called from a one-liner."""
        fn_sources = {
            "scrape": textwrap.dedent("""
                async def scrape(page, url):
                    await page.goto(url)
                    items = await page.evaluate("() => []", isolated_context=True)
                    for item in items:
                        await page.goto(item)
                        await page.wait_for_timeout(500)
                        await page.evaluate("() => ({})", isolated_context=True)
                    return items
            """).strip(),
        }
        code = 'result = await scrape(page, "https://example.com")'
        t = predict_timeout(code, function_sources=fn_sources)
        # Function body has: goto + evaluate + loop(10 iters * (goto + wait + evaluate))
        # This should produce a substantial timeout
        assert t > 100

    def test_unreferenced_function_ignored(self):
        """Functions in context that aren't called don't inflate the prediction."""
        fn_sources = {
            "heavy_unused": textwrap.dedent("""
                async def heavy_unused(page):
                    for i in range(50):
                        await page.goto("https://example.com")
            """).strip(),
        }
        code = 'print("hello")'
        t = predict_timeout(code, function_sources=fn_sources)
        assert t == BASELINE

    def test_empty_context(self):
        """Empty function_sources is the same as None."""
        code = 'await page.goto("https://example.com")'
        t_none = predict_timeout(code)
        t_empty = predict_timeout(code, function_sources={})
        assert t_none == t_empty

    def test_function_redefined(self):
        """Latest definition wins — mirrors REPL behavior."""
        fn_sources = {
            "scrape": textwrap.dedent("""
                async def scrape(page):
                    await page.goto("https://example.com")
            """).strip(),
        }
        # Code redefines scrape with heavier body + calls it
        code = textwrap.dedent("""
            async def scrape(page):
                for i in range(20):
                    await page.goto("https://example.com")
                    await page.evaluate("() => ({})", isolated_context=True)
            await scrape(page)
        """).strip()
        t = predict_timeout(code, function_sources=fn_sources)
        # The in-block definition is what matters (it shadows the context one)
        # The combined code has both defs + the call. Await counter sees all.
        assert t > 300


# ═════════════════════════════════════════════════════════════════
#  13. Realistic agent code patterns
# ═════════════════════════════════════════════════════════════════


class TestRealisticPatterns:
    def test_simple_extraction(self):
        """Agent grabs data from a page already loaded."""
        code = """
        results = await page.evaluate('''
            () => Array.from(document.querySelectorAll('.item')).map(el => ({
                text: el.querySelector('h3')?.innerText?.trim() || '',
                href: el.querySelector('a')?.href || '',
            }))
        ''', isolated_context=True)
        print(f"Got {len(results)} items")
        """
        t = _t(code)
        assert t == BASELINE

    def test_navigate_dismiss_extract(self):
        """Navigate, dismiss overlay, extract."""
        code = """
        await page.goto("https://shop.example.com", wait_until="domcontentloaded")
        await page.wait_for_selector('.cookie-banner', timeout=5_000)
        await page.click('.cookie-banner .accept')
        await page.wait_for_selector('.products', state='visible', timeout=10_000)
        data = await page.evaluate("() => []", isolated_context=True)
        await show_page(page)
        """
        t = _t(code)
        # goto(10) + wait(5) + click(5) + wait(10) + eval(3) + show(25) = 58 * 1.5 = 87
        assert t == pytest.approx(87.0)

    def test_full_pagination_scrape(self):
        """The common pattern that causes timeouts."""
        code = """
        all_items = []
        await page.goto("https://shop.example.com/products")
        await page.wait_for_selector('.product-card', state='visible')

        while True:
            items = await page.evaluate('''
                () => Array.from(document.querySelectorAll('.product-card')).map(el => ({
                    name: el.querySelector('h3')?.innerText || '',
                    price: el.querySelector('.price')?.innerText || '',
                }))
            ''', isolated_context=True)
            all_items.extend(items)
            print(f"Page done, total: {len(all_items)}")

            nxt = page.locator('a[aria-label="Next"]')
            if await nxt.count() == 0:
                break
            old_text = await page.locator('.product-card').first.inner_text()
            await nxt.click()
            await page.wait_for_function(
                "document.querySelector('.product-card')?.innerText !== arguments[0]",
                timeout=15_000
            )

        print(all_items)
        """
        t = _t(code)
        # Heavy pagination — generous budget
        assert t > 300

    def test_scroll_then_extract(self):
        code = """
        await page.goto("https://example.com/feed")
        await scroll_to_bottom(page, max_scrolls=20)
        data = await page.evaluate("() => []", isolated_context=True)
        print(data)
        """
        t = _t(code)
        # goto(10) + scroll(20) + eval(3) = 33 * 1.5 = 49.5
        assert t == pytest.approx(49.5)

    def test_spa_pagination_with_response_capture(self):
        code = """
        all_data = []
        page.on("response", on_response)

        for page_num in range(1, 11):
            await page.click(f'[data-page="{page_num}"]')
            await page.wait_for_load_state("networkidle")
            await page.wait_for_timeout(1000)
            data = await page.evaluate("() => []", isolated_context=True)
            all_data.extend(data)
        """
        t = _t(code)
        # 10 iters * (click(5) + wait_load(8) + wait(1) + eval(3)) = 10*17 = 170
        # await_floor: 4*6*10 = 240. max(170,240) = 240 * 1.5 = 360
        assert t == pytest.approx(360.0)

    def test_detail_page_crawl(self):
        """Visit multiple detail pages from a list."""
        code = """
        urls = await page.evaluate("() => Array.from(document.querySelectorAll('a.detail')).map(a => a.href)")
        results = []
        for url in urls:
            await page.goto(url)
            await page.wait_for_selector('.detail-content', timeout=10_000)
            data = await page.evaluate("() => ({})", isolated_context=True)
            results.append(data)
            await page.go_back()
            await page.wait_for_load_state("domcontentloaded")
        print(results)
        """
        t = _t(code)
        # pre-loop: evaluate(3)
        # loop body: goto(10) + wait(10) + eval(3) + go_back(8) + wait_load(8) = 39
        # 10 * 39 = 390 → capped 225 + 3 = 228. await: 1*6 + 5*6*10 = 306.
        # max(228, 306) = 306 * 1.5 = 459
        assert t == pytest.approx(459.0)


# ═════════════════════════════════════════════════════════════════
#  13. MAX_TIMEOUT cap
# ═════════════════════════════════════════════════════════════════


class TestCap:
    def test_never_exceeds_max(self):
        """No matter how heavy the code, we cap at MAX_TIMEOUT."""
        code = """
        while True:
            await page.goto("https://a.com")
            await show_page(page)
            await page.wait_for_selector('.x', timeout=60_000)
            await page.wait_for_function("true", timeout=60_000)
        """
        t = _t(code)
        assert t <= MAX_TIMEOUT

    def test_absurdly_heavy(self):
        # 50 gotos
        lines = ['await page.goto("https://x.com")'] * 50
        code = "\n".join(lines)
        t = predict_timeout(code)
        # 50 * 10 = 500 * 1.5 = 750
        assert t == pytest.approx(750.0)


# ═════════════════════════════════════════════════════════════════
#  14. Performance
# ═════════════════════════════════════════════════════════════════


class TestPerformance:
    def test_fast_on_large_code(self):
        """Algorithm should handle 500-line agent code in < 50ms."""
        lines = []
        for i in range(100):
            lines.append(f'await page.goto("https://example.com/{i}")')
            lines.append("await page.wait_for_selector('.item', timeout=5_000)")
            lines.append("data = await page.evaluate('() => []', isolated_context=True)")
            lines.append("await show_page(page)")
            lines.append(f'print("done {i}")')
        code = "\n".join(lines)

        start = time.perf_counter()
        result = predict_timeout(code)
        elapsed = time.perf_counter() - start

        assert elapsed < 0.05  # 50ms (await counter adds some cost)
        assert result == MAX_TIMEOUT  # heavy code → capped


# ═════════════════════════════════════════════════════════════════
#  15. Chained calls & edge cases
# ═════════════════════════════════════════════════════════════════


class TestEdgeCases:
    def test_chained_locator_click(self):
        """page.locator('.x').click() — should detect click."""
        code = "await page.locator('.x').click()"
        t = _t(code)
        assert t == BASELINE

    def test_locator_nth_click(self):
        code = "await page.locator('.x').nth(2).click()"
        t = _t(code)
        assert t == BASELINE

    def test_multiline_evaluate(self):
        """Multi-line JS in evaluate should still be scored once."""
        code = '''
        data = await page.evaluate("""
            () => {
                const items = document.querySelectorAll('.x');
                return Array.from(items).map(el => el.innerText);
            }
        """, isolated_context=True)
        '''
        t = _t(code)
        assert t == BASELINE

    def test_augmented_assign(self):
        code = """
        items = []
        items += await page.evaluate("() => []", isolated_context=True)
        """
        t = _t(code)
        assert t == BASELINE

    def test_mouse_wheel(self):
        code = "await page.mouse.wheel(0, 3000)"
        t = _t(code)
        assert t == BASELINE

    def test_keyboard_press_enter(self):
        code = """
        await page.fill('input[name="q"]', 'search term')
        await page.keyboard.press('Enter')
        """
        t = _t(code)
        assert t == BASELINE
