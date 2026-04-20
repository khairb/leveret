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
        # goto = 5s, * 1.2 = 6s → clamped to BASELINE
        assert t == BASELINE

    def test_evaluate(self):
        t = _t('result = await page.evaluate("() => 42")')
        assert t == BASELINE  # 1.0 * 1.2 = 1.2 → baseline

    def test_click(self):
        t = _t("await page.click('button.submit')")
        assert t == BASELINE

    def test_show_page(self):
        """show_page is known to be slow (8-22s)."""
        t = _t("await show_page(page)")
        # 15.0 * 1.2 = 18.0 → still baseline
        assert t == BASELINE

    def test_wait_for_selector_default(self):
        t = _t("await page.wait_for_selector('.item')")
        # 5s default * 1.2 = 6 → baseline
        assert t == BASELINE

    def test_wait_for_selector_with_timeout(self):
        """When the agent specifies a large timeout kwarg, respect it."""
        t = _t("await page.wait_for_selector('.item', timeout=60_000)")
        # 60s * 1.2 = 72 → above baseline
        assert t > BASELINE
        assert t == pytest.approx(72.0)

    def test_wait_for_timeout_explicit(self):
        """page.wait_for_timeout(ms) is a hard sleep."""
        t = _t("await page.wait_for_timeout(10_000)")
        # 10s * 1.2 = 12 → baseline
        assert t == BASELINE

    def test_asyncio_sleep(self):
        t = _t("await asyncio.sleep(2)")
        # 5s * 1.2 = 6 → baseline
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
        # goto(5) + wait_for_selector(15) + evaluate(1) = 21 * 1.2 = 25.2 → baseline
        assert t == BASELINE

    def test_goto_show_page(self):
        code = """
        await page.goto("https://example.com")
        await show_page(page)
        """
        t = _t(code)
        # 5 + 15 = 20 * 1.2 = 24 → baseline
        assert t == BASELINE

    def test_many_operations_exceeds_baseline(self):
        """Enough operations should push past 30s."""
        code = """
        await page.goto("https://example.com")
        await page.wait_for_selector('.a', timeout=10_000)
        await show_page(page)
        await page.click('.btn')
        await page.wait_for_load_state("domcontentloaded")
        await show_page(page)
        """
        t = _t(code)
        # goto(5) + wait(10) + show(15) + click(1.5) + wait_load(3) + show(15) = 49.5 * 1.2 = 59.4
        assert t > BASELINE
        assert t == pytest.approx(59.4)

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
        # 6 * 5 = 30 * 1.2 = 36
        assert t == pytest.approx(36.0)


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
        # body = click(1.5) + wait_load(3) = 4.5, 10 * 4.5 = 45 * 1.2 = 54
        assert t == pytest.approx(54.0)

    def test_for_range_start_stop(self):
        """range(2, 8) → 6 iterations."""
        code = """
        for i in range(2, 8):
            await page.click('.item')
        """
        t = _t(code)
        # 6 * 1.5 = 9 * 1.2 = 10.8 → baseline
        assert t == BASELINE

    def test_for_unknown_iterable(self):
        """for item in items — defaults to 5 iterations."""
        code = """
        for item in items:
            await page.goto(item)
            await show_page(page)
        """
        t = _t(code)
        # body = goto(5) + show(15) = 20, 5 * 20 = 100 (under 225 cap) * 1.2 = 120
        assert t > BASELINE
        assert t == pytest.approx(120.0)

    def test_for_range_capped(self):
        """range(1000) is capped at 50 iterations."""
        code = """
        for i in range(1000):
            await page.click('.x')
        """
        t = _t(code)
        # 50 * 1.5 = 75 (under 225 cap) * 1.2 = 90
        assert t == pytest.approx(90.0)

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
        # body: evaluate(1) + click(1.5) + wait_load(3) = 5.5
        # while: 10 * 5.5 = 55 * 1.2 = 66
        assert t == pytest.approx(66.0)

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
        # 10 * (1.5 + 3.0) = 45 * 1.2 = 54
        assert t == pytest.approx(54.0)

    def test_loop_body_cap(self):
        """Loop contribution is capped at 225s with heavy body."""
        code = """
        while True:
            await page.goto("https://example.com")
            await show_page(page)
            await page.wait_for_selector('.data', timeout=30_000)
        """
        t = _t(code)
        # body: goto(5) + show(15) + wait(30) = 50
        # 10 * 50 = 500 → capped at 225 * 1.2 = 270
        assert t == pytest.approx(270.0)

    def test_nested_loop(self):
        """Nested loop — inner scores multiply with outer iterations."""
        code = """
        for i in range(3):
            for j in range(4):
                await page.click('.item')
        """
        t = _t(code)
        # inner: 4 * 1.5 = 6 (under 225 cap)
        # outer: 3 * 6 = 18 (under 225 cap)
        # * 1.2 = 21.6 → baseline
        assert t == BASELINE

    def test_async_for(self):
        """async for — uses default 5 iterations."""
        code = """
        async for item in some_async_generator():
            await page.goto(item.url)
        """
        t = _t(code)
        # 5 * 5 = 25 * 1.2 = 30 → baseline
        assert t == BASELINE


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
        # expect_navigation(5) + click(1.5) = 6.5 * 1.2 = 7.8 → baseline
        assert t == BASELINE

    def test_expect_response(self):
        code = """
        async with page.expect_response(lambda r: "/api/data" in r.url) as resp_info:
            await page.click('#load-btn')
        response = await resp_info.value
        """
        t = _t(code)
        # expect_response(5) + click(1.5) = 6.5 * 1.2 → baseline
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
        # if-branch: goto(5) = 5
        # else-branch: goto(5) + show(15) + wait(30) = 50
        # max = 50 * 1.2 = 60
        assert t == pytest.approx(60.0)


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
        # try body: goto(5) + wait(20) = 25
        # except body: reload(5)
        # max(25, 5) = 25 * 1.2 = 30 → baseline
        assert t == BASELINE


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
        # scroll(30 * 1 = 30) + evaluate(1) = 31 * 1.2 = 37.2
        assert t == pytest.approx(37.2)

    def test_scroll_default(self):
        code = "await scroll_to_bottom(page)"
        t = _t(code)
        # 15 * 1.2 = 18 → baseline
        assert t == BASELINE

    def test_scroll_capped(self):
        code = "await scroll_to_bottom(page, max_scrolls=200)"
        t = _t(code)
        # 200 capped to 50 → 50 * 1.2 = 60
        assert t == pytest.approx(60.0)


# ═════════════════════════════════════════════════════════════════
#  9. Timeout kwarg extraction
# ═════════════════════════════════════════════════════════════════

class TestTimeoutKwarg:
    def test_wait_for_function_with_timeout(self):
        code = "await page.wait_for_function('() => true', timeout=45_000)"
        t = _t(code)
        # 45s * 1.2 = 54
        assert t == pytest.approx(54.0)

    def test_locator_wait_for_with_timeout(self):
        code = 'await loc.wait_for(state="visible", timeout=20_000)'
        t = _t(code)
        # 20s * 1.2 = 24 → baseline
        assert t == BASELINE

    def test_wait_for_selector_small_timeout(self):
        """Small timeout kwarg → uses that value, not the default."""
        code = "await page.wait_for_selector('.x', timeout=2_000)"
        t = _t(code)
        # 2s * 1.2 = 2.4 → baseline
        assert t == BASELINE


# ═════════════════════════════════════════════════════════════════
#  10. Duration argument extraction
# ═════════════════════════════════════════════════════════════════

class TestDurationArg:
    def test_wait_for_timeout_ms(self):
        code = "await page.wait_for_timeout(25_000)"
        t = _t(code)
        # 25s * 1.2 = 30 → baseline
        assert t == BASELINE

    def test_asyncio_sleep_float(self):
        code = "await asyncio.sleep(2.5)"
        t = _t(code)
        # 2.5s * 1.2 = 3 → baseline
        assert t == BASELINE

    def test_large_sleep_in_loop(self):
        code = """
        for i in range(5):
            await asyncio.sleep(3)
            await page.click('.next')
        """
        t = _t(code)
        # body: sleep(3) + click(1.5) = 4.5, 5 * 4.5 = 22.5 * 1.2 = 27 → baseline
        assert t == BASELINE

    def test_sleep_unresolvable(self):
        """If sleep arg is a variable, use conservative default."""
        code = "await asyncio.sleep(delay)"
        t = _t(code)
        # 5s default * 1.2 = 6 → baseline
        assert t == BASELINE


# ═════════════════════════════════════════════════════════════════
#  11. Function definitions (not scored)
# ═════════════════════════════════════════════════════════════════

class TestFunctionDefs:
    def test_function_body_not_scored(self):
        """Defining a function doesn't execute it — only calls do."""
        code = """
        async def do_heavy_stuff():
            await page.goto("https://a.com")
            await page.goto("https://b.com")
            await page.goto("https://c.com")
            await show_page(page)

        print("defined")
        """
        t = _t(code)
        assert t == BASELINE

    def test_function_def_then_call(self):
        """If the function is called, we score the call (not the body)."""
        code = """
        async def helper():
            await page.goto("https://a.com")

        await helper()
        """
        t = _t(code)
        # helper() is not in our cost table → 0
        assert t == BASELINE


# ═════════════════════════════════════════════════════════════════
#  12. Realistic agent code patterns
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
        # goto(5) + wait(5) + click(1.5) + wait(10) + eval(1) + show(15) = 37.5 * 1.2 = 45
        assert t == pytest.approx(45.0)

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
        # Pre-loop: goto(5) + wait_for_selector(5) = 10
        # Loop body: evaluate(1) + click(1.5) + wait_for_function(15) = 17.5
        # while: 10 * 17.5 = 175 (under 225 cap)
        # total raw: 10 + 175 = 185 * 1.2 = 222
        assert t == pytest.approx(222.0)

    def test_scroll_then_extract(self):
        code = """
        await page.goto("https://example.com/feed")
        await scroll_to_bottom(page, max_scrolls=20)
        data = await page.evaluate("() => []", isolated_context=True)
        print(data)
        """
        t = _t(code)
        # goto(5) + scroll(20) + eval(1) = 26 * 1.2 = 31.2
        assert t == pytest.approx(31.2)

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
        # loop: 10 iterations * (click 1.5 + wait_load 3 + wait_timeout 1 + eval 1) = 10 * 6.5 = 65
        # * 1.2 = 78
        assert t == pytest.approx(78.0)

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
        # pre-loop: evaluate(1) = 1
        # loop body: goto(5) + wait(10) + eval(1) + go_back(3) + wait_load(3) = 22
        # for (unknown iterable): 5 * 22 = 110 (under 225 cap)
        # total: 1 + 110 = 111 * 1.2 = 133.2
        assert t == pytest.approx(133.2)


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
        # 50 * 5 = 250 * 1.2 = 300 → exactly MAX_TIMEOUT
        assert t == MAX_TIMEOUT


# ═════════════════════════════════════════════════════════════════
#  14. Performance
# ═════════════════════════════════════════════════════════════════

class TestPerformance:
    def test_fast_on_large_code(self):
        """Algorithm should handle 500-line agent code in < 10ms."""
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

        assert elapsed < 0.01  # 10ms
        assert result == MAX_TIMEOUT  # heavy code → capped


# ═════════════════════════════════════════════════════════════════
#  15. Chained calls & edge cases
# ═════════════════════════════════════════════════════════════════

class TestEdgeCases:
    def test_chained_locator_click(self):
        """page.locator('.x').click() — should detect click."""
        code = "await page.locator('.x').click()"
        t = _t(code)
        # click(1.5) * 1.2 = 1.8 → baseline
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
