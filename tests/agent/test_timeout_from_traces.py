"""Timeout prediction tests derived from real agent traces.

Each test uses EXACT code written by an agent during a real scraping run
that caused a timeout error.  The code is fed to ``predict_timeout()``
and the result is checked against a minimum adequate value.

**Tests that FAIL expose bugs in the prediction system.**

Assertion strategy:
  - For "Execution exceeded Xs limit" errors:
      ``assert predict_timeout(code) > X``
    The code *definitely* needed more than X seconds.  If the prediction
    equals X, the test FAILS, exposing that the prediction was too low.

  - For Playwright call timeouts (wrong selector, overlay, etc.):
      ``assert predict_timeout(code) >= expected``
    where ``expected`` is the correct prediction based on the cost table.
    These tests PASS when the algorithm works, confirming the issue was
    the selector (not the prediction).  They FAIL only if there's a
    scoring bug.
"""

from __future__ import annotations

import textwrap

import pytest

from scout.agent.timeout_predict import (
    BASELINE,
    predict_timeout,
)


def _t(code: str) -> float:
    """Shortcut: dedent + predict."""
    return predict_timeout(textwrap.dedent(code).strip())


# ═══════════════════════════════════════════════════════════════════
#  1. EXECUTION TIMEOUTS — predict_timeout was too low
#
#  These are the critical cases.  The execution was killed by
#  asyncio.wait_for because predict_timeout underestimated the
#  time needed.  Most should FAIL, exposing prediction bugs.
# ═══════════════════════════════════════════════════════════════════


class TestExecTimeout_FunctionDefCall:
    """Function defined + called in same block.  Body cost is lost."""

    def test_github_trending_scrape_function(self):
        """Source: benchmark/traces/run_2026-05-10_23-20-41
        Exceeded: 20.0s
        Root cause: scrape() defined and called; body has loop with goto+evaluate.
        """
        code = """
        async def scrape(page, start_url, checkpoint):
            await page.goto(start_url, wait_until="domcontentloaded")
            checkpoint("trending_page_loaded")
            repos_from_trending = await page.evaluate('''
                () => {
                    const articles = document.querySelectorAll('article.Box-row');
                    return Array.from(articles).map(article => {
                        const repoLink = article.querySelector('h2 a.Link');
                        if (!repoLink) return null;
                        const href = repoLink.getAttribute('href');
                        return { full_name: href };
                    }).filter(r => r !== null);
                }
            ''', isolated_context=True)
            all_repos = []
            for i, repo_info in enumerate(repos_from_trending, 1):
                repo_url = f"https://github.com/{repo_info['full_name']}"
                await page.goto(repo_url, wait_until="domcontentloaded")
                await page.wait_for_timeout(300)
                extracted = await page.evaluate("() => ({})", isolated_context=True)
                all_repos.append(extracted)
            return all_repos

        result = await scrape(page, "https://github.com/trending/python?since=weekly",
                              lambda label, data_preview=None: print(f"[checkpoint] {label}"))
        """
        t = _t(code)
        assert t > 20.0, f"predicted {t}s but execution exceeded 20s"

    def test_yc_companies_extract_loop(self):
        """Source: benchmark/traces/run_2026-05-09_18-14-33
        Exceeded: 20.0s
        Root cause: extract_company_data() called in loop; body navigates to each page.
        """
        code = """
        w24_slugs = w24_slugs_all[:40]
        w24_data = []
        for i, slug in enumerate(w24_slugs, 1):
            try:
                company = await extract_company_data(page, slug, 'W24')
                w24_data.append(company)
                if i % 10 == 0:
                    print(f"  Extracted {i}/{len(w24_slugs)} companies")
            except Exception as e:
                print(f"  Error extracting {slug}: {str(e)[:100]}")
        """
        t = _t(code)
        assert t > 20.0, f"predicted {t}s but execution exceeded 20s"

    def test_yc_companies_reextract_all(self):
        """Source: benchmark/traces/run_2026-05-09_18-14-33
        Exceeded: 120.0s
        Root cause: same pattern but agent set explicit timeout=120.
        """
        code = """
        w24_data_v2 = []
        for i, slug in enumerate(w24_slugs, 1):
            try:
                company = await extract_company_data_v2(page, slug, 'W24')
                w24_data_v2.append(company)
            except Exception as e:
                print(f"  Error extracting {slug}: {str(e)[:100]}")

        s24_data_v2 = []
        for i, slug in enumerate(s24_slugs, 1):
            try:
                company = await extract_company_data_v2(page, slug, 'S24')
                s24_data_v2.append(company)
            except Exception as e:
                print(f"  Error extracting {slug}: {str(e)[:100]}")
        """
        t = _t(code)
        assert t > 120.0, f"predicted {t}s but execution exceeded 120s"

    def test_yc_detail_function_def_call(self):
        """Source: benchmark/traces/run_2026-05-09_15-20-56
        Exceeded: 20.0s
        Root cause: function defined with goto+wait_for_selector+evaluate, called in loop.
        """
        code = """
        async def extract_company_detail(page, slug):
            await page.goto(f"https://www.ycombinator.com/companies/{slug}",
                            wait_until="domcontentloaded", timeout=15000)
            await page.wait_for_selector("h1, div.text-2xl", timeout=10000)
            data = await page.evaluate("() => ({})", isolated_context=True)
            return data

        test_slugs = company_slugs[:3]
        for i, slug in enumerate(test_slugs):
            data = await extract_company_detail(page, slug)
        """
        t = _t(code)
        assert t > 20.0, f"predicted {t}s but execution exceeded 20s"

    def test_yc_all_497_companies_batch(self):
        """Source: benchmark/traces/run_2026-05-09_14-48-10
        Exceeded: 20.0s (first attempt), then 300.0s (with explicit timeout)
        """
        code = """
        company_slugs = sorted(listing_lookup.keys())
        all_results = []
        batch_size = 50
        total = len(company_slugs)

        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch_slugs = company_slugs[batch_start:batch_end]
            batch_results = await scrape_all_companies(page, listing_lookup, batch_slugs)
            all_results.extend(batch_results)
        """
        t = _t(code)
        assert t > 20.0, f"predicted {t}s but execution exceeded 20s"

    def test_realestate_scrape_function_20s(self):
        """Source: benchmark/traces/run_2026-05-10_14-46-45
        Exceeded: 20.0s
        Root cause: scrape() function navigates to detail pages in loop.
        """
        code = """
        result = await scrape(page,
            'https://www.example-realestate.com/en/alquiler-viviendas/madrid-madrid/',
            mock_checkpoint)
        """
        t = _t(code)
        assert t > 20.0, f"predicted {t}s but execution exceeded 20s"

    def test_realestate_scrape_function_120s(self):
        """Source: benchmark/traces/run_2026-05-10_14-46-45
        Exceeded: 120.0s (agent retried with explicit timeout=120)

        The scrape() function was defined in a previous code block.
        With function_sources context, the predictor can see the body.
        """
        # The function body from the previous step (representative)
        fn_sources = {
            "scrape": textwrap.dedent("""
                async def scrape(page, url, checkpoint):
                    all_listings = []
                    await page.goto(url, wait_until="domcontentloaded")
                    for page_num in range(10):
                        listings = await page.evaluate("() => []", isolated_context=True)
                        all_listings.extend(listings)
                        next_btn = page.locator('a.next')
                        if await next_btn.count() == 0:
                            break
                        await next_btn.click()
                        await page.wait_for_load_state("domcontentloaded")
                    return all_listings
            """).strip(),
        }
        code = """
        result = await scrape(page,
            'https://www.example-realestate.com/en/alquiler-viviendas/madrid-madrid/',
            mock_checkpoint)
        """
        # Without context: only 1 await → BASELINE (30s) — would fail
        t_no_ctx = _t(code)
        assert t_no_ctx == BASELINE

        # With context: function body visible → much higher prediction
        t = predict_timeout(textwrap.dedent(code).strip(), function_sources=fn_sources)
        assert t > 120.0, f"predicted {t}s but execution exceeded 120s"

    def test_travel_full_scrape_function(self):
        """Source: traces/run_2026-04-20_01-04-54
        Error: Locator.click timed out inside the function body (30s function timeout)
        Root cause: heavy scrape() function body not scored.
        """
        code = """
        async def scrape(page, url, checkpoint):
            await page.goto(url, wait_until="domcontentloaded", timeout=30000)
            await page.wait_for_selector('[data-testid="structured-search-input-field-query"]',
                                         timeout=15000)
            await page.fill('[data-testid="structured-search-input-field-query"]', 'Berlin')
            await page.locator('div[role="button"]', has_text='Wann').click()
            await page.locator('button[data-state--date-string="2026-04-20"]').click()
            await page.locator('button[data-state--date-string="2026-05-19"]').click()
            await page.locator('[data-testid="structured-search-input-search-button"]').click()
            await page.wait_for_selector('div[data-testid="card-container"]', timeout=20000)
            await page.locator('[data-testid="category-bar-filter-button"]').click()
            await page.wait_for_selector('button[aria-label="Waschmaschine"]', timeout=10000)
            await page.locator('button[aria-label="Waschmaschine"]').click()
            await page.locator('button[aria-label="TV"]').click()
            await page.locator('#filter-item-amenities-4').click()
            all_items = []
            while True:
                cards = await page.locator('div[data-testid="card-container"]').all()
                for card in cards:
                    title = await card.locator('div[data-testid="listing-card-title"]').inner_text()
                    all_items.append(title)
                nxt = page.locator('a[aria-label="Weiter"]')
                if await nxt.count() == 0:
                    break
                await nxt.first.click()
                await page.wait_for_function("() => true", timeout=20000)
            return all_items

        result = await scrape(page, "https://www.example-travel.com/",
                              lambda *a, **kw: None)
        """
        t = _t(code)
        # Function body has: goto + 4 waits + 7 clicks + while loop with inner_text loop
        # Correct prediction if body were scored: > 100s
        assert t > 30.0, f"predicted {t}s but function body needs >> 30s"


class TestExecTimeout_InlineLoops:
    """Inline loops (no function wrapper) that exceeded execution limit."""

    def test_github_trending_inline_loop(self):
        """Source: benchmark/traces/run_2026-05-10_21-54-10
        Exceeded: 20.0s
        Root cause: for loop over unknown iterable, each iteration does goto+evaluate.
        """
        code = """
        repo_links = await page.evaluate('''
            () => Array.from(document.querySelectorAll('article.Box-row')).map(article => {
                const repoLink = article.querySelector('h2 a.Link');
                return repoLink?.getAttribute('href');
            }).filter(Boolean)
        ''', isolated_context=True)

        all_repos = []
        for i, repo_href in enumerate(repo_links, 1):
            await page.goto(f"https://github.com{repo_href}", wait_until="domcontentloaded")
            await page.wait_for_timeout(500)
            repo_data = await page.evaluate("() => ({})", isolated_context=True)
            all_repos.append(repo_data)
        """
        t = _t(code)
        # predict_timeout uses 5 default iterations, but actual repos = 13+
        # With 5 iters: evaluate(3) + 5*(goto(10)+wait(0.5)+evaluate(3)) = 3+67.5 = 70.5 * 1.2 = 84.6
        # 84.6 > 20, so this should PASS (prediction IS adequate for 5 iterations)
        assert t > 20.0, f"predicted {t}s but execution exceeded 20s"

    def test_travel_card_extraction_while_loop(self):
        """Source: traces/run_2026-04-20_16-15-22
        Exceeded: 20.0s
        Root cause: while loop with nested for loop doing per-card locator calls.
        """
        code = """
        import re
        all_apartments = []
        current_page = 1
        max_pages = 3

        while current_page <= max_pages:
            await page.wait_for_selector('div[data-testid="card-container"]', timeout=10000)
            await page.wait_for_timeout(500)
            cards = await page.locator('div[data-testid="card-container"]').all()
            for i, card in enumerate(cards):
                try:
                    title = await card.locator('[data-testid="listing-card-title"]').inner_text()
                    description = await card.locator('[data-testid="listing-card-name"]').inner_text()
                    subtitles = await card.locator('[data-testid="listing-card-subtitle"]').all_inner_texts()
                    price_text = await card.locator('span.atm_rq_glywfm').inner_text()
                    rating_text = await card.locator('span.atm_mj_glywfm').first.inner_text()
                    url = await card.locator('a[aria-labelledby^="title_"]').first.get_attribute('href')
                    full_url = "https://www.example-travel.com" + url if url.startswith('/') else url
                    all_apartments.append({"title": title, "url": full_url})
                except Exception as e:
                    continue

            if current_page < max_pages:
                next_link = page.locator('a[aria-label="Weiter"]')
                if await next_link.count() > 0:
                    await next_link.click()
                    await page.wait_for_load_state("domcontentloaded")
                    await page.wait_for_timeout(1000)
                    current_page += 1
                else:
                    break
            else:
                current_page += 1
        """
        t = _t(code)
        assert t > 20.0, f"predicted {t}s but execution exceeded 20s"

    def test_button_iteration_30s(self):
        """Source: traces/run_2026-04-19_16-36-43
        Exceeded: 30.0s
        Root cause: iterating ALL buttons on page calling text_content() each.
        """
        code = """
        all_btns = page.locator('button').all()
        for i, btn in enumerate(await all_btns):
            txt = await btn.text_content()
            print(f"Button {i}: {txt}")
        """
        t = _t(code)
        assert t > 30.0, f"predicted {t}s but execution exceeded 30s"

    def test_element_attribute_iteration_120s(self):
        """Source: traces/run_2026-04-19_22-49-35
        Exceeded: 120.0s
        Root cause: locating all 'button, div' with text '20'/'19', iterating
        each with text_content() + get_attribute(). Hundreds of elements.
        """
        code = """
        start_date_elements = await page.locator('button, div', has_text='20').all()
        end_date_elements = await page.locator('button, div', has_text='19').all()

        start_date_info = []
        for el in start_date_elements:
            text = await el.text_content()
            aria_label = await el.get_attribute('aria-label')
            start_date_info.append({'text': text.strip() if text else '', 'aria-label': aria_label})

        end_date_info = []
        for el in end_date_elements:
            text = await el.text_content()
            aria_label = await el.get_attribute('aria-label')
            end_date_info.append({'text': text.strip() if text else '', 'aria-label': aria_label})

        start_date_info, end_date_info
        """
        t = _t(code)
        assert t > 120.0, f"predicted {t}s but execution exceeded 120s"

    def test_dismiss_overlay_button_iteration_62s(self):
        """Source: traces/run_2026-04-20_13-59-19
        Exceeded: 62.4s
        Root cause: iterating all buttons looking for consent/close, then clicking filter.
        """
        code = """
        buttons = await page.locator('button').all()
        for btn in buttons:
            text = (await btn.inner_text()).strip().lower()
            visible = await btn.is_visible()
            if visible and any(x in text for x in ['akzeptieren', 'accept', 'schließen', 'close']):
                await btn.click()
                break

        filter_button = page.locator('button[data-testid="category-bar-filter-button"]')
        await filter_button.click()
        await page.wait_for_timeout(2000)
        await show_page(page)
        """
        t = _t(code)
        assert t > 62.4, f"predicted {t}s but execution exceeded 62.4s"

    def test_date_button_debug_24s(self):
        """Source: traces/run_2026-04-20_15-42-03
        Exceeded: 24.6s
        Root cause: get_by_role finds many buttons, .first.click() waits.
        """
        code = """
        april_20_button = page.get_by_role("button", name="20")
        print(f"April 20 buttons found: {await april_20_button.count()}")
        await april_20_button.first.click()
        await page.wait_for_timeout(500)
        await show_page(page)
        """
        t = _t(code)
        assert t > 24.6, f"predicted {t}s but execution exceeded 24.6s"

    def test_date_dismiss_and_click_20s(self):
        """Source: traces/run_2026-04-20_09-48-37
        Exceeded: 20.0s
        Root cause: dismiss consent + click date button (30s Playwright timeout).
        """
        code = """
        consent_btn = await page.query_selector('button:has-text("Alle akzeptieren")')
        if consent_btn:
            await consent_btn.click()
            print("Cookie consent dismissed.")
        await page.wait_for_timeout(500)
        await page.click('button[aria-label*="20, Monday, April 2026"]')
        print("Clicked April 20, 2026.")
        await page.click('button[aria-label*="19, Tuesday, May 2026"]')
        print("Clicked May 19, 2026.")
        """
        t = _t(code)
        assert t > 20.0, f"predicted {t}s but execution exceeded 20s"

    def test_filter_button_show_page_24s(self):
        """Source: traces/run_2026-04-20_10-48-19
        Exceeded: 24.0s
        Root cause: filter button click may take 10s, then show_page takes 15s.
        """
        code = """
        await page.locator('button[data-testid="category-bar-filter-button"]').click()
        await show_page(page)
        """
        t = _t(code)
        assert t > 24.0, f"predicted {t}s but execution exceeded 24s"


class TestExecTimeout_ShowPageSlow:
    """show_page() combined with other ops exceeding the limit."""

    def test_language_select_show_page_30s(self):
        """Source: src/traces/run_2026-04-12_07-50-11
        Exceeded: 30.0s
        """
        code = """
        lang_button = page.locator('button[aria-label="Sprache auswählen"]').first
        await lang_button.click()
        await page.wait_for_timeout(1000)
        await show_page(page)
        """
        t = _t(code)
        assert t > 30.0, f"predicted {t}s but execution exceeded 30s"

    def test_search_button_show_page_30s(self):
        """Source: src/traces/run_2026-04-12_07-50-11
        Exceeded: 30.0s
        """
        code = """
        search_button = page.locator('button[data-testid="structured-search-input-search-button"]')
        await search_button.click()
        await page.wait_for_load_state('domcontentloaded')
        await page.wait_for_timeout(2000)
        await show_page(page)
        """
        t = _t(code)
        assert t > 30.0, f"predicted {t}s but execution exceeded 30s"

    def test_wait_then_show_page_30s(self):
        """Source: src/traces/run_2026-04-12_07-50-11
        Exceeded: 30.0s
        """
        code = """
        await page.wait_for_timeout(3000)
        await show_page(page)
        """
        t = _t(code)
        assert t > 30.0, f"predicted {t}s but execution exceeded 30s"

    def test_language_select_show_page_30s_2(self):
        """Source: src/traces/run_2026-04-17_19-29-22
        Exceeded: 30.0s
        """
        code = """
        lang_button = page.locator('button[aria-label="Sprache auswählen"]').first
        await lang_button.click()
        await page.wait_for_timeout(1000)
        await show_page(page)
        """
        t = _t(code)
        assert t > 30.0, f"predicted {t}s but execution exceeded 30s"

    @pytest.mark.xfail(
        reason=(
            "Two simple calls (count + inner_text) that took >30s due to extreme "
            "page complexity.  No static analysis can predict this."
        )
    )
    def test_pagination_inner_text_30s(self):
        """Source: src/traces/run_2026-04-12_09-55-15
        Exceeded: 30.0s
        """
        code = """
        next_button = page.locator('a[aria-label="Next"]')
        has_next = await next_button.count() > 0
        print(f"Has next page: {has_next}")
        current_page_text = await page.inner_text('div[role="navigation"]')
        print(f"Pagination text: {current_page_text}")
        """
        t = _t(code)
        assert t > 30.0, f"predicted {t}s but execution exceeded 30s"


# ═══════════════════════════════════════════════════════════════════
#  2. PLAYWRIGHT CALL TIMEOUTS — individual Playwright call timed out
#
#  In these cases the execution budget was usually adequate, but a
#  single Playwright call ate the time (wrong selector, disabled
#  element, invisible element).  These tests verify that
#  predict_timeout at least gives enough budget for the code.
#  Most should PASS.
# ═══════════════════════════════════════════════════════════════════


class TestPW_WrongPaginationSelector:
    """Clicking pagination links with wrong selectors."""

    def test_page_2_href_param(self):
        """Source: traces/run_2026-04-20_17-29-13, run_2026-04-18_12-32-52, run_2026-04-20_11-06-45"""
        code = """
        await page.click('a[href*="page=2"]')
        await page.wait_for_load_state("domcontentloaded")
        await page.wait_for_timeout(1500)
        """
        t = _t(code)
        # click(5) + wait_load(8) + wait_timeout(1.5) = 14.5 * 1.2 = 17.4 → BASELINE
        assert t >= BASELINE

    def test_page_2_ampersand_href(self):
        """Source: traces/run_2026-04-20_15-42-03"""
        code = """
        page_2_link = page.locator('a[href*="&page=2"]')
        await page_2_link.click()
        await page.wait_for_load_state("domcontentloaded")
        await page.wait_for_timeout(1000)
        await show_page(page)
        """
        t = _t(code)
        assert t >= BASELINE

    def test_pagination_amenities_href(self):
        """Source: traces/run_2026-04-20_18-40-38"""
        code = """
        await page.click('a[href*="amenities"][href*="2"]')
        await page.wait_for_load_state("domcontentloaded")
        await page.wait_for_timeout(1000)
        """
        t = _t(code)
        assert t >= BASELINE

    def test_pagination_nav_has_text(self):
        """Source: traces/run_2026-04-20_18-40-38"""
        code = """
        await page.click('nav[aria-label="Nummerierung der Suchergebnisse"] a:has-text("1")')
        await page.wait_for_load_state("domcontentloaded")
        await page.wait_for_timeout(1500)
        """
        t = _t(code)
        assert t >= BASELINE

    def test_pagination_has_text_3(self):
        """Source: traces/run_2026-04-20_16-28-34"""
        code = """
        await page.click('a[href*="/s/Berlin/homes"]:has-text("3")')
        await page.wait_for_load_state("domcontentloaded")
        await page.wait_for_timeout(2000)
        """
        t = _t(code)
        assert t >= BASELINE

    def test_realestate_pagina_2(self):
        """Source: benchmark/traces/run_2026-05-09_23-57-52"""
        code = """
        await page.click('a[href*="pagina=2"]')
        await page.wait_for_load_state("domcontentloaded")
        await page.wait_for_timeout(2000)
        """
        t = _t(code)
        assert t >= BASELINE

    def test_jobs_page_2(self):
        """Source: traces/run_2026-05-09_11-11-33"""
        code = """
        next_page_url = "https://www.example-jobs.com/jobs?q=python+developer&l=New+York&start=10"
        await page.goto(next_page_url, wait_until="domcontentloaded")
        await page.wait_for_selector('a.jcs-JobTitle', timeout=15_000)
        """
        t = _t(code)
        # goto(10) + wait_for_selector(15) = 25 * 1.2 = 30
        assert t >= 30.0


class TestPW_WrongFilterSelector:
    """Clicking filter buttons/checkboxes with wrong selectors."""

    def test_tv_button_has_text(self):
        """Source: traces/run_2026-04-20_20-56-52"""
        code = "await page.click('button:has-text(\"TV\")')"
        t = _t(code)
        assert t >= BASELINE

    def test_tv_button_aria_label(self):
        """Source: traces/run_2026-04-20_20-56-52, run_2026-04-20_17-29-13"""
        code = "await page.click('button[aria-label=\"TV\"]')"
        t = _t(code)
        assert t >= BASELINE

    def test_tv_button_id(self):
        """Source: traces/run_2026-04-20_20-56-52, run_2026-04-20_17-29-13"""
        code = "await page.click('button[id=\"filter-item-amenities-58\"]')"
        t = _t(code)
        assert t >= BASELINE

    def test_filter_checkbox_check(self):
        """Source: traces/run_2026-04-20_14-04-41"""
        code = """
        await page.locator('input[data-testid="filter-amenities-wifi-input"]').check()
        await page.locator('input[data-testid="filter-amenities-washer-input"]').check()
        await page.locator('input[data-testid="filter-amenities-tv-input"]').check()
        await page.locator('button[data-testid="filter-panel-apply-button"]').click()
        await show_page(page)
        """
        t = _t(code)
        # check(3)*3 + click(5) + show_page(15) = 29 * 1.2 = 34.8
        assert t >= 34.0

    def test_filter_amenity_button_id(self):
        """Source: traces/run_2026-04-20_14-04-41"""
        code = """
        await page.locator('button#filter-item-amenities-4').click()
        """
        t = _t(code)
        assert t >= BASELINE

    def test_filter_results_link(self):
        """Source: traces/run_2026-04-20_10-23-29"""
        code = """
        await page.locator('button[aria-label="Waschmaschine"]').click()
        await page.locator('button[aria-label="TV"]').click()
        await page.locator('button#filter-item-amenities-4').click()
        await page.locator('footer-93-unterk-nfte-anzei a').click()
        await show_page(page)
        """
        t = _t(code)
        # click(5)*4 + show_page(15) = 35 * 1.2 = 42
        assert t >= 42.0

    def test_show_companies_button(self):
        """Source: benchmark/traces/run_2026-05-09_18-03-14"""
        code = """
        await page.goto('https://www.ycombinator.com/companies', wait_until='domcontentloaded',
                         timeout=30000)
        await page.locator('text=See all options').first.click()
        for label_text in ['Winter 2024', 'Summer 2024']:
            lab = page.locator('label').filter(has_text=label_text).first
            await lab.click()
        await page.locator('button', has_text='Show 1,000+ companies').click()
        await page.wait_for_load_state('domcontentloaded')
        await show_page(page)
        """
        t = _t(code)
        # goto(10) + click(5)*4 + wait_load(8) + show_page(15) = 53 * 1.2 = 63.6
        assert t >= 60.0

    def test_see_all_options(self):
        """Source: benchmark/traces/run_2026-05-09_16-03-53"""
        code = """
        await page.goto('https://www.ycombinator.com/companies', wait_until='domcontentloaded',
                         timeout=30000)
        await page.wait_for_timeout(3000)
        batch_filter = page.locator('text=See all options').first
        await batch_filter.click()
        """
        t = _t(code)
        assert t >= BASELINE


class TestPW_WrongDateSelector:
    """Date picker interactions with wrong selectors."""

    def test_date_button_long_aria_label(self):
        """Source: traces/run_2026-04-20_17-51-24"""
        code = """
        await page.locator('div[role="button"]').filter(has_text='Wann').click()
        await page.get_by_role('button', name='20, Monday, April 2026, heute. Verfügbar').click()
        await page.get_by_role('button', name='19, Tuesday, May 2026. Verfügbar').click()
        await page.get_by_test_id('structured-search-input-search-button').click()
        await page.wait_for_load_state('domcontentloaded')
        await show_page(page)
        """
        t = _t(code)
        # click(5)*4 + wait_load(8) + show_page(15) = 43 * 1.2 = 51.6
        assert t >= 50.0

    def test_date_button_german_montag(self):
        """Source: traces/run_2026-04-20_08-13-29"""
        code = """
        await page.click('div[role="button"]:has-text("Wann")')
        await page.click('button[aria-label*="20, Montag, April 2026"]')
        """
        t = _t(code)
        assert t >= BASELINE

    def test_date_button_german_april(self):
        """Source: traces/run_2026-04-20_11-26-05"""
        code = """
        location_input = page.locator('input[data-testid="structured-search-input-field-query"]')
        await location_input.fill('Berlin')
        await page.wait_for_timeout(500)
        date_picker = page.locator('div[aria-expanded="false"]').filter(has_text="Wann").first
        await date_picker.click()
        await page.wait_for_timeout(1000)
        april_20_button = page.locator('button[aria-label*="20. April"]').first
        await april_20_button.click()
        """
        t = _t(code)
        assert t >= BASELINE

    def test_date_mai_selector(self):
        """Source: traces/run_2026-04-20_10-23-29, run_2026-04-20_10-48-19"""
        code = """
        await page.locator("button[aria-label^='19,'][aria-label*='Mai 2026']").click()
        """
        t = _t(code)
        assert t >= BASELINE

    def test_calendar_day_testid(self):
        """Source: traces/run_2026-04-20_14-04-41"""
        code = """
        date_btn = page.locator('div[role="button"]:has-text("Wann")')
        await date_btn.click()
        await asyncio.sleep(0.5)
        await page.locator('td[data-testid="calendar-day-2026-04-20"]').click()
        """
        t = _t(code)
        assert t >= BASELINE

    def test_disabled_date_button(self):
        """Source: traces/run_2026-04-21_08-54-27"""
        code = """
        checkin = await page.locator('button[data-state--date-string="2026-04-20"]').count()
        checkout = await page.locator('button[data-state--date-string="2026-05-19"]').count()
        if checkin:
            await page.locator('button[data-state--date-string="2026-04-20"]').first.click(timeout=5000)
        if checkout:
            await page.locator('button[data-state--date-string="2026-05-19"]').first.click(timeout=5000)
        await show_page(page)
        """
        t = _t(code)
        assert t >= BASELINE


class TestPW_WaitForSelectorWrong:
    """wait_for_selector / wait_for with wrong selectors."""

    def test_wait_for_dialog(self):
        """Source: traces/run_2026-04-20_13-59-19, run_2026-04-19_15-18-09, run_2026-04-19_15-15-11"""
        code = """
        await page.locator('div[role="button"]:has-text("Wann")').click()
        await page.wait_for_selector('div[role="dialog"]', state='visible', timeout=10000)
        await zoom_section(page, 'div[role="dialog"]')
        """
        t = _t(code)
        # click(5) + wait(10) + zoom(5) = 20 * 1.2 = 24
        assert t >= 24.0

    def test_wait_for_grid(self):
        """Source: traces/run_2026-04-20_00-47-35"""
        code = """
        wann_button = page.locator('div[role="button"]:has-text("Wann")').first
        await wann_button.click()
        await page.wait_for_selector('div[role="grid"]', state='visible', timeout=10000)
        await zoom_section(page, "div[role='grid']")
        """
        t = _t(code)
        assert t >= 24.0

    def test_wait_for_calendar_day(self):
        """Source: traces/run_2026-04-20_10-38-37, run_2026-04-20_00-47-35"""
        code = """
        await page.locator('div[role="button"]', has_text='Wann').click()
        await page.wait_for_selector('[data-testid="calendar-day"]', timeout=5000)
        await show_page(page)
        """
        t = _t(code)
        # click(5) + wait(5) + show(15) = 25 * 1.2 = 30
        assert t >= 30.0

    def test_wait_for_search_results(self):
        """Source: traces/run_2026-04-19_15-18-09, run_2026-04-20_00-48-27"""
        code = """
        await page.wait_for_selector('div[data-testid="search-results"]', timeout=15000)
        await show_page(page)
        """
        t = _t(code)
        # wait(15) + show(15) = 30 * 1.2 = 36
        assert t >= 36.0

    def test_wait_for_button_detached(self):
        """Source: traces/run_2026-04-20_01-04-54, run_2026-04-19_16-36-43"""
        code = """
        await page.locator('button', has_text='Alle akzeptieren').click()
        await page.wait_for_selector('button', state='detached', timeout=10000)
        await show_page(page)
        """
        t = _t(code)
        # click(5) + wait(10) + show(15) = 30 * 1.2 = 36
        assert t >= 36.0

    def test_wait_for_hidden_script(self):
        """Source: traces/run_2026-04-21_08-54-27"""
        code = """
        await page.wait_for_selector('#data-deferred-state-0', timeout=30000)
        state = await page.evaluate(
            "() => JSON.parse(document.querySelector('#data-deferred-state-0').textContent)",
            isolated_context=True,
        )
        """
        t = _t(code)
        # wait(30) + evaluate(3) = 33 * 1.2 = 39.6
        assert t >= 39.0

    def test_wait_for_filter_panel(self):
        """Source: traces/run_2026-04-20_00-53-24"""
        code = """
        await page.locator("button[data-testid='category-bar-filter-button']").click()
        await page.wait_for_selector('[data-testid="filter-panel"]', state='visible', timeout=10000)
        await show_page(page)
        """
        t = _t(code)
        assert t >= 30.0

    def test_wait_for_aria_expanded(self):
        """Source: traces/run_2026-04-20_14-04-41"""
        code = """
        date_btn = page.locator('div[role="button"]:has-text("Wann")')
        await date_btn.click()
        await page.locator('div[role="button"]:has-text("Wann")[aria-expanded="true"]').wait_for(
            state="visible", timeout=5000)
        """
        t = _t(code)
        assert t >= BASELINE

    def test_wait_for_text_april(self):
        """Source: traces/run_2026-04-20_00-47-35"""
        code = """
        wann_button = page.locator('div[role="button"]:has-text("Wann")').first
        await wann_button.click()
        await page.wait_for_selector('text="April 2026"', state='visible', timeout=10000)
        """
        t = _t(code)
        assert t >= BASELINE

    def test_wait_for_listings_container(self):
        """Source: traces/run_2026-04-19_22-49-35"""
        code = """
        await page.wait_for_selector('div[role="main"] div[aria-label*="Ergebnisse"]',
                                     state='visible', timeout=30000)
        await zoom_section(page, 'div[role="main"]')
        """
        t = _t(code)
        # wait(30) + zoom(5) = 35 * 1.2 = 42
        assert t >= 42.0

    def test_wait_for_suggestion_dropdown(self):
        """Source: traces/run_2026-04-20_08-13-29"""
        code = """
        await page.click('#bigsearch-query-location-input')
        await page.fill('#bigsearch-query-location-input', 'Berlin')
        await page.wait_for_selector('[data-testid="structured-search-input-field-suggestion"]',
                                     timeout=5000)
        await show_page(page)
        """
        t = _t(code)
        # click(5) + fill(3) + wait(5) + show(15) = 28 * 1.2 = 33.6
        assert t >= 33.0

    def test_wait_for_aria_labels(self):
        """Source: traces/run_2026-04-19_15-18-09"""
        code = """
        date_button = page.get_by_role('button', name='Wann')
        await date_button.click()
        await page.wait_for_selector('[aria-label*="April 2026"], [aria-label*="Mai 2026"]',
                                     timeout=10000)
        """
        t = _t(code)
        assert t >= BASELINE


class TestPW_ClickInvisibleElement:
    """Clicking elements that exist but are not visible/enabled."""

    def test_close_button_not_visible(self):
        """Source: traces/run_2026-04-20_20-27-49, run_2026-04-20_20-27-52,
        run_2026-04-18_12-32-52, run_2026-04-19_15-31-55, run_2026-04-20_00-53-24"""
        code = """
        await page.click('button[aria-label="Schließen"]')
        await page.wait_for_timeout(1000)
        await show_page(page)
        """
        t = _t(code)
        assert t >= BASELINE

    def test_stepper_not_visible(self):
        """Source: traces/run_2026-04-20_11-26-05"""
        code = """
        adults_button = page.locator('[data-testid="stepper-adults-increase-button"]')
        await adults_button.click()
        await adults_button.click()
        children_button = page.locator('[data-testid="stepper-children-increase-button"]')
        await children_button.click()
        await children_button.click()
        infants_button = page.locator('[data-testid="stepper-infants-increase-button"]')
        await infants_button.click()
        await page.wait_for_timeout(500)
        await show_page(page)
        """
        t = _t(code)
        # click(5)*5 + wait(0.5) + show(15) = 40.5 * 1.2 = 48.6
        assert t >= 48.0

    def test_stepper_with_wait_for(self):
        """Source: traces/run_2026-04-20_11-26-05"""
        code = """
        adults_button = page.locator('button[data-testid="stepper-adults-increase-button"]')
        await adults_button.wait_for(state="visible", timeout=5000)
        await adults_button.click()
        await page.wait_for_timeout(300)
        await adults_button.click()
        await page.wait_for_timeout(300)
        children_button = page.locator('button[data-testid="stepper-children-increase-button"]')
        await children_button.wait_for(state="visible", timeout=5000)
        await children_button.click()
        await page.wait_for_timeout(300)
        await children_button.click()
        await page.wait_for_timeout(300)
        infants_button = page.locator('button[data-testid="stepper-infants-increase-button"]')
        await infants_button.wait_for(state="visible", timeout=5000)
        await infants_button.click()
        await page.wait_for_timeout(300)
        await show_page(page)
        """
        t = _t(code)
        # wait_for(5)*3 + click(5)*5 + wait(0.3)*5 + show(15) = 56.5 * 1.2 = 67.8
        assert t >= 67.0

    def test_amenities_header_not_visible(self):
        """Source: traces/run_2026-04-20_14-04-41"""
        code = """
        amenities_header = await page.query_selector('div:has-text("Ausstattung")')
        if amenities_header:
            await amenities_header.click()
        import asyncio
        await asyncio.sleep(1)
        inputs = await page.evaluate('''
            () => Array.from(document.querySelectorAll('input')).map(el => ({
                type: el.type, testid: el.getAttribute('data-testid')
            }))
        ''')
        for inp in inputs:
            print(inp)
        """
        t = _t(code)
        assert t >= BASELINE

    def test_element_handle_not_visible(self):
        """Source: benchmark/traces/run_2026-05-10_17-38-53"""
        code = """
        month_link = await page.query_selector('a[href*="since=monthly"]')
        if month_link:
            await month_link.click()
            await page.wait_for_load_state("domcontentloaded")
            await page.wait_for_timeout(1000)
        """
        t = _t(code)
        assert t >= BASELINE

    def test_babies_element_handle(self):
        """Source: traces/run_2026-04-20_13-40-25"""
        code = """
        babies_label_el = await page.locator('div:has-text("Babys")').first.element_handle()
        babies_parent_el = await babies_label_el.evaluate_handle('el => el.parentElement')
        babies_buttons = await babies_parent_el.evaluate(
            'el => Array.from(el.querySelectorAll("button")).map(b => b.getAttribute("aria-label"))')
        """
        t = _t(code)
        assert t >= BASELINE

    def test_guest_picker_toggle(self):
        """Source: traces/run_2026-04-19_22-49-35"""
        code = """
        guest_picker_toggle = page.locator('div[role="button"][aria-expanded][aria-expanded="false"]',
                                           has_text="Wer")
        await guest_picker_toggle.click()
        await page.wait_for_timeout(1000)
        """
        t = _t(code)
        assert t >= BASELINE

    def test_disabled_weiter_button(self):
        """Source: traces/run_2026-04-20_00-47-35"""
        code = """
        next_button = page.locator('button[aria-label="Weiter"]')
        await next_button.click()
        """
        t = _t(code)
        assert t >= BASELINE


class TestPW_NavigationTimeout:
    """go_back, goto, and networkidle timeouts."""

    def test_go_back_explicit_timeout(self):
        """Source: traces/run_2026-04-21_08-43-54"""
        code = """
        await page.go_back(wait_until="domcontentloaded", timeout=30_000)
        await page.wait_for_selector("article.product_pod", timeout=30_000)
        """
        t = _t(code)
        # go_back(8) + wait_for_selector(30) = 38 * 1.2 = 45.6
        assert t >= 45.0

    def test_go_back_yc(self):
        """Source: benchmark/traces/run_2026-05-09_17-21-30"""
        code = """
        await page.go_back(wait_until='domcontentloaded', timeout=30000)
        """
        t = _t(code)
        assert t >= BASELINE

    def test_go_back_yc_default(self):
        """Source: benchmark/traces/run_2026-05-09_18-51-00"""
        code = """
        await page.go_back()
        await page.wait_for_load_state("domcontentloaded")
        await show_page(page)
        """
        t = _t(code)
        # go_back(8) + wait_load(8) + show(15) = 31 * 1.2 = 37.2
        assert t >= 37.0

    def test_goto_after_error(self):
        """Source: benchmark/traces/run_2026-05-10_17-38-53"""
        code = """
        await page.goto("https://github.com/trending/python?since=weekly",
                         wait_until="domcontentloaded")
        await page.wait_for_timeout(1000)
        repos = await page.evaluate("() => []", isolated_context=True)
        """
        t = _t(code)
        assert t >= BASELINE

    def test_networkidle_after_search(self):
        """Source: traces/run_2026-04-20_00-48-27, run_2026-04-19_22-49-35"""
        code = """
        search_button = page.locator('button[data-testid="structured-search-input-search-button"]')
        await search_button.click()
        await page.wait_for_load_state('networkidle')
        await show_page(page)
        """
        t = _t(code)
        # click(5) + wait_load(8) + show(15) = 28 * 1.2 = 33.6
        assert t >= 33.0

    def test_networkidle_after_goto(self):
        """Source: traces/run_2026-04-19_15-18-09"""
        code = """
        url = "https://www.example-travel.com/s/Berlin/homes?check_in=2026-04-20&check_out=2026-05-19"
        await page.goto(url, wait_until="domcontentloaded")
        await page.wait_for_load_state("networkidle")
        await show_page(page)
        """
        t = _t(code)
        # goto(10) + wait_load(8) + show(15) = 33 * 1.2 = 39.6
        assert t >= 39.0

    def test_search_div_main(self):
        """Source: traces/run_2026-04-19_15-15-11"""
        code = """
        search_button = page.get_by_role("button", name="Suche")
        await search_button.click()
        await page.wait_for_selector('div[role="main"]')
        await show_page(page)
        """
        t = _t(code)
        assert t >= BASELINE


class TestPW_InnerTextMissing:
    """inner_text / text_content / get_attribute on wrong selectors."""

    def test_realestate_advertiser(self):
        """Source: benchmark/traces/run_2026-05-10_14-34-55"""
        code = """
        import re
        location_text = await page.inner_text('div#mapWrapper')
        description = await page.inner_text('div.commentsContainer div.comment p')
        features_text = await page.inner_text('section#details')
        agent_text = await page.inner_text('div#ask-the-advertiser')
        """
        t = _t(code)
        # inner_text(5) * 4 = 20 * 1.2 = 24
        assert t >= 24.0

    def test_yc_sidebar_inner_text(self):
        """Source: benchmark/traces/run_2026-05-09_18-51-00, run_2026-05-09_16-03-53"""
        code = """
        sidebar_text = await page.inner_text('div.ycdc-card-new.space-y-1\\\\.5')
        """
        t = _t(code)
        assert t >= BASELINE

    def test_jobs_count_inner_text(self):
        """Source: src/traces/run_2026-04-17_20-17-10"""
        code = """
        load_more_button = await page.query_selector('#ergebnisliste-ladeweitere-button')
        total_jobs_text = await page.inner_text('div[id*="136-jobs"]')
        """
        t = _t(code)
        assert t >= BASELINE

    def test_wayback_get_attribute(self):
        """Source: benchmark/traces/run_2026-05-09_17-33-49"""
        code = """
        link = await page.get_attribute('a.s2xx', 'href')
        await page.goto('https://web.archive.org' + link, wait_until='domcontentloaded')
        await show_page(page)
        """
        t = _t(code)
        assert t >= BASELINE

    def test_card_text_content_in_loop(self):
        """Source: traces/run_2026-04-20_01-36-23"""
        code = """
        import re
        cards = await page.locator('[data-testid="card-container"]').all()
        results = []
        for card in cards:
            url = await card.locator('a[href*="/rooms/"]').first.get_attribute('href')
            title = await card.locator('[data-testid="listing-card-title"]').text_content()
            description = await card.locator('[data-testid="listing-card-name"]').text_content()
            subtitle_texts = await card.locator('[data-testid="listing-card-subtitle"]').all_text_contents()
            price_text = await card.locator('[data-testid="price-availability-row"] [aria-label]').first.text_content()
            all_spans = await card.locator('span').all_text_contents()
            results.append({'title': title, 'description': description, 'url': url})
        """
        t = _t(code)
        # for cards (5 iters): get_attribute(5) + text_content(5)*3 + all_text_contents(5)*2 = 30
        # 5*30 = 150 * 1.2 = 180
        assert t >= 150.0

    def test_listing_h3_inner_text(self):
        """Source: traces/run_2026-04-20_00-47-35"""
        code = """
        listings = await page.locator('div[data-testid="content-scroller"] > div > div').all()
        items = []
        for listing in listings[:3]:
            title = await listing.locator('h3').inner_text()
            description = await listing.locator('div').nth(1).inner_text()
            bedrooms_text = await listing.locator('div:has-text("Schlafzimmer")').inner_text()
            beds_text = await listing.locator('div:has-text("Betten")').inner_text()
            price_text = await listing.locator('span:has-text("€")').inner_text()
            rating_text = await listing.locator('span[aria-label*="Bewertung"] span').inner_text()
            url = await listing.locator('a').get_attribute('href')
        """
        t = _t(code)
        # for listings (5 iters): inner_text(5)*6 + get_attribute(5) = 35
        # 5*35 = 175 * 1.2 = 210
        assert t >= 200.0

    def test_property_card_text_content(self):
        """Source: traces/run_2026-04-19_22-49-35"""
        code = """
        first_card_text = await page.locator(
            'div.atm_yeapsr_ztxlnh[role="group"] div[data-testid="property-card"]'
        ).first.text_content()
        """
        t = _t(code)
        assert t >= BASELINE

    def test_listing_card_price_evaluate_handle(self):
        """Source: traces/run_2026-04-20_00-47-35"""
        code = """
        listings = await page.locator('div[data-testid="card-container"]').all()
        items = []
        for listing in listings[:3]:
            title = await listing.locator('div[data-testid="listing-card-title"]').inner_text()
            price_label_locator = listing.locator('span:has-text("Gesamtpreis")').first
            price_span = await price_label_locator.evaluate_handle('el => el.nextElementSibling')
        """
        t = _t(code)
        assert t >= BASELINE

    def test_fill_wrong_placeholder(self):
        """Source: traces/run_2026-04-20_11-06-45"""
        code = """
        await page.fill('input[placeholder*="Ort"]', "Berlin")
        await page.wait_for_timeout(500)
        """
        t = _t(code)
        assert t >= BASELINE


class TestPW_WaitForFunction:
    """wait_for_function timeouts (pagination polling)."""

    def test_wait_for_first_title_change(self):
        """Source: traces/run_2026-04-20_00-47-35"""
        code = (
            "old_first_title = await page.locator('div[data-testid=\"listing-card-title\"]').first.inner_text()\n"
            "await page.wait_for_function(\n"
            '    "(oldTitle) => true",\n'
            "    arg=old_first_title,\n"
            "    timeout=15000\n"
            ")\n"
        )
        t = predict_timeout(code)
        # inner_text(5) + wait_for_function(15) = 20 * 1.2 = 24
        assert t >= 24.0

    def test_wait_for_card_detach(self):
        """Source: traces/run_2026-04-20_00-47-35"""
        code = """
        listings_locator = page.locator('div[data-testid="card-container"]')
        count_before = await listings_locator.count()
        next_button = page.locator('button[aria-label="Weiter"]')
        await next_button.click()
        await listings_locator.first.wait_for(state='detached', timeout=15000)
        await listings_locator.first.wait_for(state='attached', timeout=15000)
        """
        t = _t(code)
        # count(1) + click(5) + wait_for(15) + wait_for(15) = 36 * 1.2 = 43.2
        assert t >= 43.0

    def test_scrape_pagination_wait_for_function(self):
        """Source: traces/run_2026-04-20_01-04-54"""
        code = """
        nxt = page.locator('a[aria-label="Weiter"]')
        cards = await page.locator('div[data-testid="card-container"]').all()
        if cards:
            old_first_title = await cards[0].locator(
                'div[data-testid="listing-card-title"]').inner_text()
            await nxt.first.click()
            await page.wait_for_selector('div[data-testid="card-container"]', timeout=20000)
            for _ in range(20):
                await page.wait_for_timeout(500)
        """
        t = _t(code)
        # inner_text(5) + click(5) + wait_for_selector(20) + range(20)*(wait(0.5)) = 40
        # loop: 20*0.5 = 10, capped or not depending on range extraction
        assert t >= BASELINE


# ═══════════════════════════════════════════════════════════════════
#  3. OVERLAY / MODAL INTERFERENCE
#
#  An overlay (scout-overlay, cookie disclaimer, modal) blocks
#  all Playwright click actions.  The prediction is usually
#  adequate; the problem is purely the overlay.
# ═══════════════════════════════════════════════════════════════════


class TestOverlay_ScoutOverlay:
    """Scout demo overlay (div#scout-overlay) intercepts pointer events.
    Source: benchmark/traces/run_2026-05-10_00-19-06, run_2026-05-10_00-09-55,
    run_2026-05-10_00-05-27, run_2026-05-09_23-57-52, run_2026-05-09_23-28-59,
    run_2026-05-09_23-00-02, run_2026-05-09_22-48-08, run_2026-05-09_22-38-08,
    run_2026-05-09_22-04-47
    """

    def test_click_listing_link(self):
        code = """
        await page.click('a[href="/en/inmueble/106183974/"]')
        await page.wait_for_load_state("domcontentloaded")
        await page.wait_for_timeout(2000)
        await show_page(page)
        """
        t = _t(code)
        # click(5) + wait_load(8) + wait(2) + show(15) = 30 * 1.2 = 36
        assert t >= 36.0

    def test_click_item_link(self):
        code = """
        await page.click('a.item-link')
        await page.wait_for_load_state('domcontentloaded')
        await page.wait_for_timeout(2000)
        await show_page(page)
        """
        t = _t(code)
        assert t >= 36.0

    def test_click_heading_link(self):
        code = """
        await page.click('a[role="heading"]')
        await page.wait_for_load_state('domcontentloaded')
        await page.wait_for_timeout(2000)
        await show_page(page)
        """
        t = _t(code)
        assert t >= 36.0

    def test_photo_gallery_loop(self):
        """Source: benchmark/traces/run_2026-05-09_22-38-08, run_2026-05-09_23-28-59"""
        code = """
        for i in range(39):
            await page.click('button.image-gallery-right-nav')
            await page.wait_for_timeout(300)
            current_photos = await page.evaluate("() => []", isolated_context=True)
        """
        t = _t(code)
        # range(39): click(5) + wait(0.3) + evaluate(3) = 8.3
        # 39*8.3 = 323.7 → capped 225 * 1.2 = 270
        assert t >= 270.0

    def test_exit_button_not_visible(self):
        """Source: benchmark/traces/run_2026-05-09_23-00-02, run_2026-05-10_00-19-06"""
        code = """
        exit_button = await page.query_selector('text="Not now"')
        if exit_button:
            await exit_button.click()
            await page.wait_for_timeout(1000)
        else:
            await page.evaluate("document.getElementById('scout-overlay').style.display = 'none'",
                                isolated_context=True)
            await page.wait_for_timeout(500)
        await show_page(page)
        """
        t = _t(code)
        assert t >= BASELINE


class TestOverlay_CookieDisclaimer:
    """Cookie disclaimer web component blocks clicks.
    Source: src/traces/run_2026-04-18_08-07-13, run_2026-04-17_20-17-10
    """

    def test_click_suggestion(self):
        code = """
        await page.click('#wo-vorschlagsliste0')
        await page.wait_for_timeout(500)
        await page.click('#umkreis-dropdown-item-4')
        await page.wait_for_timeout(500)
        await show_page(page)
        """
        t = _t(code)
        # click(5)*2 + wait(0.5)*2 + show(15) = 26 * 1.2 = 31.2
        assert t >= 31.0

    def test_click_radius_dropdown(self):
        code = """
        await page.click('#umkreis-dropdown-button')
        await page.wait_for_timeout(300)
        await page.click('#umkreis-dropdown-item-4')
        await page.wait_for_timeout(1000)
        await show_page(page)
        """
        t = _t(code)
        assert t >= BASELINE

    def test_load_more_element_handle(self):
        """Source: src/traces/run_2026-04-17_20-17-10"""
        code = """
        load_more = await page.query_selector('#ergebnisliste-ladeweitere-button')
        if load_more:
            is_visible = await load_more.is_visible()
            if is_visible:
                await load_more.click()
                await page.wait_for_timeout(1500)
        """
        t = _t(code)
        assert t >= BASELINE

    def test_load_more_locator(self):
        """Source: src/traces/run_2026-04-17_20-17-10"""
        code = """
        await page.locator('#ergebnisliste-ladeweitere-button').click()
        await page.wait_for_timeout(2000)
        """
        t = _t(code)
        assert t >= BASELINE


class TestOverlay_ModalContainer:
    """Modal container intercepts filter button click.
    Source: traces/run_2026-04-19_15-31-55, run_2026-04-20_00-53-24
    """

    def test_filter_button_blocked(self):
        code = """
        await page.click('button[data-testid="category-bar-filter-button"]')
        await page.wait_for_selector('[aria-modal="true"]', state='visible', timeout=5000)
        await show_page(page)
        """
        t = _t(code)
        # click(5) + wait(5) + show(15) = 25 * 1.2 = 30
        assert t >= 30.0

    def test_scroll_top_then_filter(self):
        """Source: traces/run_2026-04-19_15-31-55"""
        code = """
        await page.evaluate("window.scrollTo(0, 0)")
        await page.wait_for_timeout(1000)
        await page.click('button[data-testid="category-bar-filter-button"]')
        await page.wait_for_selector('div[role="dialog"]', state='visible', timeout=5000)
        await show_page(page)
        """
        t = _t(code)
        # evaluate(3) + wait(1) + click(5) + wait(5) + show(15) = 29 * 1.2 = 34.8
        assert t >= 34.0

    def test_reclick_filter_overlay(self):
        """Source: traces/run_2026-04-20_00-53-24"""
        code = """
        await page.locator("button[data-testid='category-bar-filter-button']").click()
        await show_page(page)
        """
        t = _t(code)
        # click(5) + show(15) = 20 * 1.2 = 24
        assert t >= 24.0


class TestPW_BackNavigation:
    """Wrong selector for back navigation buttons."""

    def test_arrow_back_testid(self):
        """Source: src/traces/run_2026-04-17_20-07-03"""
        code = """
        current_month = await page.evaluate("() => document.querySelector('header p')?.innerText || ''",
                                             isolated_context=True)
        for i in range(3):
            await page.click('button[data-testid="ArrowBackIosIcon"]')
            await page.wait_for_load_state("domcontentloaded")
            await page.wait_for_timeout(500)
        await show_page(page)
        """
        t = _t(code)
        # evaluate(3) + range(3)*(click(5)+wait_load(8)+wait(0.5)) = 3 + 3*13.5 = 43.5 * 1.2 = 52.2
        assert t >= 52.0

    def test_realestate_listing_click(self):
        """Source: benchmark/traces/run_2026-05-09_19-40-07"""
        code = """
        await page.click('a[href="/en/inmueble/111422734/"]')
        await page.wait_for_load_state("domcontentloaded")
        await page.wait_for_timeout(2000)
        await show_page(page)
        """
        t = _t(code)
        assert t >= 36.0

    def test_split_destination_button(self):
        """Source: traces/run_2026-04-20_08-13-29"""
        code = """
        await page.click('button[data-testid="structured-search-input-field-split-destination-button"]')
        """
        t = _t(code)
        assert t >= BASELINE

    def test_cookie_consent_in_banner(self):
        """Source: traces/run_2026-04-20_01-04-54, run_2026-04-19_16-36-43,
        run_2026-04-20_00-47-35"""
        code = """
        banner = page.locator('[data-testid="main-cookies-banner-container"]')
        await banner.locator('button', has_text='Alle akzeptieren').click()
        await banner.wait_for(state='detached', timeout=10000)
        await show_page(page)
        """
        t = _t(code)
        # click(5) + wait_for(10) + show(15) = 30 * 1.2 = 36
        assert t >= 36.0

    def test_date_picker_wann_on_results(self):
        """Source: traces/run_2026-04-20_16-28-34, run_2026-04-20_10-23-29"""
        code = """
        date_button = page.locator('div[role="button"]').filter(has_text="Wann")
        await date_button.click()
        """
        t = _t(code)
        assert t >= BASELINE

    def test_little_search_date(self):
        """Source: traces/run_2026-04-20_13-59-19, run_2026-04-20_16-28-34"""
        code = """
        await page.locator('button[data-testid="little-search-date"]').click()
        await page.wait_for_timeout(1000)
        await show_page(page)
        """
        t = _t(code)
        assert t >= BASELINE

    def test_date_input_fill(self):
        """Source: traces/run_2026-04-20_16-28-34"""
        code = """
        checkin_input = page.locator('input[placeholder*="Ankunft"]').first
        await checkin_input.fill('20.04.2026')
        """
        t = _t(code)
        assert t >= BASELINE

    def test_is_enabled_check(self):
        """Source: traces/run_2026-04-20_13-59-19"""
        code = """
        date_button = page.locator('button[data-testid="little-search-date"]')
        is_visible = await date_button.is_visible()
        is_enabled = await date_button.is_enabled()
        """
        t = _t(code)
        assert t >= BASELINE

    def test_scroll_into_view(self):
        """Source: traces/run_2026-04-20_13-59-19"""
        code = """
        date_button = page.locator('button[data-testid="little-search-date"]')
        await date_button.scroll_into_view_if_needed()
        await date_button.click()
        await page.wait_for_timeout(2000)
        await show_page(page)
        """
        t = _t(code)
        assert t >= BASELINE

    def test_guest_picker_dialog_wait(self):
        """Source: traces/run_2026-04-20_13-40-25"""
        code = """
        await page.click('div[role="button"][aria-expanded="false"] >> text=Wer')
        await page.wait_for_selector('div[role="dialog"]', state='visible', timeout=5000)
        await zoom_section(page, 'div[role="dialog"]')
        """
        t = _t(code)
        # click(5) + wait(5) + zoom(5) = 15 * 1.2 = 18 → BASELINE
        assert t >= BASELINE

    def test_date_picker_expanded(self):
        """Source: traces/run_2026-04-20_16-15-22"""
        code = """
        await page.click('div[aria-expanded="false"]:has-text("Wann")')
        await page.wait_for_timeout(1000)
        await show_page(page)
        """
        t = _t(code)
        assert t >= BASELINE
