"""Tests for the URL Parameter Detector.

Covers fill-value extraction from AI-generated code and URL parameter
matching — driven by realistic Playwright code patterns.
"""

from __future__ import annotations

from scout.agent.param_detector import (
    FillValue,
    detect_url_params,
    extract_fill_values,
    format_hint,
    format_hint_complex,
)

# ═══════════════════════════════════════════════════════════════════════
#  Test data — fill value extraction
# ═══════════════════════════════════════════════════════════════════════

# ── Direct page.fill / page.type / page.select_option ────────────────

DIRECT_FILL = """\
await page.fill('#bigsearch-query-location-input', 'Berlin')
await page.wait_for_timeout(1000)
"""

DIRECT_TYPE = """\
await page.type('#search-input', "laptop stand")
await page.wait_for_timeout(500)
"""

DIRECT_SELECT = """\
await page.select_option('#guests', '2')
"""

DIRECT_FILL_DOUBLE_QUOTES = """\
await page.fill("#location", "New York")
"""

DIRECT_MULTIPLE_FILLS = """\
await page.fill('#location', 'Berlin')
await page.fill('#checkin', '2026-06-01')
await page.fill('#checkout', '2026-06-05')
await page.select_option('#adults', '2')
"""

# ── Locator chain calls ──────────────────────────────────────────────

LOCATOR_FILL = """\
await page.locator('#search-box').fill('Berlin apartments')
await page.locator('button[type="submit"]').click()
"""

LOCATOR_TYPE = """\
await page.locator('input[name="q"]').type("gaming mouse")
"""

LOCATOR_SELECT_OPTION = """\
await page.locator('#sort-by').select_option('price_asc')
"""

LOCATOR_PRESS_SEQUENTIALLY = """\
await page.locator('#search').press_sequentially("Berlin")
"""

# ── get_by_* chains ──────────────────────────────────────────────────

GETBY_FILL = """\
await page.get_by_placeholder('Where are you going?').fill('Berlin')
await page.get_by_role('button', name='Search').click()
"""

GETBY_LABEL_FILL = """\
await page.get_by_label('Destination').fill('Paris')
"""

# ── Variable tracking ────────────────────────────────────────────────

VARIABLE_FILL = """\
search_input = page.locator('#location')
await search_input.fill('Berlin')
await search_input.press('Enter')
"""

VARIABLE_CHAIN = """\
loc = page.get_by_placeholder('Search')
await loc.fill('cheap flights')
await loc.press('Enter')
"""

# ── page.keyboard.type ───────────────────────────────────────────────

KEYBOARD_TYPE = """\
await page.locator('#search').click()
await page.keyboard.type('Berlin hotels')
await page.keyboard.press('Enter')
"""

# ── select_option with keyword arg ───────────────────────────────────

SELECT_KW_VALUE = """\
await page.locator('#currency').select_option(value="EUR")
"""

SELECT_KW_LABEL = """\
await page.locator('#sort').select_option(label="Price: low to high")
"""

# ── Edge cases ────────────────────────────────────────────────────────

COMMENTED_OUT = """\
# await page.fill('#location', 'Berlin')
await page.fill('#location', 'Munich')
"""

FSTRING_VALUE = """\
city = "Berlin"
await page.fill('#location', f'City: {city}')
"""

VARIABLE_VALUE = """\
city = "Berlin"
await page.fill('#location', city)
"""

MULTILINE_CALL = """\
await page.fill(
    '#bigsearch-query-location-input',
    'Berlin'
)
"""

MIXED_PATTERNS = """\
await page.fill('#location', 'Berlin')
await page.locator('#checkin').fill('2026-06-01')
await page.get_by_label('Checkout').fill('2026-06-05')
guests_select = page.locator('#guests')
await guests_select.select_option('2')
await page.keyboard.type('pool')
"""

NO_FILLS = """\
await page.click('button.search')
await page.wait_for_load_state('networkidle')
results = await page.query_selector_all('.result-card')
"""

FRAME_LOCATOR_CHAIN = """\
await page.frame_locator('#booking-frame').locator('#dest').fill('Rome')
"""

NTH_LOCATOR = """\
await page.locator('input.search').nth(0).fill('Berlin')
"""

FIRST_LOCATOR = """\
await page.locator('input.search').first.fill('Berlin')
"""

# ── inputs["key"] references ─────────────────────────────────────────

INPUTS_DIRECT = """\
await page.fill('#location', inputs['destination'])
await page.fill('#checkin', inputs["check_in"])
"""

INPUTS_LOCATOR_CHAIN = """\
await page.locator('#search-box').fill(inputs['destination'])
"""

INPUTS_STR_WRAP = """\
await page.fill('#adults', str(inputs['adults']))
"""

INPUTS_MIXED = """\
await page.fill('#location', inputs['destination'])
await page.fill('#checkin', '2026-06-01')
await page.locator('#guests').select_option(str(inputs['adults']))
"""


# ═══════════════════════════════════════════════════════════════════════
#  Tests — extract_fill_values
# ═══════════════════════════════════════════════════════════════════════


class TestExtractFillValues:
    """Tests for extract_fill_values()."""

    def test_direct_fill(self):
        fills = extract_fill_values(DIRECT_FILL)
        assert len(fills) == 1
        assert fills[0].action == "fill"
        assert fills[0].value == "Berlin"
        assert fills[0].line == 1

    def test_direct_type(self):
        fills = extract_fill_values(DIRECT_TYPE)
        assert len(fills) == 1
        assert fills[0].action == "type"
        assert fills[0].value == "laptop stand"

    def test_direct_select_option(self):
        fills = extract_fill_values(DIRECT_SELECT)
        assert len(fills) == 1
        assert fills[0].action == "select_option"
        assert fills[0].value == "2"

    def test_direct_double_quotes(self):
        fills = extract_fill_values(DIRECT_FILL_DOUBLE_QUOTES)
        assert len(fills) == 1
        assert fills[0].value == "New York"

    def test_multiple_fills(self):
        fills = extract_fill_values(DIRECT_MULTIPLE_FILLS)
        assert len(fills) == 4
        assert [f.value for f in fills] == [
            "Berlin",
            "2026-06-01",
            "2026-06-05",
            "2",
        ]

    def test_locator_fill(self):
        fills = extract_fill_values(LOCATOR_FILL)
        assert len(fills) == 1
        assert fills[0].action == "fill"
        assert fills[0].value == "Berlin apartments"

    def test_locator_type(self):
        fills = extract_fill_values(LOCATOR_TYPE)
        assert len(fills) == 1
        assert fills[0].value == "gaming mouse"

    def test_locator_select_option(self):
        fills = extract_fill_values(LOCATOR_SELECT_OPTION)
        assert len(fills) == 1
        assert fills[0].value == "price_asc"

    def test_locator_press_sequentially(self):
        fills = extract_fill_values(LOCATOR_PRESS_SEQUENTIALLY)
        assert len(fills) == 1
        assert fills[0].action == "press_sequentially"
        assert fills[0].value == "Berlin"

    def test_getby_fill(self):
        fills = extract_fill_values(GETBY_FILL)
        assert len(fills) == 1
        assert fills[0].value == "Berlin"

    def test_getby_label_fill(self):
        fills = extract_fill_values(GETBY_LABEL_FILL)
        assert len(fills) == 1
        assert fills[0].value == "Paris"

    def test_variable_fill(self):
        fills = extract_fill_values(VARIABLE_FILL)
        assert len(fills) == 1
        assert fills[0].value == "Berlin"

    def test_variable_chain(self):
        fills = extract_fill_values(VARIABLE_CHAIN)
        assert len(fills) == 1
        assert fills[0].value == "cheap flights"

    def test_keyboard_type(self):
        fills = extract_fill_values(KEYBOARD_TYPE)
        assert len(fills) == 1
        assert fills[0].action == "keyboard.type"
        assert fills[0].value == "Berlin hotels"

    def test_select_kw_value(self):
        fills = extract_fill_values(SELECT_KW_VALUE)
        assert len(fills) == 1
        assert fills[0].value == "EUR"

    def test_select_kw_label(self):
        fills = extract_fill_values(SELECT_KW_LABEL)
        assert len(fills) == 1
        assert fills[0].value == "Price: low to high"

    def test_commented_out_skipped(self):
        fills = extract_fill_values(COMMENTED_OUT)
        assert len(fills) == 1
        assert fills[0].value == "Munich"

    def test_fstring_skipped(self):
        """f-string values can't be resolved statically — skip them."""
        fills = extract_fill_values(FSTRING_VALUE)
        # page.fill('#location', f'City: {city}') — the f-string prefix
        # means the regex won't match because f'...' is not '...' or "..."
        assert len(fills) == 0

    def test_variable_value_skipped(self):
        """Variable references as values can't be resolved — skip them."""
        fills = extract_fill_values(VARIABLE_VALUE)
        # page.fill('#location', city) — no string literal as second arg
        assert len(fills) == 0

    def test_multiline_call(self):
        fills = extract_fill_values(MULTILINE_CALL)
        assert len(fills) == 1
        assert fills[0].value == "Berlin"

    def test_mixed_patterns(self):
        fills = extract_fill_values(MIXED_PATTERNS)
        assert len(fills) == 5
        values = [f.value for f in fills]
        assert "Berlin" in values
        assert "2026-06-01" in values
        assert "2026-06-05" in values
        assert "2" in values
        assert "pool" in values

    def test_no_fills_returns_empty(self):
        fills = extract_fill_values(NO_FILLS)
        assert fills == []

    def test_empty_code(self):
        assert extract_fill_values("") == []

    def test_frame_locator_chain(self):
        fills = extract_fill_values(FRAME_LOCATOR_CHAIN)
        assert len(fills) == 1
        assert fills[0].value == "Rome"

    def test_nth_locator(self):
        fills = extract_fill_values(NTH_LOCATOR)
        assert len(fills) == 1
        assert fills[0].value == "Berlin"

    def test_first_locator(self):
        fills = extract_fill_values(FIRST_LOCATOR)
        assert len(fills) == 1
        assert fills[0].value == "Berlin"

    def test_results_sorted_by_line(self):
        fills = extract_fill_values(DIRECT_MULTIPLE_FILLS)
        lines = [f.line for f in fills]
        assert lines == sorted(lines)

    def test_no_duplicate_from_direct_and_chain(self):
        """page.fill('sel', 'val') should not produce two results —
        one from the direct regex and one from the chain regex."""
        fills = extract_fill_values(DIRECT_FILL)
        assert len(fills) == 1
        # The value should be 'Berlin' (from 2nd arg), not the selector
        assert fills[0].value == "Berlin"

    # ── inputs["key"] resolution tests ───────────────────────────

    def test_inputs_direct(self):
        inputs = {"destination": "Berlin", "check_in": "2026-06-01"}
        fills = extract_fill_values(INPUTS_DIRECT, inputs=inputs)
        assert len(fills) == 2
        assert fills[0].value == "Berlin"
        assert fills[1].value == "2026-06-01"

    def test_inputs_without_dict_returns_nothing(self):
        """Without inputs dict, inputs['key'] can't be resolved."""
        fills = extract_fill_values(INPUTS_DIRECT)
        assert len(fills) == 0

    def test_inputs_locator_chain(self):
        inputs = {"destination": "Barcelona"}
        fills = extract_fill_values(INPUTS_LOCATOR_CHAIN, inputs=inputs)
        assert len(fills) == 1
        assert fills[0].value == "Barcelona"

    def test_inputs_str_wrap(self):
        """str(inputs['adults']) should resolve to the string value."""
        inputs = {"adults": 2}
        fills = extract_fill_values(INPUTS_STR_WRAP, inputs=inputs)
        assert len(fills) == 1
        assert fills[0].value == "2"

    def test_inputs_mixed_with_literals(self):
        """Mix of inputs refs and string literals should all be captured."""
        inputs = {"destination": "Berlin", "adults": 2}
        fills = extract_fill_values(INPUTS_MIXED, inputs=inputs)
        assert len(fills) == 3
        values = [f.value for f in fills]
        assert "Berlin" in values
        assert "2026-06-01" in values
        assert "2" in values

    def test_inputs_missing_key_skipped(self):
        """If the key doesn't exist in inputs dict, skip it."""
        inputs = {"destination": "Berlin"}  # no "check_in"
        fills = extract_fill_values(INPUTS_DIRECT, inputs=inputs)
        assert len(fills) == 1
        assert fills[0].value == "Berlin"


# ═══════════════════════════════════════════════════════════════════════
#  Test data — URL parameter detection
# ═══════════════════════════════════════════════════════════════════════

_BOOKING_BEFORE = "https://www.booking.com/"
_BOOKING_AFTER = (
    "https://www.booking.com/searchresults.html"
    "?ss=Berlin&checkin=2026-06-01&checkout=2026-06-05&group_adults=2"
)

_BOOKING_FILLS = [
    FillValue("fill", "Berlin", 1),
    FillValue("fill", "2026-06-01", 2),
    FillValue("fill", "2026-06-05", 3),
    FillValue("select_option", "2", 4),
]

_AMAZON_BEFORE = "https://www.amazon.com/"
_AMAZON_AFTER = "https://www.amazon.com/s?k=laptop+stand&ref=nb_sb_noss"
_AMAZON_FILLS = [FillValue("fill", "laptop stand", 1)]

_SPA_BEFORE = "https://app.example.com/"
_SPA_AFTER = "https://app.example.com/#/search?q=Berlin&type=hotels"
_SPA_FILLS = [FillValue("fill", "Berlin", 1)]

_ENCODED_BEFORE = "https://example.com/"
_ENCODED_AFTER = "https://example.com/search?location=New%20York&guests=2"
_ENCODED_FILLS = [
    FillValue("fill", "New York", 1),
    FillValue("select_option", "2", 2),
]


# ═══════════════════════════════════════════════════════════════════════
#  Tests — detect_url_params
# ═══════════════════════════════════════════════════════════════════════


class TestDetectUrlParams:
    """Tests for detect_url_params()."""

    def test_booking_full_match(self):
        result = detect_url_params(
            _BOOKING_BEFORE,
            _BOOKING_AFTER,
            _BOOKING_FILLS,
        )
        assert result is not None
        assert len(result.matches) == 4
        matched_params = {m.param_name for m in result.matches}
        assert "ss" in matched_params
        assert "checkin" in matched_params
        assert "checkout" in matched_params
        assert "group_adults" in matched_params

    def test_booking_value_mapping(self):
        result = detect_url_params(
            _BOOKING_BEFORE,
            _BOOKING_AFTER,
            _BOOKING_FILLS,
        )
        assert result is not None
        by_param = {m.param_name: m for m in result.matches}
        assert by_param["ss"].fill.value == "Berlin"
        assert by_param["checkin"].fill.value == "2026-06-01"

    def test_amazon_with_noise_filtering(self):
        """Amazon adds ref= which should be filtered as noise."""
        result = detect_url_params(
            _AMAZON_BEFORE,
            _AMAZON_AFTER,
            _AMAZON_FILLS,
        )
        assert result is not None
        assert len(result.matches) == 1
        assert result.matches[0].param_name == "k"
        assert result.matches[0].fill.value == "laptop stand"

    def test_url_encoded_spaces(self):
        """'New York' should match 'New%20York' after URL decoding."""
        result = detect_url_params(
            _ENCODED_BEFORE,
            _ENCODED_AFTER,
            _ENCODED_FILLS,
        )
        assert result is not None
        by_param = {m.param_name: m for m in result.matches}
        assert "location" in by_param
        assert by_param["location"].fill.value == "New York"

    def test_plus_encoded_spaces(self):
        """'laptop stand' should match 'laptop+stand'."""
        result = detect_url_params(
            _AMAZON_BEFORE,
            _AMAZON_AFTER,
            _AMAZON_FILLS,
        )
        assert result is not None
        assert result.matches[0].fill.value == "laptop stand"

    def test_spa_fragment_routing(self):
        """Query params in the URL fragment should be detected."""
        result = detect_url_params(_SPA_BEFORE, _SPA_AFTER, _SPA_FILLS)
        assert result is not None
        assert len(result.matches) == 1
        assert result.matches[0].param_name == "q"

    def test_case_insensitive_matching(self):
        fills = [FillValue("fill", "berlin", 1)]
        result = detect_url_params(
            "https://example.com/",
            "https://example.com/search?city=Berlin",
            fills,
        )
        assert result is not None
        assert result.matches[0].param_name == "city"

    def test_no_match_returns_none(self):
        fills = [FillValue("fill", "Berlin", 1)]
        result = detect_url_params(
            "https://example.com/",
            "https://example.com/search?city=Paris",
            fills,
        )
        assert result is None

    def test_no_url_change_returns_none(self):
        """Same URL before and after — no new params."""
        url = "https://example.com/search?q=Berlin"
        fills = [FillValue("fill", "Berlin", 1)]
        result = detect_url_params(url, url, fills)
        assert result is None

    def test_no_query_params_returns_none(self):
        result = detect_url_params(
            "https://example.com/",
            "https://example.com/results",
            [FillValue("fill", "Berlin", 1)],
        )
        assert result is None

    def test_empty_fills_returns_none(self):
        result = detect_url_params(
            "https://example.com/",
            "https://example.com/search?q=Berlin",
            [],
        )
        assert result is None

    def test_none_url_after_returns_none(self):
        result = detect_url_params(
            "https://example.com/",
            None,
            [FillValue("fill", "Berlin", 1)],
        )
        assert result is None

    def test_none_url_before_still_works(self):
        """url_before can be None (first navigation)."""
        result = detect_url_params(
            None,
            "https://example.com/search?q=Berlin",
            [FillValue("fill", "Berlin", 1)],
        )
        assert result is not None
        assert result.matches[0].param_name == "q"

    def test_noise_params_filtered(self):
        """utm_*, fbclid, ref etc. should not appear in matches."""
        fills = [FillValue("fill", "Berlin", 1)]
        result = detect_url_params(
            "https://example.com/",
            "https://example.com/search?q=Berlin&utm_source=google&fbclid=abc123",
            fills,
        )
        assert result is not None
        param_names = {m.param_name for m in result.matches}
        assert "q" in param_names
        assert "utm_source" not in param_names
        assert "fbclid" not in param_names

    def test_session_token_filtered(self):
        """Long random-looking values are filtered as noise."""
        fills = [FillValue("fill", "Berlin", 1)]
        result = detect_url_params(
            "https://example.com/",
            "https://example.com/search?q=Berlin&sid=a8f29c3b7de14f8a9b2c1d3e5f7a9b2c",
            fills,
        )
        assert result is not None
        param_names = {m.param_name for m in result.matches}
        assert "q" in param_names
        assert "sid" not in param_names

    def test_only_noise_params_returns_none(self):
        """If all new params are noise, return None."""
        fills = [FillValue("fill", "Berlin", 1)]
        result = detect_url_params(
            "https://example.com/",
            "https://example.com/search?utm_source=google&ref=homepage",
            fills,
        )
        assert result is None

    def test_pre_existing_params_ignored(self):
        """Params that already existed before should not match."""
        fills = [FillValue("fill", "Berlin", 1)]
        result = detect_url_params(
            "https://example.com/search?q=Berlin",
            "https://example.com/search?q=Berlin&sort=price",
            fills,
        )
        # q=Berlin existed before — not new. sort=price is new but
        # doesn't match any fill value.
        assert result is None


# ═══════════════════════════════════════════════════════════════════════
#  Tests — format_hint
# ═══════════════════════════════════════════════════════════════════════


class TestFormatHint:
    """Tests for format_hint()."""

    def test_basic_hint_format(self):
        result = detect_url_params(
            _BOOKING_BEFORE,
            _BOOKING_AFTER,
            _BOOKING_FILLS,
        )
        assert result is not None
        hint = format_hint(result)
        assert "[URL PARAMETER DETECTION HINT]" in hint
        assert "page navigated to:" in hint
        assert _BOOKING_AFTER in hint
        assert '"Berlin"' in hint
        assert "→  ss" in hint
        assert "page.goto()" in hint
        assert "don't guess" in hint

    def test_hint_shows_unmatched_params(self):
        fills = [FillValue("fill", "Berlin", 1)]
        result = detect_url_params(
            "https://example.com/",
            "https://example.com/search?q=Berlin&sort=price&page=1",
            fills,
        )
        assert result is not None
        hint = format_hint(result)
        assert "sort=price" in hint
        assert "etc.)" in hint


# ═══════════════════════════════════════════════════════════════════════
#  Integration test — full pipeline
# ═══════════════════════════════════════════════════════════════════════


class TestFullPipeline:
    """End-to-end tests: code → fill extraction → URL detection → hint."""

    def test_booking_flow(self):
        code = """\
await page.fill('#bigsearch-query-location-input', 'Berlin')
await page.fill('#checkin', '2026-06-01')
await page.fill('#checkout', '2026-06-05')
await page.select_option('#group_adults', '2')
await page.click('button[type="submit"]')
"""
        fills = extract_fill_values(code)
        assert len(fills) == 4

        result = detect_url_params(
            "https://www.booking.com/",
            "https://www.booking.com/searchresults.html"
            "?ss=Berlin&checkin=2026-06-01&checkout=2026-06-05"
            "&group_adults=2&no_rooms=1",
            fills,
        )
        assert result is not None
        assert len(result.matches) == 4
        hint = format_hint(result)
        assert "ss=Berlin" in hint
        assert "no_rooms=1" in hint  # unmatched but shown

    def test_amazon_flow(self):
        code = """\
await page.locator('#twotabsearchtextbox').fill('laptop stand')
await page.locator('#nav-search-submit-button').click()
"""
        fills = extract_fill_values(code)
        assert len(fills) == 1
        assert fills[0].value == "laptop stand"

        result = detect_url_params(
            "https://www.amazon.com/",
            "https://www.amazon.com/s?k=laptop+stand&ref=nb_sb_noss",
            fills,
        )
        assert result is not None
        assert result.matches[0].param_name == "k"

    def test_airbnb_flow(self):
        code = """\
await page.get_by_placeholder('Search destinations').fill('Berlin')
await page.get_by_role('button', name='Search').click()
"""
        fills = extract_fill_values(code)
        assert len(fills) == 1

        result = detect_url_params(
            "https://www.airbnb.com/",
            "https://www.airbnb.com/s/Berlin/homes?query=Berlin&checkin=2026-06-01",
            fills,
        )
        assert result is not None
        assert any(m.param_name == "query" for m in result.matches)

    def test_no_params_no_hint(self):
        """POST-based search with no URL params — should return None."""
        code = """\
await page.fill('#search', 'Berlin')
await page.click('#submit')
"""
        fills = extract_fill_values(code)
        assert len(fills) == 1

        result = detect_url_params(
            "https://example.com/",
            "https://example.com/results",
            fills,
        )
        assert result is None

    def test_booking_real_flow_with_inputs(self):
        """Real-world Booking.com flow: AI fills form using inputs dict
        across multiple code blocks, then clicks submit separately.
        Accumulated fills should match the resulting URL params."""
        inputs = {
            "destination": "Barcelona, Spain",
            "check_in": "2026-10-06",
            "check_out": "2026-10-10",
            "adults": 2,
        }

        # Step 1: fill destination (separate code block)
        code1 = """\
destination_input = page.locator('input[name="ss"]')
await destination_input.fill(inputs['destination'])
await page.wait_for_timeout(500)
"""
        fills1 = extract_fill_values(code1, inputs=inputs)
        assert len(fills1) == 1
        assert fills1[0].value == "Barcelona, Spain"

        # Step 2: fill dates (separate code block)
        code2 = """\
await page.fill('#checkin', inputs['check_in'])
await page.fill('#checkout', inputs['check_out'])
"""
        fills2 = extract_fill_values(code2, inputs=inputs)
        assert len(fills2) == 2

        # Step 3: click submit (no fills here)
        code3 = """\
await page.click('button[type="submit"]')
await page.wait_for_load_state("domcontentloaded")
"""
        fills3 = extract_fill_values(code3, inputs=inputs)
        assert len(fills3) == 0

        # Accumulated fills from all steps
        all_fills = fills1 + fills2 + fills3

        # URL after submit
        url_after = (
            "https://www.booking.com/searchresults.html"
            "?ss=Barcelona%2C+Spain"
            "&checkin=2026-10-06&checkout=2026-10-10"
            "&group_adults=2&no_rooms=1"
            "&label=gen173nr&sid=a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6"
        )
        result = detect_url_params(
            "https://www.booking.com/",
            url_after,
            all_fills,
        )
        assert result is not None
        matched_params = {m.param_name for m in result.matches}
        assert "ss" in matched_params
        assert "checkin" in matched_params
        assert "checkout" in matched_params
        # sid should be filtered as noise
        assert "sid" not in matched_params


# ═══════════════════════════════════════════════════════════════════════
#  Tier 2 — Complex-encoded URL parameters
# ═══════════════════════════════════════════════════════════════════════

_ZILLOW_JSON_STATE = (
    '{"isMapVisible":true,"mapBounds":{"north":40.06,"south":39.48,'
    '"east":-104.33,"west":-105.36},"filterState":{"sort":{"value":'
    '"globalrelevanceex"},"price":{"min":400000,"max":700000}},'
    '"isListVisible":true,"usersSearchTerm":"Denver, CO",'
    '"regionSelection":[{"regionId":11093,"regionType":6}]}'
)


class TestTier2ComplexParams:
    """Tests for Tier 2: complex-encoded URL parameter detection."""

    def test_zillow_json_returns_tier2(self):
        """Zillow's JSON blob should produce a Tier 2 result, not None."""
        fills = [FillValue("fill", "Denver, CO", 1)]
        url_after = f"https://www.zillow.com/denver-co/?searchQueryState={_ZILLOW_JSON_STATE}"
        result = detect_url_params(
            "https://www.zillow.com/",
            url_after,
            fills,
        )
        assert result is not None
        # No clean matches (JSON blob is not a simple param)
        assert len(result.matches) == 0
        # But complex_params should be populated
        assert result.complex_params is not None
        assert "searchQueryState" in result.complex_params

    def test_zillow_json_hint_format(self):
        """Tier 2 hint should show the URL and observational guidance."""
        fills = [FillValue("fill", "Denver, CO", 1)]
        url_after = f"https://www.zillow.com/denver-co/?searchQueryState={_ZILLOW_JSON_STATE}"
        result = detect_url_params(
            "https://www.zillow.com/",
            url_after,
            fills,
        )
        assert result is not None
        hint = format_hint_complex(result)
        assert "[URL PARAMETER DETECTION HINT]" in hint
        assert "page navigated to:" in hint
        assert "zillow.com" in hint
        assert "encoded in this URL" in hint
        assert "observe how the URL changes" in hint

    def test_pure_noise_still_returns_none(self):
        """Pure tracking noise (no clean, no complex) → None."""
        fills = [FillValue("fill", "Berlin", 1)]
        result = detect_url_params(
            "https://example.com/",
            "https://example.com/search?utm_source=google&ref=homepage",
            fills,
        )
        assert result is None

    def test_mixed_clean_and_complex(self):
        """URL with both clean params and a JSON blob (>50 chars).

        Clean params should produce Tier 1 matches; the JSON blob
        should appear in complex_params.
        """
        fills = [FillValue("fill", "Berlin", 1)]
        # Must be >50 chars to be classified as complex.
        json_blob = (
            '{"filterState":{"sort":{"value":"relevance"},'
            '"price":{"min":100000,"max":500000},'
            '"beds":{"min":2},"isListVisible":true}}'
        )
        url = f"https://example.com/search?q=Berlin&state={json_blob}"
        result = detect_url_params("https://example.com/", url, fills)
        assert result is not None
        # Tier 1 match for q=Berlin
        assert len(result.matches) == 1
        assert result.matches[0].param_name == "q"
        # Complex param for state={json}
        assert result.complex_params is not None
        assert "state" in result.complex_params

    def test_tier1_hint_used_when_clean_matches_exist(self):
        """When both clean and complex params exist, Tier 1 hint is used."""
        fills = [FillValue("fill", "Berlin", 1)]
        json_blob = (
            '{"filterState":{"sort":{"value":"relevance"},'
            '"price":{"min":100000,"max":500000},'
            '"beds":{"min":2},"isListVisible":true}}'
        )
        url = f"https://example.com/search?q=Berlin&state={json_blob}"
        result = detect_url_params("https://example.com/", url, fills)
        assert result is not None
        assert len(result.matches) > 0
        # Tier 1 hint should work (has matches)
        hint = format_hint(result)
        assert "Berlin" in hint
        assert "\u2192  q" in hint
