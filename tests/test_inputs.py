"""Tests for the dynamic inputs module."""

from __future__ import annotations

import pytest

from scout.inputs import (
    Input,
    normalize_inputs,
    build_inputs_section,
    build_inputs_hint,
    build_inputs_rule,
    build_inputs_fragments,
    build_inputs_phase3_example,
    format_inputs_metadata,
    parse_inputs_metadata,
    validate_inputs_against_metadata,
    _PHASE3_EXAMPLE_NO_INPUTS,
)
from scout.errors import ConfigError


# ═══════════════════════════════════════════════════════════════
#  Input class
# ═══════════════════════════════════════════════════════════════


class TestInputClass:

    def test_string_value_infers_str(self):
        i = Input("hello")
        assert i.value == "hello"
        assert i.type_ is str
        assert i.description is None

    def test_int_value_infers_int(self):
        i = Input(42)
        assert i.type_ is int

    def test_float_value_infers_float(self):
        i = Input(3.14)
        assert i.type_ is float

    def test_bool_value_infers_bool(self):
        i = Input(True)
        assert i.type_ is bool

    def test_explicit_type_overrides_inference(self):
        i = Input("50", type_=str)
        assert i.type_ is str

    def test_description_stored(self):
        i = Input("Berlin", description="City to filter by")
        assert i.description == "City to filter by"

    def test_unsupported_type_raises(self):
        with pytest.raises(ConfigError, match="Supported types"):
            Input([1, 2, 3])

    def test_unsupported_explicit_type_raises(self):
        with pytest.raises(ConfigError, match="one of str, int, float, bool"):
            Input("x", type_=list)

    def test_repr_basic(self):
        r = repr(Input("hello"))
        assert "hello" in r
        assert "Input(" in r

    def test_repr_with_description(self):
        r = repr(Input("hello", description="greeting"))
        assert "description='greeting'" in r


# ═══════════════════════════════════════════════════════════════
#  normalize_inputs
# ═══════════════════════════════════════════════════════════════


class TestNormalizeInputs:

    def test_none_returns_none(self):
        assert normalize_inputs(None) == (None, None)

    def test_empty_dict_returns_none(self):
        assert normalize_inputs({}) == (None, None)

    def test_bare_values(self):
        values, defs = normalize_inputs({"q": "hello", "n": 5})
        assert values == {"q": "hello", "n": 5}
        assert len(defs) == 2
        assert defs[0]["name"] == "q"
        assert defs[0]["type"] is str
        assert defs[0]["example"] == "hello"
        assert defs[1]["name"] == "n"
        assert defs[1]["type"] is int

    def test_input_instances(self):
        values, defs = normalize_inputs({
            "city": Input("Berlin", description="City"),
        })
        assert values == {"city": "Berlin"}
        assert defs[0]["description"] == "City"
        assert defs[0]["type"] is str

    def test_mixed_bare_and_input(self):
        values, defs = normalize_inputs({
            "q": "python",
            "city": Input("Berlin", description="City"),
        })
        assert values == {"q": "python", "city": "Berlin"}
        assert defs[0]["description"] is None
        assert defs[1]["description"] == "City"

    def test_invalid_key_raises(self):
        with pytest.raises(ConfigError, match="valid Python identifier"):
            normalize_inputs({"123bad": "value"})

    def test_unsupported_type_raises(self):
        with pytest.raises(ConfigError, match="unsupported type"):
            normalize_inputs({"data": [1, 2, 3]})

    def test_not_dict_raises(self):
        with pytest.raises(ConfigError, match="must be a dict"):
            normalize_inputs("not a dict")  # type: ignore


# ═══════════════════════════════════════════════════════════════
#  Prompt builders
# ═══════════════════════════════════════════════════════════════


class TestBuildInputsSection:

    def _defs(self, raw):
        _, defs = normalize_inputs(raw)
        return defs

    def test_single_field(self):
        section = build_inputs_section(self._defs({"q": "hello"}))
        assert "## Dynamic Inputs" in section
        assert '"q" (str)' in section
        assert 'e.g. "hello"' in section
        assert 'inputs["q"]' in section

    def test_multiple_fields(self):
        section = build_inputs_section(self._defs({
            "query": "python",
            "location": "Berlin",
        }))
        assert 'inputs["query"]' in section
        assert 'inputs["location"]' in section
        assert "and" in section

    def test_description_included(self):
        section = build_inputs_section(self._defs({
            "city": Input("Berlin", description="City to filter"),
        }))
        assert "City to filter" in section

    def test_int_example(self):
        section = build_inputs_section(self._defs({"max": 50}))
        assert "e.g. 50" in section
        assert '"max" (int)' in section

    def test_never_hardcode_message(self):
        section = build_inputs_section(self._defs({"q": "test"}))
        assert "never hardcode" in section.lower()


class TestBuildInputsHint:

    def test_contains_access_pattern(self):
        _, defs = normalize_inputs({"q": "python", "loc": "Berlin"})
        hint = build_inputs_hint(defs)
        assert 'inputs["q"]' in hint
        assert 'inputs["loc"]' in hint
        assert "`inputs` dict" in hint


class TestBuildInputsRule:

    def test_rule_number(self):
        _, defs = normalize_inputs({"q": "python"})
        rule = build_inputs_rule(defs)
        assert "11." in rule

    def test_anti_pattern_includes_example(self):
        _, defs = normalize_inputs({"q": "python developer"})
        rule = build_inputs_rule(defs)
        assert '"python developer"' in rule
        assert 'inputs["q"]' in rule


class TestBuildInputsPhase3:

    def test_has_inputs_param(self):
        _, defs = normalize_inputs({"q": "python"})
        example = build_inputs_phase3_example(defs)
        assert "async def scrape(page, start_url, inputs, checkpoint)" in example
        assert 'inputs["q"]' in example
        assert "Never hardcode" in example

    def test_no_inputs_example(self):
        assert "async def scrape(page, start_url, checkpoint)" in _PHASE3_EXAMPLE_NO_INPUTS
        assert "inputs" not in _PHASE3_EXAMPLE_NO_INPUTS


class TestBuildInputsFragments:

    def test_all_keys_present(self):
        _, defs = normalize_inputs({"q": "python"})
        fragments = build_inputs_fragments(defs)
        assert "inputs_tool_desc" in fragments
        assert "inputs_section" in fragments
        assert "phase3_code_example" in fragments
        assert "inputs_rule" in fragments


# ═══════════════════════════════════════════════════════════════
#  Metadata helpers
# ═══════════════════════════════════════════════════════════════


class TestFormatInputsMetadata:

    def test_single_field(self):
        _, defs = normalize_inputs({"q": "hello"})
        assert format_inputs_metadata(defs) == "q (str)"

    def test_two_fields(self):
        _, defs = normalize_inputs({"q": "hello", "n": 5})
        assert format_inputs_metadata(defs) == "q (str) and n (int)"

    def test_three_fields(self):
        _, defs = normalize_inputs({"a": "x", "b": "y", "c": 1})
        result = format_inputs_metadata(defs)
        assert result == "a (str), b (str) and c (int)"


class TestParseInputsMetadata:

    def test_round_trip_single(self):
        _, defs = normalize_inputs({"q": "hello"})
        meta = format_inputs_metadata(defs)
        parsed = parse_inputs_metadata(meta)
        assert parsed == {"q": str}

    def test_round_trip_multiple(self):
        _, defs = normalize_inputs({"q": "hello", "n": 5, "f": 3.14})
        meta = format_inputs_metadata(defs)
        parsed = parse_inputs_metadata(meta)
        assert parsed == {"q": str, "n": int, "f": float}

    def test_parse_with_and(self):
        parsed = parse_inputs_metadata("query (str), location (str) and max_results (int)")
        assert parsed == {"query": str, "location": str, "max_results": int}


# ═══════════════════════════════════════════════════════════════
#  Validation
# ═══════════════════════════════════════════════════════════════


class TestValidateInputsAgainstMetadata:

    def test_both_none_ok(self):
        validate_inputs_against_metadata(None, None)

    def test_both_empty_ok(self):
        validate_inputs_against_metadata({}, "")

    def test_valid_inputs(self):
        validate_inputs_against_metadata(
            {"q": "hello", "n": 5},
            "q (str) and n (int)",
        )

    def test_missing_key(self):
        with pytest.raises(ConfigError, match='"q" is missing'):
            validate_inputs_against_metadata(
                {"n": 5},
                "q (str) and n (int)",
            )

    def test_unexpected_key(self):
        with pytest.raises(ConfigError, match='unexpected key "extra"'):
            validate_inputs_against_metadata(
                {"q": "hello", "extra": "oops"},
                "q (str)",
            )

    def test_type_mismatch(self):
        with pytest.raises(ConfigError, match="type mismatch"):
            validate_inputs_against_metadata(
                {"n": "not_an_int"},
                "n (int)",
            )

    def test_int_allowed_for_float(self):
        validate_inputs_against_metadata(
            {"f": 5},
            "f (float)",
        )

    def test_meta_expects_inputs_but_none_given(self):
        with pytest.raises(ConfigError, match="expects inputs"):
            validate_inputs_against_metadata(None, "q (str)")

    def test_inputs_given_but_no_meta(self):
        with pytest.raises(ConfigError, match="without dynamic inputs"):
            validate_inputs_against_metadata({"q": "hello"}, None)

    def test_input_instance_unwrapped_for_type_check(self):
        validate_inputs_against_metadata(
            {"q": Input("hello")},
            "q (str)",
        )
