"""Tests for dynamic inputs integration in the prompt system."""

from __future__ import annotations

from scout.agent.prompt import build_initial_user_message, build_system_prompt
from scout.inputs import build_inputs_fragments, build_inputs_hint, normalize_inputs


class TestSystemPromptNoInputs:
    """When no inputs are provided, the prompt is unchanged."""

    def test_no_inputs_text(self):
        prompt = build_system_prompt(schema_prompt="## Output Schema\ntest")
        assert "inputs" not in prompt.lower().split("patchright")[0]
        # "inputs" should not appear before the Patchright guide
        # (it may appear in the guide text itself, which is fine)

    def test_no_rule_11(self):
        prompt = build_system_prompt(schema_prompt="## Output Schema\ntest")
        assert "11." not in prompt.split("## Rules")[1].split("## Patchright")[0]

    def test_phase3_has_3_param(self):
        prompt = build_system_prompt(schema_prompt="## Output Schema\ntest")
        assert "async def scrape(page, start_url, checkpoint)" in prompt

    def test_phase3_no_inputs_param(self):
        prompt = build_system_prompt(schema_prompt="## Output Schema\ntest")
        # The 3-param version should be shown, not the 4-param
        assert "async def scrape(page, start_url, inputs, checkpoint)" not in prompt


class TestSystemPromptWithInputs:
    """When inputs are provided, the prompt includes all 4 fragments."""

    def _make_fragments(self, raw):
        _, defs = normalize_inputs(raw)
        return build_inputs_fragments(defs)

    def test_tool_desc_mentions_inputs(self):
        fragments = self._make_fragments({"q": "python"})
        prompt = build_system_prompt(
            schema_prompt="## Output Schema\ntest",
            inputs_fragments=fragments,
        )
        assert "an `inputs` dict" in prompt

    def test_dynamic_inputs_section_present(self):
        fragments = self._make_fragments({"q": "python"})
        prompt = build_system_prompt(
            schema_prompt="## Output Schema\ntest",
            inputs_fragments=fragments,
        )
        assert "## Dynamic Inputs" in prompt

    def test_phase3_has_4_param(self):
        fragments = self._make_fragments({"q": "python"})
        prompt = build_system_prompt(
            schema_prompt="## Output Schema\ntest",
            inputs_fragments=fragments,
        )
        assert "async def scrape(page, start_url, inputs, checkpoint)" in prompt

    def test_rule_11_present(self):
        fragments = self._make_fragments({"q": "python"})
        prompt = build_system_prompt(
            schema_prompt="## Output Schema\ntest",
            inputs_fragments=fragments,
        )
        rules_section = prompt.split("## Rules")[1].split("## Patchright")[0]
        assert "11." in rules_section
        assert "never hardcode" in rules_section.lower()

    def test_access_patterns_in_section(self):
        fragments = self._make_fragments({"query": "python", "location": "Berlin"})
        prompt = build_system_prompt(
            schema_prompt="## Output Schema\ntest",
            inputs_fragments=fragments,
        )
        assert 'inputs["query"]' in prompt
        assert 'inputs["location"]' in prompt

    def test_example_values_marked(self):
        fragments = self._make_fragments({"q": "python developer"})
        prompt = build_system_prompt(
            schema_prompt="## Output Schema\ntest",
            inputs_fragments=fragments,
        )
        assert 'e.g. "python developer"' in prompt


class TestInitialMessageNoInputs:
    def test_no_inputs_hint(self):
        msg = build_initial_user_message("task", "https://example.com")
        assert "inputs" not in msg.lower()

    def test_show_page_line_present(self):
        msg = build_initial_user_message("task", "https://example.com")
        assert "show_page(page)" in msg


class TestInitialMessageWithInputs:
    def test_inputs_hint_appended(self):
        _, defs = normalize_inputs({"q": "python"})
        hint = build_inputs_hint(defs)
        msg = build_initial_user_message(
            "task",
            "https://example.com",
            inputs_hint=hint,
        )
        assert "`inputs` dict" in msg
        assert 'inputs["q"]' in msg

    def test_show_page_still_present(self):
        _, defs = normalize_inputs({"q": "python"})
        hint = build_inputs_hint(defs)
        msg = build_initial_user_message(
            "task",
            "https://example.com",
            inputs_hint=hint,
        )
        assert "show_page(page)" in msg
