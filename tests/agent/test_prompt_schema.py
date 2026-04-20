"""Tests for schema integration in the agent prompt.

These tests verify that the system prompt correctly incorporates the
schema section and that all 8 text changes from the spec are applied.
They also check prompt quality from the agent's perspective.
"""

import pytest

from scout.agent.prompt import (
    build_initial_user_message,
    build_show_page_analysis_prompt_a,
    build_show_page_analysis_prompt_b,
    build_system_prompt,
)
from scout.schema.compiler import compile_schema
from scout.schema.types import Field, List


# ── Helpers ──────────────────────────────────────────────────────

def _build_prompt_with_schema(schema):
    """Compile a schema and build the system prompt with it."""
    cs = compile_schema(schema)
    return build_system_prompt(schema_prompt=cs.prompt)


# ── Change 1: build_system_prompt requires schema_prompt ─────────

class TestBuildSystemPromptSignature:

    def test_requires_schema_prompt_keyword(self):
        """schema_prompt is keyword-only — cannot pass positionally."""
        with pytest.raises(TypeError):
            build_system_prompt("some prompt")

    def test_returns_string(self):
        cs = compile_schema([{"x": str}])
        result = build_system_prompt(schema_prompt=cs.prompt)
        assert isinstance(result, str)
        assert len(result) > 1000  # non-trivial prompt


# ── Change 2: Function comment references schema ─────────────────

class TestFunctionComment:

    def test_return_value_must_match_schema(self):
        prompt = _build_prompt_with_schema([{"title": str}])
        assert "Return value must match the output schema below." in prompt

    def test_no_any_json_serializable(self):
        prompt = _build_prompt_with_schema([{"x": str}])
        assert "any JSON-serializable" not in prompt.lower()
        assert "whatever shape is natural" not in prompt


# ── Change 3: Output subsection references schema validation ─────

class TestOutputSubsection:

    def test_schema_validation_mentioned(self):
        prompt = _build_prompt_with_schema([{"x": str}])
        assert "validates the return value against the schema" in prompt

    def test_rejection_mentioned(self):
        prompt = _build_prompt_with_schema([{"x": str}])
        assert "rejected with the specific validation errors" in prompt


# ── Change 4: Schema section injected between sections ───────────

class TestSchemaInjection:

    def test_output_schema_heading_present(self):
        prompt = _build_prompt_with_schema([{"title": str, "price": float}])
        assert "## Output Schema" in prompt

    def test_structure_and_requirements_present(self):
        prompt = _build_prompt_with_schema([{"title": str}])
        assert "### Structure" in prompt
        assert "### Requirements" in prompt

    def test_schema_between_robust_functions_and_rules(self):
        """Schema section must appear after 'Writing Robust Functions'
        and before 'Rules'."""
        prompt = _build_prompt_with_schema([{"x": str}])
        robust_pos = prompt.index("## Writing Robust Functions")
        schema_pos = prompt.index("## Output Schema")
        rules_pos = prompt.index("## Rules")
        assert robust_pos < schema_pos < rules_pos

    def test_schema_fields_appear_in_prompt(self):
        prompt = _build_prompt_with_schema([{
            "title": str,
            "price": Field(float, min=0),
            "currency": Field(str, enum=["USD", "EUR"]),
        }])
        for field in ["title", "price", "currency"]:
            assert field in prompt

    def test_constraints_appear_in_prompt(self):
        prompt = _build_prompt_with_schema(
            List({"x": Field(int, min=1, max=5)}, min=10)
        )
        assert "minimum 10" in prompt or "at least **10" in prompt
        assert "between 1 and 5" in prompt.lower() or "Between 1 and 5" in prompt

    def test_optional_paragraph_when_optional_fields(self):
        prompt = _build_prompt_with_schema({
            "name": str,
            "bio": Field(str, optional=True),
        })
        assert "For optional fields" in prompt

    def test_no_optional_paragraph_when_all_required(self):
        prompt = _build_prompt_with_schema({"name": str, "age": int})
        # The Output Schema section should NOT have the optional paragraph
        schema_section = prompt.split("## Output Schema")[1].split("## Rules")[0]
        assert "For optional fields" not in schema_section


# ── Change 5: Rule 9 references schema ───────────────────────────

class TestRule9:

    def test_rule_9_mentions_output_schema(self):
        prompt = _build_prompt_with_schema([{"x": str}])
        # Find rule 9 text
        rule9_start = prompt.index("9. **Print progress, return the data.**")
        rule10_start = prompt.index("10. **Add checkpoints")
        rule9_text = prompt[rule9_start:rule10_start]
        assert "output schema" in rule9_text


# ── Change 6: Post-workflow paragraph ─────────────────────────────

class TestPostWorkflow:

    def test_validated_against_schema(self):
        prompt = _build_prompt_with_schema([{"x": str}])
        assert "validates the return value against the schema" in prompt

    def test_no_output_will_be_reviewed(self):
        prompt = _build_prompt_with_schema([{"x": str}])
        assert "output will be reviewed" not in prompt


# ── Change 7: Reasoning point 3 ──────────────────────────────────

class TestReasoningSection:

    def test_designing_mentions_schema(self):
        prompt = _build_prompt_with_schema([{"x": str}])
        assert "how it shapes the data to match the output schema" in prompt


# ── Change 8: Don't ship broken fields ────────────────────────────

class TestDontShipBrokenFields:

    def test_checklist_sentence(self):
        prompt = _build_prompt_with_schema([{"x": str}])
        assert "use it as your checklist" in prompt


# ── Prompt coherence: full end-to-end checks ─────────────────────

class TestPromptCoherence:
    """Verifies the prompt reads naturally for an AI agent."""

    def test_schema_section_not_empty(self):
        """The schema_section placeholder is always filled."""
        prompt = _build_prompt_with_schema([{"x": str}])
        assert "{schema_section}" not in prompt

    def test_patchright_guide_placeholder_filled(self):
        prompt = _build_prompt_with_schema([{"x": str}])
        assert "{patchright_guide}" not in prompt

    def test_no_unfilled_template_placeholders(self):
        """The two template placeholders are filled."""
        prompt = _build_prompt_with_schema([{"x": str}])
        assert "{schema_section}" not in prompt
        assert "{patchright_guide}" not in prompt

    def test_complex_schema_produces_readable_prompt(self):
        """A real-world schema should produce a prompt an agent can follow."""
        schema = List({
            "title": Field(str, min_length=1),
            "price": Field(float, min=0),
            "currency": Field(str, enum=["USD", "EUR", "GBP"]),
            "rating": Field(int, min=1, max=5, optional=True),
            "in_stock": bool,
        }, min=20)
        prompt = _build_prompt_with_schema(schema)

        # Agent sees the schema heading
        assert "## Output Schema" in prompt

        # Agent sees all fields
        for field in ["title", "price", "currency", "rating", "in_stock"]:
            assert field in prompt

        # Agent sees the minimum count
        assert "20" in prompt

        # Agent sees enum values
        assert '"USD"' in prompt

        # Agent sees optional marking
        assert "optional" in prompt.lower() or "Optional" in prompt

    def test_schema_references_are_consistent(self):
        """All mentions of 'output schema' point to the same section."""
        prompt = _build_prompt_with_schema([{"x": str}])
        # Count references to "output schema" (case-insensitive)
        import re
        refs = re.findall(r"output schema", prompt, re.IGNORECASE)
        # At least: Change 2, Change 3, Change 4, Change 5, Change 6,
        # Change 7, Change 8 = multiple references
        assert len(refs) >= 5, (
            f"Expected at least 5 'output schema' references, got {len(refs)}"
        )

    def test_heading_hierarchy_is_correct(self):
        """All major sections use ## headings, subsections use ###."""
        prompt = _build_prompt_with_schema([{"x": str}])
        # These should all be ## level headings
        for heading in [
            "## Your Tool",
            "## Reasoning",
            "## Workflow",
            "## Writing Robust Functions",
            "## Output Schema",
            "## Rules",
            "## Patchright API Reference",
        ]:
            assert heading in prompt, f"Missing heading: {heading}"


# ── Edge case schemas ─────────────────────────────────────────────

class TestEdgeCaseSchemas:
    """Unusual but valid schemas should produce usable prompts."""

    def test_bare_scalar_has_requirements(self):
        """Even a bare scalar schema should have a non-empty Requirements."""
        prompt = _build_prompt_with_schema(str)
        req_section = prompt.split("### Requirements")[1].split("---")[0]
        assert "string" in req_section.lower()
        assert req_section.strip()  # not empty

    def test_bare_dict_has_requirements(self):
        prompt = _build_prompt_with_schema(dict)
        req_section = prompt.split("### Requirements")[1].split("---")[0]
        assert "freestyle" in req_section.lower()

    def test_deeply_nested_schema_all_fields_present(self):
        prompt = _build_prompt_with_schema([{
            "categories": [{
                "products": [{
                    "title": str,
                    "variants": [{"color": str, "size": str}],
                }],
            }],
        }])
        for field in ["categories", "products", "title", "variants", "color", "size"]:
            assert field in prompt, f"Missing field {field!r}"

    def test_many_fields_object(self):
        """An object with many fields should list them all."""
        schema = {f"field_{i}": str for i in range(15)}
        prompt = _build_prompt_with_schema(schema)
        for i in range(15):
            assert f"field_{i}" in prompt


# ── build_initial_user_message is unchanged ───────────────────────

class TestInitialUserMessage:

    def test_contains_task_and_url(self):
        msg = build_initial_user_message("Extract prices", "https://example.com")
        assert "Extract prices" in msg
        assert "https://example.com" in msg

    def test_prompts_show_page(self):
        msg = build_initial_user_message("task", "https://x.com")
        assert "show_page" in msg


# ── Show-page analysis prompts (Task 5) ─────────────────────────

class TestShowPageAnalysisPromptA:
    """Variant A — full analysis prompt."""

    def test_returns_string(self):
        result = build_show_page_analysis_prompt_a()
        assert isinstance(result, str)

    def test_starts_with_page_analysis_header(self):
        result = build_show_page_analysis_prompt_a()
        assert result.startswith("── Page Analysis ──")

    def test_ends_with_closing_rule(self):
        result = build_show_page_analysis_prompt_a()
        assert result.rstrip().endswith("──")

    def test_mentions_section_ids(self):
        result = build_show_page_analysis_prompt_a()
        assert "section ID" in result or "section id" in result.lower()

    def test_mentions_interactive_elements(self):
        result = build_show_page_analysis_prompt_a()
        assert "interactive element" in result.lower() or "full tags" in result.lower()

    def test_mentions_context_clearing(self):
        result = build_show_page_analysis_prompt_a()
        assert "cleared from context" in result


class TestShowPageAnalysisPromptB:
    """Variant B — page update prompt."""

    def test_returns_string(self):
        result = build_show_page_analysis_prompt_b()
        assert isinstance(result, str)

    def test_starts_with_page_update_header(self):
        result = build_show_page_analysis_prompt_b()
        assert result.startswith("── Page Update ──")

    def test_ends_with_closing_rule(self):
        result = build_show_page_analysis_prompt_b()
        assert result.rstrip().endswith("──")

    def test_mentions_relevant_content(self):
        result = build_show_page_analysis_prompt_b()
        assert "relevant" in result.lower()

    def test_mentions_changes(self):
        result = build_show_page_analysis_prompt_b()
        assert "changed" in result.lower()

    def test_mentions_context_clearing(self):
        result = build_show_page_analysis_prompt_b()
        assert "cleared from context" in result
