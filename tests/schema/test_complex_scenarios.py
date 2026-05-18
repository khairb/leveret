"""Super-complex end-to-end tests for the schema system.

These tests simulate realistic scraping scenarios with deeply nested
schemas, many fields, edge cases in data, and multi-layered validation
failures. Each test verifies the full pipeline: schema definition →
prompt rendering → data validation → error formatting.

The goal is to catch subtle bugs that only appear with complex,
production-realistic inputs.
"""

import pytest

from scout.schema.compiler import compile_schema
from scout.schema.types import Field, List

# ═══════════════════════════════════════════════════════════════════
#  Scenario 1: E-commerce product catalog
#  3 levels of nesting, 12 fields, all constraint types
# ═══════════════════════════════════════════════════════════════════

ECOMMERCE_SCHEMA = List(
    {
        "title": Field(str, min_length=1),
        "price": Field(float, min=0),
        "currency": Field(str, enum=["USD", "EUR", "GBP"]),
        "rating": Field(int, min=1, max=5, optional=True),
        "description": Field(str, min_length=20),
        "in_stock": bool,
        "specs": dict,
        "tags": [str],
        "variants": [
            {
                "sku": str,
                "color": str,
                "size": Field(str, enum=["XS", "S", "M", "L", "XL", "XXL"]),
                "price_override": Field(float, min=0, optional=True),
            }
        ],
    },
    min_items=20,
)


class TestEcommerceScenario:
    @pytest.fixture
    def cs(self):
        return compile_schema(ECOMMERCE_SCHEMA)

    def _make_good_product(self, i):
        return {
            "title": f"Premium Widget {i}",
            "price": round(9.99 + i * 5.0, 2),
            "currency": ["USD", "EUR", "GBP"][i % 3],
            "rating": (i % 5) + 1 if i % 3 != 0 else None,
            "description": f"This is a detailed product description for widget number {i}.",
            "in_stock": i % 4 != 0,
            "specs": {"weight": f"{i * 100}g", "material": "steel"},
            "tags": ["widget", f"category-{i % 5}"],
            "variants": [
                {
                    "sku": f"W{i:04d}-{c}",
                    "color": c,
                    "size": ["S", "M", "L", "XL"][j % 4],
                    "price_override": round(i * 5.0 + j * 2.0, 2) if j % 2 == 0 else None,
                }
                for j, c in enumerate(["Red", "Blue", "Green"])
            ],
        }

    def test_fully_correct_data_passes(self, cs):
        data = [self._make_good_product(i) for i in range(25)]
        valid, msg = cs.validate(data)
        assert valid is True
        assert msg == ""

    def test_prompt_contains_all_fields(self, cs):
        prompt = cs.prompt
        for field in [
            "title",
            "price",
            "currency",
            "rating",
            "description",
            "in_stock",
            "specs",
            "tags",
            "variants",
            "sku",
            "color",
            "size",
            "price_override",
        ]:
            assert field in prompt, f"Field {field!r} missing from prompt"

    def test_prompt_contains_all_constraints(self, cs):
        prompt = cs.prompt
        assert "min length: 1" in prompt
        assert ">= 0" in prompt
        assert '"USD"' in prompt and '"EUR"' in prompt and '"GBP"' in prompt
        assert "1 to 5" in prompt
        assert "min length: 20" in prompt
        assert "minimum 20 items" in prompt
        assert '"XS"' in prompt and '"XXL"' in prompt

    def test_prompt_has_optional_paragraph(self, cs):
        assert "For optional fields" in cs.prompt

    def test_mixed_errors_across_items(self, cs):
        """Multiple error types scattered across many items."""
        data = []
        for i in range(25):
            item = self._make_good_product(i)
            if i % 5 == 0:
                item["price"] = f"${item['price']}"  # string
            if i % 7 == 0:
                item["currency"] = "Dollar"  # invalid enum
            if i % 4 == 0:
                item["description"] = "Too short"  # min_length violation
            if i == 3:
                item["rating"] = 0  # below min (optional present)
            if i == 10:
                item["variants"][0]["size"] = "XXXL"  # invalid enum
            data.append(item)

        valid, msg = cs.validate(data)
        assert valid is False
        # Should have multiple grouped errors
        assert "price" in msg
        assert "currency" in msg
        assert "description" in msg
        # Error count header
        assert "error type" in msg or "error" in msg

    def test_all_items_missing_required_field(self, cs):
        """Every item is missing 'description'."""
        data = []
        for i in range(25):
            item = self._make_good_product(i)
            del item["description"]
            data.append(item)

        valid, msg = cs.validate(data)
        assert valid is False
        assert "description" in msg
        assert "missing" in msg
        assert "25 of 25" in msg

    def test_nested_variant_errors(self, cs):
        """Errors deep in the variants array."""
        data = [self._make_good_product(i) for i in range(20)]
        # Corrupt all variants in all items
        for item in data:
            for v in item["variants"]:
                v["size"] = "XXXL"  # invalid enum

        valid, msg = cs.validate(data)
        assert valid is False
        assert "size" in msg
        assert "XXXL" in msg
        assert "not one of" in msg

    def test_too_few_items(self, cs):
        data = [self._make_good_product(i) for i in range(5)]
        valid, msg = cs.validate(data)
        assert valid is False
        assert "5" in msg
        assert "20" in msg

    def test_empty_tags_list_is_valid(self, cs):
        """tags: [str] with no min constraint — empty list is valid."""
        data = [self._make_good_product(i) for i in range(20)]
        for item in data:
            item["tags"] = []
        valid, msg = cs.validate(data)
        assert valid is True


# ═══════════════════════════════════════════════════════════════════
#  Scenario 2: Job listings with salary ranges
#  Nested optional objects, pattern constraints
# ═══════════════════════════════════════════════════════════════════

JOB_SCHEMA = List(
    {
        "title": Field(str, min_length=3),
        "company": str,
        "location": str,
        "salary": {
            "min": Field(int, min=0, optional=True),
            "max": Field(int, min=0, optional=True),
            "currency": Field(str, enum=["USD", "EUR", "GBP"], optional=True),
        },
        "posted": Field(str, pattern=r"\d{4}-\d{2}-\d{2}"),
        "remote": bool,
        "skills": [str],
    },
    min_items=10,
)


class TestJobListingsScenario:
    @pytest.fixture
    def cs(self):
        return compile_schema(JOB_SCHEMA)

    def test_correct_data(self, cs):
        data = [
            {
                "title": f"Senior Engineer {i}",
                "company": f"Company {i}",
                "location": "Remote",
                "salary": {
                    "min": 80000 + i * 5000,
                    "max": 120000 + i * 5000,
                    "currency": "USD",
                },
                "posted": f"2024-{(i % 12) + 1:02d}-15",
                "remote": True,
                "skills": ["Python", "TypeScript"],
            }
            for i in range(15)
        ]
        valid, msg = cs.validate(data)
        assert valid is True

    def test_salary_as_number_not_object(self, cs):
        """Agent flattened the salary object to a number."""
        data = [
            {
                "title": "Engineer",
                "company": "Acme",
                "location": "NYC",
                "salary": 100000,  # should be an object
                "posted": "2024-03-15",
                "remote": True,
                "skills": ["Python"],
            }
            for _ in range(10)
        ]

        valid, msg = cs.validate(data)
        assert valid is False
        assert "salary" in msg
        assert "expected an object" in msg

    def test_salary_optional_fields_all_null(self, cs):
        """All salary sub-fields null (they're optional)."""
        data = [
            {
                "title": "Engineer",
                "company": "Acme",
                "location": "NYC",
                "salary": {"min": None, "max": None, "currency": None},
                "posted": "2024-03-15",
                "remote": True,
                "skills": [],
            }
            for _ in range(10)
        ]

        valid, msg = cs.validate(data)
        assert valid is True

    def test_date_pattern_violations(self, cs):
        """Agent extracted dates in wrong format."""
        data = [
            {
                "title": "Engineer",
                "company": "Acme",
                "location": "NYC",
                "salary": {"min": 80000, "max": 120000, "currency": "USD"},
                "posted": "March 15, 2024",  # wrong format
                "remote": True,
                "skills": ["Python"],
            }
            for _ in range(10)
        ]

        valid, msg = cs.validate(data)
        assert valid is False
        assert "does not match pattern" in msg
        assert "March 15, 2024" in msg

    def test_negative_salary(self, cs):
        data = [
            {
                "title": "Engineer",
                "company": "Acme",
                "location": "NYC",
                "salary": {"min": -50000, "max": 120000, "currency": "USD"},
                "posted": "2024-03-15",
                "remote": True,
                "skills": [],
            }
            for _ in range(10)
        ]

        valid, msg = cs.validate(data)
        assert valid is False
        assert ">= 0" in msg
        assert "salary" in msg or "min" in msg


# ═══════════════════════════════════════════════════════════════════
#  Scenario 3: Search engine results page
#  Object with nested constrained list
# ═══════════════════════════════════════════════════════════════════

SERP_SCHEMA = {
    "total_results": int,
    "query": str,
    "results": List(
        {
            "title": str,
            "url": Field(str, min_length=10),
            "snippet": Field(str, max_length=500),
            "published": Field(str, pattern=r"\d{4}-\d{2}-\d{2}", optional=True),
            "source": str,
        },
        min_items=10,
        max_items=50,
    ),
}


class TestSearchResultsScenario:
    @pytest.fixture
    def cs(self):
        return compile_schema(SERP_SCHEMA)

    def test_correct_data(self, cs):
        data = {
            "total_results": 1500,
            "query": "python web scraping",
            "results": [
                {
                    "title": f"Result {i}",
                    "url": f"https://example.com/page/{i}",
                    "snippet": f"This is a snippet for result {i}.",
                    "published": f"2024-0{(i % 9) + 1}-15" if i % 2 == 0 else None,
                    "source": "example.com",
                }
                for i in range(25)
            ],
        }
        valid, msg = cs.validate(data)
        assert valid is True

    def test_prompt_renders_as_object(self, cs):
        """Top-level is an object, not a list."""
        assert "Return an **object**" in cs.prompt
        assert "10 to 50 items" in cs.prompt

    def test_returned_as_list_instead_of_object(self, cs):
        """Agent returned list of results instead of the wrapper object."""
        data = [
            {"title": "R", "url": "https://x.com/1", "snippet": "s", "source": "x.com"}
            for _ in range(20)
        ]
        valid, msg = cs.validate(data)
        assert valid is False
        assert "Expected an object" in msg

    def test_too_many_results(self, cs):
        """Agent extracted more than the max."""
        data = {
            "total_results": 100,
            "query": "test",
            "results": [
                {"title": f"R{i}", "url": f"https://x.com/{i}", "snippet": "ok", "source": "x.com"}
                for i in range(60)  # max is 50
            ],
        }
        valid, msg = cs.validate(data)
        assert valid is False
        assert "60" in msg
        assert "50" in msg

    def test_snippet_too_long(self, cs):
        data = {
            "total_results": 10,
            "query": "test",
            "results": [
                {
                    "title": "R",
                    "url": "https://x.com/1",
                    "snippet": "x" * 600,  # max_length=500
                    "source": "x.com",
                }
                for _ in range(10)
            ],
        }
        valid, msg = cs.validate(data)
        assert valid is False
        assert "500" in msg
        assert "characters" in msg


# ═══════════════════════════════════════════════════════════════════
#  Scenario 4: Real estate listings — 4 levels deep
# ═══════════════════════════════════════════════════════════════════

REALESTATE_SCHEMA = List(
    {
        "title": Field(str, min_length=5),
        "price": Field(float, min=0),
        "address": {
            "street": str,
            "city": str,
            "state": Field(str, min_length=2, max_length=2),
            "zip": Field(str, pattern=r"\d{5}"),
        },
        "bedrooms": Field(int, min=0),
        "bathrooms": Field(float, min=0),
        "sqft": Field(int, min=100, optional=True),
        "listing_type": Field(str, enum=["sale", "rent", "auction"]),
        "features": [str],
        "images": [
            {
                "url": Field(str, min_length=10),
                "alt": Field(str, optional=True),
            }
        ],
        "agent": {
            "name": str,
            "phone": Field(str, pattern=r"\d{3}-\d{3}-\d{4}", optional=True),
            "company": str,
        },
    },
    min_items=15,
)


class TestRealEstateScenario:
    @pytest.fixture
    def cs(self):
        return compile_schema(REALESTATE_SCHEMA)

    def _make_listing(self, i):
        return {
            "title": f"Beautiful Home in District {i}",
            "price": 250000.0 + i * 50000,
            "address": {
                "street": f"{100 + i} Main Street",
                "city": "Springfield",
                "state": "IL",
                "zip": f"{62700 + i}",
            },
            "bedrooms": 2 + (i % 4),
            "bathrooms": 1.5 + (i % 3) * 0.5,
            "sqft": 1200 + i * 100 if i % 3 != 0 else None,
            "listing_type": ["sale", "rent", "auction"][i % 3],
            "features": ["garage", "pool"] if i % 2 == 0 else ["garden"],
            "images": [
                {
                    "url": f"https://img.example.com/listing-{i}-{j}.jpg",
                    "alt": f"Photo {j}" if j % 2 == 0 else None,
                }
                for j in range(3)
            ],
            "agent": {
                "name": f"Agent {i}",
                "phone": f"555-{i:03d}-{1000 + i}" if i % 2 == 0 else None,
                "company": "Realty Corp",
            },
        }

    def test_fully_correct(self, cs):
        data = [self._make_listing(i) for i in range(20)]
        valid, msg = cs.validate(data)
        assert valid is True

    def test_deep_path_error_in_address(self, cs):
        data = [self._make_listing(i) for i in range(15)]
        data[5]["address"]["zip"] = "ABCDE"
        data[10]["address"]["state"] = "Illinois"  # too long, max 2

        valid, msg = cs.validate(data)
        assert valid is False
        assert "zip" in msg
        assert "state" in msg
        assert "does not match pattern" in msg

    def test_agent_phone_pattern_error(self, cs):
        data = [self._make_listing(i) for i in range(15)]
        for item in data:
            item["agent"]["phone"] = "(555) 123-4567"  # wrong format

        valid, msg = cs.validate(data)
        assert valid is False
        assert "phone" in msg
        assert "does not match pattern" in msg

    def test_sqft_optional_with_constraint(self, cs):
        """sqft is optional but when present must be >= 100."""
        data = [self._make_listing(i) for i in range(15)]
        data[3]["sqft"] = 50  # below min

        valid, msg = cs.validate(data)
        assert valid is False
        assert "sqft" in msg
        assert ">= 100" in msg or "100" in msg

    def test_multiple_nesting_errors_reported_clearly(self, cs):
        """Errors at multiple nesting levels."""
        data = [self._make_listing(i) for i in range(15)]
        # Level 0: wrong listing_type
        data[0]["listing_type"] = "unknown"
        # Level 1: address.zip wrong
        data[1]["address"]["zip"] = "bad"
        # Level 2: images[0].url too short
        data[2]["images"][0]["url"] = "http://x"  # < 10 chars
        # Level 1: agent.name wrong type
        data[3]["agent"]["name"] = 42

        valid, msg = cs.validate(data)
        assert valid is False
        # All errors should be mentioned
        assert "listing_type" in msg
        assert "zip" in msg
        assert "url" in msg or "images" in msg
        assert "name" in msg or "agent" in msg


# ═══════════════════════════════════════════════════════════════════
#  Scenario 5: Type coercion edge cases at scale
# ═══════════════════════════════════════════════════════════════════


class TestTypeCoercionEdgeCases:
    def test_int_field_accepts_json_numbers_without_decimals(self):
        """JSON has no int/float distinction. json.loads gives float for 42.0."""
        cs = compile_schema([{"count": int}])
        # When JSON is parsed, 42 becomes int, but 42.0 stays float
        valid, msg = cs.validate([{"count": 42}])
        assert valid is True

        valid, msg = cs.validate([{"count": 42.0}])
        assert valid is True

    def test_int_field_rejects_non_whole_float(self):
        cs = compile_schema([{"count": int}])
        valid, msg = cs.validate([{"count": 42.5}])
        assert valid is False

    def test_int_field_rejects_infinity(self):
        cs = compile_schema([{"count": int}])
        valid, msg = cs.validate([{"count": float("inf")}])
        assert valid is False

    def test_int_field_rejects_nan(self):
        cs = compile_schema([{"count": int}])
        valid, msg = cs.validate([{"count": float("nan")}])
        assert valid is False

    def test_float_field_accepts_int(self):
        cs = compile_schema([{"price": float}])
        valid, msg = cs.validate([{"price": 42}])
        assert valid is True

    def test_float_field_accepts_float(self):
        cs = compile_schema([{"price": float}])
        valid, msg = cs.validate([{"price": 42.5}])
        assert valid is True

    def test_bool_rejects_0_and_1(self):
        cs = compile_schema([{"flag": bool}])
        for val in [0, 1, 0.0, 1.0]:
            valid, msg = cs.validate([{"flag": val}])
            assert valid is False, f"bool should reject {val!r}"

    def test_bool_rejects_strings(self):
        cs = compile_schema([{"flag": bool}])
        for val in ["true", "false", "True", "False", "yes", "no", ""]:
            valid, msg = cs.validate([{"flag": val}])
            assert valid is False, f"bool should reject {val!r}"

    def test_str_rejects_numbers(self):
        cs = compile_schema([{"name": str}])
        for val in [42, 3.14, True, False, None]:
            if val is None:
                continue  # None is a separate check (null vs missing)
            valid, msg = cs.validate([{"name": val}])
            assert valid is False, f"str should reject {val!r}"

    def test_int_rejects_string_numbers(self):
        """'42' is NOT coerced to 42."""
        cs = compile_schema([{"count": int}])
        valid, msg = cs.validate([{"count": "42"}])
        assert valid is False
        assert "expected int" in msg

    def test_float_rejects_string_numbers(self):
        """'3.14' is NOT coerced to 3.14."""
        cs = compile_schema([{"price": float}])
        valid, msg = cs.validate([{"price": "3.14"}])
        assert valid is False


# ═══════════════════════════════════════════════════════════════════
#  Scenario 6: Error formatting stress tests
# ═══════════════════════════════════════════════════════════════════


class TestErrorFormattingStress:
    def test_many_different_error_types_capped_at_10(self):
        """15 fields all with wrong types — should cap at 10 groups."""
        schema = [{f"f{i}": int for i in range(15)}]
        cs = compile_schema(schema)
        data = [{f"f{i}": f"bad_{i}" for i in range(15)}]

        valid, msg = cs.validate(data)
        assert valid is False
        # Count error group markers at start of line: "  [N] "
        import re

        group_markers = re.findall(r"^\s+\[\d+\]", msg, re.MULTILINE)
        assert len(group_markers) == 10
        assert "and 5 more" in msg

    def test_hundreds_of_items_same_error(self):
        """200 items all with the same error — grouped into one."""
        cs = compile_schema([{"x": int}])
        data = [{"x": "bad"} for _ in range(200)]

        valid, msg = cs.validate(data)
        assert valid is False
        assert "200 of 200" in msg
        # Should show max 3 examples
        examples = msg.count("Examples:")
        assert examples == 1  # one group, one examples line

    def test_long_string_value_truncated_in_examples(self):
        """Very long string values should be truncated in error display."""
        cs = compile_schema([{"title": int}])
        long_str = "x" * 200
        data = [{"title": long_str}]

        valid, msg = cs.validate(data)
        assert valid is False
        # The value should be truncated, not show all 200 chars
        assert len(msg) < 500

    def test_error_header_singular(self):
        """One error type uses singular 'error'."""
        cs = compile_schema([{"x": int}])
        data = [{"x": "bad"}]
        valid, msg = cs.validate(data)
        assert "1 error)" in msg
        assert "1 error types)" not in msg

    def test_error_header_plural(self):
        """Multiple error types use plural 'error types'."""
        cs = compile_schema([{"x": int, "y": float}])
        data = [{"x": "a", "y": "b"}]
        valid, msg = cs.validate(data)
        assert "error types)" in msg

    def test_optional_note_appears_for_optional_constraint_error(self):
        """When an optional field has a constraint violation (not null,
        but invalid value), the note should mention null is valid."""
        cs = compile_schema([{"rating": Field(int, min=1, max=5, optional=True)}])
        data = [{"rating": 0}]  # present but invalid

        valid, msg = cs.validate(data)
        assert valid is False
        assert "optional" in msg.lower()
        assert "null is valid" in msg or "null" in msg


# ═══════════════════════════════════════════════════════════════════
#  Scenario 7: Hierarchical short-circuiting
#  Verify that child errors are NOT reported when parent fails
# ═══════════════════════════════════════════════════════════════════


class TestHierarchicalShortCircuiting:
    def test_wrong_type_does_not_recurse(self):
        """If address is a string, don't report missing street/city."""
        cs = compile_schema([{"address": {"street": str, "city": str, "zip": str}}])
        data = [{"address": "123 Main St"}]

        valid, msg = cs.validate(data)
        assert valid is False
        assert "expected an object" in msg
        # Should NOT mention street, city, zip as separate errors
        assert "street" not in msg
        assert "city" not in msg

    def test_missing_field_does_not_check_constraints(self):
        """If a field is missing, don't also report constraint errors."""
        cs = compile_schema(
            [
                {
                    "name": Field(str, min_length=10),
                    "age": Field(int, min=0),
                }
            ]
        )
        data = [{}]  # both missing

        valid, msg = cs.validate(data)
        assert valid is False
        assert "missing" in msg
        # Should NOT mention min_length or >= 0
        assert "10 character" not in msg
        assert ">= 0" not in msg

    def test_null_for_none_return(self):
        """None return is one error, not a cascade."""
        cs = compile_schema(
            List(
                {
                    "title": Field(str, min_length=1),
                    "price": Field(float, min=0),
                },
                min_items=20,
            )
        )

        valid, msg = cs.validate(None)
        assert valid is False
        assert "1 error)" in msg  # single error, not 3+
        assert "return statement" in msg


# ═══════════════════════════════════════════════════════════════════
#  Scenario 8: Prompt rendering for spec case 4 (exact match)
# ═══════════════════════════════════════════════════════════════════


class TestSpecCase4ExactMatch:
    """Verify the deeply nested spec example renders correctly."""

    @pytest.fixture
    def cs(self):
        return compile_schema(
            List(
                {
                    "category": str,
                    "products": List(
                        {
                            "title": Field(str, min_length=1),
                            "price": Field(float, min=0),
                            "rating": Field(int, min=1, max=5, optional=True),
                            "variants": [
                                {
                                    "color": str,
                                    "size": str,
                                    "in_stock": bool,
                                }
                            ],
                            "specs": dict,
                        },
                        min_items=1,
                    ),
                },
                min_items=3,
            )
        )

    def test_structure_has_correct_nesting(self, cs):
        struct = cs.prompt.split("### Structure")[1].split("### Requirements")[0]
        # Outer list
        assert '"category": ...,' in struct
        # Inner list
        assert '"title": ...,' in struct
        assert '"price": ...,' in struct
        # Innermost list
        assert '"color": ...,' in struct
        assert '"size": ...,' in struct
        assert '"in_stock": ...,' in struct

    def test_requirements_nesting(self, cs):
        reqs = cs.prompt.split("### Requirements")[1]
        assert "at least **3 items**" in reqs
        assert "`category`" in reqs
        assert "`products`" in reqs
        assert "at least **1 item**" in reqs
        assert "`title`" in reqs
        assert "`price`" in reqs
        assert "`rating`" in reqs
        assert "Optional" in reqs
        assert "between 1 and 5" in reqs
        assert "`variants`" in reqs
        assert "`color`" in reqs
        assert "`specs`" in reqs
        assert "freestyle" in reqs

    def test_optional_paragraph_present(self, cs):
        assert "For optional fields" in cs.prompt

    def test_comment_format(self, cs):
        struct = cs.prompt.split("### Structure")[1].split("### Requirements")[0]
        assert "# str, required" in struct
        assert "# float, required, >= 0" in struct
        assert "# int, optional, 1 to 5" in struct
        assert "# bool, required" in struct
        assert "# dict, freestyle" in struct
        assert "# minimum 3 items" in struct
        assert "# minimum 1 item" in struct


# ═══════════════════════════════════════════════════════════════════
#  Scenario 9: Full pipeline — compile → prompt → validate → format
#  Tests that all layers agree with each other
# ═══════════════════════════════════════════════════════════════════


class TestFullPipelineAgreement:
    """The prompt tells the agent what to return. The validator checks
    it. These tests verify they agree — if the prompt says 'required',
    the validator rejects when missing; if the prompt says 'optional',
    the validator accepts null."""

    def test_every_required_field_rejection(self):
        """For each required field in the prompt, removing it from
        data triggers a validation error mentioning that field."""
        schema = {
            "name": str,
            "age": int,
            "email": Field(str, pattern=r".+@.+"),
            "score": Field(float, min=0, max=100),
        }
        cs = compile_schema(schema)

        # Verify prompt says all are required
        for field in ["name", "age", "email", "score"]:
            assert "Required" in cs.prompt

        # Remove each field one at a time and verify error
        base = {"name": "Alice", "age": 30, "email": "a@b.com", "score": 85.0}
        for field in ["name", "age", "email", "score"]:
            data = {k: v for k, v in base.items() if k != field}
            valid, msg = cs.validate(data)
            assert valid is False, f"Missing {field} should fail"
            assert field in msg, f"Error should mention {field}"

    def test_optional_field_null_accepted(self):
        """Every optional field in the prompt accepts null."""
        schema = {
            "name": str,
            "bio": Field(str, optional=True),
            "rating": Field(int, min=1, max=5, optional=True),
        }
        cs = compile_schema(schema)
        assert "Optional" in cs.prompt

        data = {"name": "Alice", "bio": None, "rating": None}
        valid, msg = cs.validate(data)
        assert valid is True

    def test_enum_values_in_prompt_match_validator(self):
        """Every enum value shown in the prompt is accepted by the validator."""
        schema = [{"status": Field(str, enum=["active", "pending", "archived"])}]
        cs = compile_schema(schema)

        for val in ["active", "pending", "archived"]:
            assert f'"{val}"' in cs.prompt
            valid, msg = cs.validate([{"status": val}])
            assert valid is True, f"Enum value {val!r} shown in prompt but rejected"

    def test_constraint_bounds_match(self):
        """If prompt says 'between 1 and 5', values 1 and 5 must be valid."""
        cs = compile_schema([{"rating": Field(int, min=1, max=5)}])
        assert "between 1 and 5" in cs.prompt.lower() or "Between 1 and 5" in cs.prompt

        # Boundary values should be valid (inclusive)
        valid, _ = cs.validate([{"rating": 1}])
        assert valid is True
        valid, _ = cs.validate([{"rating": 5}])
        assert valid is True

        # Outside boundaries should fail
        valid, _ = cs.validate([{"rating": 0}])
        assert valid is False
        valid, _ = cs.validate([{"rating": 6}])
        assert valid is False
