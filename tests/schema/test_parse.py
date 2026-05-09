"""Tests for Layer 2: Schema parser and constraint validation."""

import pytest

from scout.errors import ScoutSchemaError
from scout.schema.nodes import (
    FreestyleDictNode,
    ListNode,
    ObjectNode,
    ScalarNode,
)
from scout.schema.parse import parse_schema
from scout.schema.types import Field, List


# ---------------------------------------------------------------------------
# Happy paths — every valid schema form
# ---------------------------------------------------------------------------

class TestParseBaretypes:
    """Bare Python types → ScalarNode or FreestyleDictNode."""

    @pytest.mark.parametrize("t", [str, int, float, bool])
    def test_bare_scalar_types(self, t):
        node = parse_schema(t)
        assert isinstance(node, ScalarNode)
        assert node.type_ is t
        assert node.optional is False

    def test_bare_dict_type(self):
        node = parse_schema(dict)
        assert isinstance(node, FreestyleDictNode)


class TestParseDicts:
    """Dict schemas → ObjectNode."""

    def test_simple_object(self):
        node = parse_schema({"title": str, "price": float})
        assert isinstance(node, ObjectNode)
        assert set(node.fields.keys()) == {"title", "price"}
        title_node, title_opt = node.fields["title"]
        assert isinstance(title_node, ScalarNode)
        assert title_node.type_ is str
        assert title_opt is False

    def test_field_in_object_extracts_optional(self):
        node = parse_schema({"bio": Field(str, optional=True)})
        bio_node, bio_opt = node.fields["bio"]
        assert bio_opt is True
        assert isinstance(bio_node, ScalarNode)
        assert bio_node.optional is True

    def test_field_order_preserved(self):
        schema = {"zebra": str, "apple": int, "mango": float}
        node = parse_schema(schema)
        assert list(node.fields.keys()) == ["zebra", "apple", "mango"]

    def test_nested_object(self):
        node = parse_schema({"address": {"street": str, "city": str}})
        addr_node, _ = node.fields["address"]
        assert isinstance(addr_node, ObjectNode)
        assert "street" in addr_node.fields

    def test_nested_list_in_object(self):
        node = parse_schema({"tags": [str]})
        tags_node, _ = node.fields["tags"]
        assert isinstance(tags_node, ListNode)
        assert isinstance(tags_node.item, ScalarNode)

    def test_freestyle_dict_in_object(self):
        node = parse_schema({"specs": dict})
        specs_node, _ = node.fields["specs"]
        assert isinstance(specs_node, FreestyleDictNode)


class TestParseLists:
    """List schemas → ListNode."""

    def test_bare_list_syntax(self):
        node = parse_schema([str])
        assert isinstance(node, ListNode)
        assert isinstance(node.item, ScalarNode)
        assert node.min is None
        assert node.max is None

    def test_list_of_objects(self):
        node = parse_schema([{"title": str}])
        assert isinstance(node, ListNode)
        assert isinstance(node.item, ObjectNode)

    def test_list_class_with_constraints(self):
        node = parse_schema(List(str, min=5, max=50))
        assert isinstance(node, ListNode)
        assert node.min == 5
        assert node.max == 50

    def test_list_class_with_dict_item(self):
        node = parse_schema(List({"title": str, "price": float}, min=20))
        assert isinstance(node, ListNode)
        assert isinstance(node.item, ObjectNode)
        assert node.min == 20


class TestParseField:
    """Field schemas → ScalarNode with constraints."""

    def test_field_with_all_string_constraints(self):
        node = parse_schema(Field(str, min_length=5, max_length=100, pattern=r"\d+"))
        assert isinstance(node, ScalarNode)
        assert node.min_length == 5
        assert node.max_length == 100
        assert node.pattern == r"\d+"

    def test_field_with_numeric_constraints(self):
        node = parse_schema(Field(int, min=1, max=5))
        assert node.min == 1
        assert node.max == 5

    def test_field_with_enum(self):
        node = parse_schema(Field(str, enum=["a", "b", "c"]))
        assert node.enum == ["a", "b", "c"]


class TestParseDeepNesting:
    """Deeply nested schemas parse correctly."""

    def test_three_levels_deep(self):
        schema = List({
            "categories": [{
                "products": [{
                    "title": str,
                    "variants": [{"color": str, "size": str}],
                }],
            }],
        }, min=1)
        node = parse_schema(schema)
        assert isinstance(node, ListNode)
        cat_node = node.item.fields["categories"][0]
        assert isinstance(cat_node, ListNode)
        prod_node = cat_node.item.fields["products"][0]
        assert isinstance(prod_node, ListNode)
        var_node = prod_node.item.fields["variants"][0]
        assert isinstance(var_node, ListNode)
        assert "color" in var_node.item.fields


# ---------------------------------------------------------------------------
# Error cases — every invalid schema form
# ---------------------------------------------------------------------------

class TestParseErrors:
    """Invalid schemas raise ScoutSchemaError with clear messages."""

    def test_non_string_key(self):
        with pytest.raises(ScoutSchemaError, match="strings"):
            parse_schema({42: str})

    def test_multi_element_list(self):
        with pytest.raises(ScoutSchemaError, match="exactly one element"):
            parse_schema([str, int])

    def test_empty_list(self):
        with pytest.raises(ScoutSchemaError, match="exactly one element"):
            parse_schema([])

    def test_invalid_value_type(self):
        with pytest.raises(ScoutSchemaError, match="not a value"):
            parse_schema(42)

    def test_invalid_value_in_object_shows_field_name(self):
        with pytest.raises(ScoutSchemaError, match="field 'title'"):
            parse_schema({"title": 42})

    def test_field_list_type_special_message(self):
        with pytest.raises(ScoutSchemaError, match="Use List()"):
            parse_schema(Field(list))

    def test_field_invalid_type(self):
        with pytest.raises(ScoutSchemaError, match="Allowed types"):
            parse_schema(Field(bytes))


class TestFieldConstraintErrors:
    """Field constraint validation catches incompatible combinations."""

    def test_min_max_on_str(self):
        with pytest.raises(ScoutSchemaError, match="min_length"):
            parse_schema(Field(str, min=5))

    def test_min_length_on_int(self):
        with pytest.raises(ScoutSchemaError, match="min.*max"):
            parse_schema(Field(int, min_length=5))

    def test_pattern_on_int(self):
        with pytest.raises(ScoutSchemaError, match="pattern"):
            parse_schema(Field(int, pattern=r"."))

    def test_min_greater_than_max(self):
        with pytest.raises(ScoutSchemaError, match=r"'min' \(10\) must be <= 'max' \(5\)"):
            parse_schema(Field(int, min=10, max=5))

    def test_min_length_greater_than_max_length(self):
        with pytest.raises(ScoutSchemaError, match="min_length"):
            parse_schema(Field(str, min_length=10, max_length=5))

    def test_negative_min_length(self):
        with pytest.raises(ScoutSchemaError, match="non-negative"):
            parse_schema(Field(str, min_length=-1))

    def test_negative_max_length(self):
        with pytest.raises(ScoutSchemaError, match="non-negative"):
            parse_schema(Field(str, max_length=-1))


class TestEnumConstraintErrors:
    """Enum constraint validation."""

    def test_enum_on_non_str(self):
        with pytest.raises(ScoutSchemaError, match="'enum' is not valid"):
            parse_schema(Field(int, enum=[1, 2]))

    def test_enum_not_a_list(self):
        with pytest.raises(ScoutSchemaError, match="list of strings"):
            parse_schema(Field(str, enum="small"))

    def test_enum_empty(self):
        with pytest.raises(ScoutSchemaError, match="must not be empty"):
            parse_schema(Field(str, enum=[]))

    def test_enum_non_string_values(self):
        with pytest.raises(ScoutSchemaError, match="must be strings"):
            parse_schema(Field(str, enum=["a", 42]))

    def test_enum_with_pattern(self):
        with pytest.raises(ScoutSchemaError, match="cannot be combined"):
            parse_schema(Field(str, enum=["a"], pattern=r"."))

    def test_enum_with_min_length(self):
        with pytest.raises(ScoutSchemaError, match="cannot be combined"):
            parse_schema(Field(str, enum=["a"], min_length=1))

    def test_enum_with_max_length(self):
        with pytest.raises(ScoutSchemaError, match="cannot be combined"):
            parse_schema(Field(str, enum=["a"], max_length=10))

    def test_enum_with_optional_is_valid(self):
        """enum + optional is allowed (null valid, non-null must be in list)."""
        node = parse_schema(Field(str, enum=["a", "b"], optional=True))
        assert node.enum == ["a", "b"]
        assert node.optional is True


class TestListConstraintErrors:
    """List constraint validation."""

    def test_negative_min(self):
        with pytest.raises(ScoutSchemaError, match="non-negative"):
            parse_schema(List(str, min=-1))

    def test_negative_max(self):
        with pytest.raises(ScoutSchemaError, match="non-negative"):
            parse_schema(List(str, max=-1))

    def test_min_greater_than_max(self):
        with pytest.raises(ScoutSchemaError, match=r"'min' \(10\) must be <= 'max' \(5\)"):
            parse_schema(List(str, min=10, max=5))
