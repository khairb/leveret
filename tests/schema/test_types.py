"""Tests for Layer 1: Field and List schema types."""

from scout.schema.types import Field, List, SchemaType


class TestField:
    """Field is a pure data container — stores parameters, no validation."""

    def test_bare_type_defaults(self):
        f = Field(str)
        assert f.type_ is str
        assert f.optional is False
        assert f.min is None
        assert f.max is None
        assert f.min_length is None
        assert f.max_length is None
        assert f.pattern is None
        assert f.enum is None

    def test_all_string_constraints(self):
        f = Field(str, min_length=5, max_length=100, pattern=r"\d+")
        assert f.min_length == 5
        assert f.max_length == 100
        assert f.pattern == r"\d+"

    def test_numeric_constraints(self):
        f = Field(int, min=0, max=100)
        assert f.type_ is int
        assert f.min == 0
        assert f.max == 100

    def test_float_constraints(self):
        f = Field(float, min=-1.5, max=1.5)
        assert f.type_ is float
        assert f.min == -1.5
        assert f.max == 1.5

    def test_bool_type(self):
        f = Field(bool)
        assert f.type_ is bool

    def test_optional(self):
        f = Field(str, optional=True)
        assert f.optional is True

    def test_optional_with_constraints(self):
        f = Field(int, min=1, max=5, optional=True)
        assert f.optional is True
        assert f.min == 1
        assert f.max == 5

    def test_enum(self):
        f = Field(str, enum=["small", "medium", "large"])
        assert f.enum == ["small", "medium", "large"]

    def test_enum_with_optional(self):
        f = Field(str, enum=["a", "b"], optional=True)
        assert f.enum == ["a", "b"]
        assert f.optional is True

    def test_repr_minimal(self):
        assert repr(Field(str)) == "Field(str)"
        assert repr(Field(int)) == "Field(int)"

    def test_repr_with_constraints(self):
        r = repr(Field(float, min=0, max=1))
        assert "float" in r
        assert "min=0" in r
        assert "max=1" in r

    def test_repr_includes_all_set_params(self):
        f = Field(str, min_length=1, max_length=50, pattern=r"\w+", enum=["a"], optional=True)
        r = repr(f)
        for part in ["min_length=1", "max_length=50", "pattern=", "enum=", "optional=True"]:
            assert part in r, f"Missing {part!r} in repr: {r}"

    def test_type_is_stored_as_type_object(self):
        """The first arg is stored as the actual type object, not a string."""
        f = Field(str)
        assert f.type_ is str  # identity, not equality
        assert f.type_("hello") == "hello"  # it's the real str constructor


class TestList:
    """List is a pure data container — stores item + constraints."""

    def test_bare_item(self):
        lst = List(str)
        assert lst.item is str
        assert lst.min_items is None
        assert lst.max_items is None

    def test_with_constraints(self):
        lst = List(str, min_items=5, max_items=50)
        assert lst.min_items == 5
        assert lst.max_items == 50

    def test_dict_item(self):
        lst = List({"title": str, "price": float}, min_items=20)
        assert isinstance(lst.item, dict)
        assert lst.min_items == 20

    def test_nested_field_item(self):
        lst = List(Field(str, min_length=1), min_items=5)
        assert isinstance(lst.item, Field)

    def test_nested_list_item(self):
        inner = List(str, min_items=1)
        outer = List(inner, min_items=5)
        assert isinstance(outer.item, List)

    def test_repr_with_constraints(self):
        r = repr(List(str, min_items=5, max_items=10))
        assert "min_items=5" in r
        assert "max_items=10" in r

    def test_repr_no_constraints(self):
        r = repr(List(str))
        assert "min" not in r
        assert "max" not in r


class TestSchemaType:
    """SchemaType is a structural alias for type annotations."""

    def test_exists(self):
        assert SchemaType is not None
