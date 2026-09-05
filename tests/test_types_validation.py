"""
Tests for type validation in dataclasses.
"""

from dataclasses import dataclass
from hola.errors import validated_dataclass, value_assert
from hola.ollama.config import ClientConfig
from hola.types import (
    ChatRequest,
    GenerationOptions,
    GenerateRequest,
    Message,
    Role,
)
from hola.types import (
    GenerateChunk,
    GenerateResponse,
)
from hola.tools.types import (
    ToolCall,
    ToolCallFunction,
    ToolDefinition,
)
import unittest


class TestTypeValidation(unittest.TestCase):
    """Test type validation catches errors."""

    def test_message_role_string_not_enum(self):
        """Test that Message with string role raises TypeError."""
        with self.assertRaises(TypeError) as ctx:
            Message(role="assistant", content="hi")

        assert "Message.role" in str(ctx.exception)
        assert "Role" in str(ctx.exception)
        assert "str" in str(ctx.exception)

    def test_message_role_correct_enum(self):
        """Test that Message with Role enum works."""
        msg = Message(role=Role.USER, content="hi")
        assert msg.role == Role.USER
        assert msg.content == "hi"

    def test_message_optional_fields_none_allowed(self):
        """Test that None is allowed for Optional fields."""
        msg = Message(
            role=Role.USER,
            content="hi",
            tool_calls=None
        )
        assert msg.tool_calls is None

    def test_message_tool_calls_wrong_type(self):
        """Test that tool_calls must be list or None."""
        with self.assertRaises(TypeError) as ctx:
            Message(role=Role.USER, content="hi", tool_calls="not a list")

        assert "Message.tool_calls" in str(ctx.exception)
        assert "list" in str(ctx.exception)

    def test_tool_call_function_wrong_type(self):
        """Test that function must be ToolCallFunction."""
        with self.assertRaises(TypeError) as ctx:
            ToolCall(function="not a function object")

        assert "ToolCall.function" in str(ctx.exception)
        assert "ToolCallFunction" in str(ctx.exception)

    def test_tool_call_function_correct_type(self):
        """Test that ToolCall with correct function works."""
        func = ToolCallFunction(name="test", arguments={})
        tc = ToolCall(function=func)
        assert tc.function == func
        assert tc.function.name == "test"

    def test_nested_list_validation(self):
        """Test that list element types are validated."""
        func = ToolCallFunction(name="test", arguments={})
        tool_call = ToolCall(function=func)

        msg = Message(role=Role.USER, content="hi", tool_calls=[tool_call])
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0] == tool_call

    def test_nested_list_wrong_element_type(self):
        """Test that list element types are validated."""
        with self.assertRaises(TypeError) as ctx:
            Message(role=Role.USER, content="hi", tool_calls=[{}])

        assert "Message.tool_calls" in str(ctx.exception)
        assert "ToolCall" in str(ctx.exception)

    def test_chat_request_messages_wrong_type(self):
        """Test that messages must be list of Message objects."""
        with self.assertRaises(TypeError) as ctx:
            ChatRequest(model="test", messages=["not message objects"])

        assert "ChatRequest.messages" in str(ctx.exception)
        assert "Message" in str(ctx.exception)

    def test_chat_request_messages_correct_type(self):
        """Test that ChatRequest with correct messages works."""
        msg = Message(role=Role.USER, content="hi")
        req = ChatRequest(model="test", messages=[msg])
        assert len(req.messages) == 1
        assert req.messages[0] == msg

    def test_generation_options_numeric_types(self):
        """Test that GenerationOptions numeric types are validated."""
        opts = GenerationOptions(
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            num_predict=100,
            seed=42
        )
        assert opts.temperature == 0.7
        assert opts.top_p == 0.9
        assert opts.top_k == 40
        assert opts.num_predict == 100
        assert opts.seed == 42

    def test_generation_options_wrong_numeric_type(self):
        """Test that GenerationOptions rejects wrong types."""
        with self.assertRaises(TypeError) as ctx:
            GenerationOptions(temperature="not a float")

        assert "GenerationOptions.temperature" in str(ctx.exception)
        assert "float" in str(ctx.exception)

    def test_tool_definition_correct_type(self):
        """Test that ToolDefinition with correct fields works."""
        td = ToolDefinition(
            name="test",
            description="test function",
            parameters={}
        )
        assert td.name == "test"
        assert td.description == "test function"
        assert td.parameters == {}

    def test_generate_request_wrong_type(self):
        """Test that GenerateRequest has correct types."""
        with self.assertRaises(TypeError) as ctx:
            GenerateRequest(model=123, prompt="test")

        assert "GenerateRequest.model" in str(ctx.exception)
        assert "str" in str(ctx.exception)

    def test_generate_request_correct_types(self):
        """Test that GenerateRequest with correct types works."""
        req = GenerateRequest(
            model="test",
            prompt="test",
            system="system prompt",
            stream=False,
            keep_alive="1h"
        )
        assert req.model == "test"
        assert req.prompt == "test"
        assert req.system == "system prompt"
        assert req.stream is False
        assert req.keep_alive == "1h"

    def test_nested_tool_calls_validation(self):
        """Test that nested tool_calls are validated."""
        inner_func = ToolCallFunction(name="inner", arguments={})
        inner_tc = ToolCall(function=inner_func)
        outer_func = ToolCallFunction(name="outer", arguments={"data":inner_tc})
        outer_tc = ToolCall(function=outer_func)


class TestPostInitChaining(unittest.TestCase):
    """Custom `__post_init__` inheritance semantics for
        `validated_dataclass` classes.
    """

    def test_subclass_runs_base_custom_post_init(self):
        """A decorated subclass runs the decorated base's custom
            `__post_init__` value checks.
        """
        @validated_dataclass
        class Base:
            n: int = 1

            def __post_init__(self) -> None:
                value_assert(self.n >= 1, "n must be >= 1")

        @validated_dataclass
        class Child(Base):
            m: int = 2

        assert Child(n=1).n == 1
        with self.assertRaises(ValueError):
            Child(n=0)

    def test_subclass_post_init_overrides_base(self):
        """Only the subclass's own `__post_init__` runs, matching
            standard Python override semantics.
        """
        calls: list[str] = []

        @validated_dataclass
        class Base:
            n: int = 1

            def __post_init__(self) -> None:
                calls.append("base")

        @validated_dataclass
        class Child(Base):
            m: int = 2

            def __post_init__(self) -> None:
                calls.append("child")

        Child()
        assert calls == ["child"]

    def test_plain_dataclass_base_post_init_runs(self):
        """A decorated class inherits a plain dataclass base's
            `__post_init__`.
        """
        calls: list[str] = []

        @dataclass
        class PlainBase:
            n: int = 1

            def __post_init__(self) -> None:
                calls.append("plain")

        @validated_dataclass
        class Child(PlainBase):
            m: int = 2

        Child()
        assert calls == ["plain"]


class TestBoolRejection(unittest.TestCase):
    """`bool` values are rejected for `int` and `float` fields in all
        positions (scalar, list elements, dict values, union arms),
        but remain valid for `bool` fields.
    """

    def test_bool_rejected_for_int_field(self):
        """Test that bools raise TypeError for int fields."""
        @validated_dataclass
        class Counter:
            n: int

        with self.assertRaises(TypeError):
            Counter(n=True)

        with self.assertRaises(TypeError):
            Counter(n=False)

    def test_bool_rejected_for_int_containers(self):
        """Test that bools are rejected inside list/dict of int."""
        @validated_dataclass
        class Bucket:
            xs: list[int]
            counts: dict[str, int]

        with self.assertRaises(TypeError):
            Bucket(xs=[1, True], counts={})

        with self.assertRaises(TypeError):
            Bucket(xs=[1], counts={"a": True})

    def test_bool_rejected_for_int_in_union(self):
        """Test that bools are rejected for `int | None` fields."""
        @validated_dataclass
        class OptionalCount:
            n: int | None = None

        with self.assertRaises(TypeError):
            OptionalCount(n=True)

        assert OptionalCount(n=5).n == 5
        assert OptionalCount(n=None).n is None

    def test_bool_field_still_accepts_bool(self):
        """Test that bool fields are unaffected by the rejection."""
        @validated_dataclass
        class Flags:
            on: bool = False

        assert Flags(on=True).on is True
        assert Flags(on=False).on is False

    def test_real_types_reject_bool_for_int_fields(self):
        """Test that real codebase types reject bools for int fields."""
        with self.assertRaises(TypeError):
            ClientConfig(max_retries=True)

        with self.assertRaises(TypeError):
            GenerationOptions(seed=True)


class TestNumericTowerRecast(unittest.TestCase):
    """`int` is accepted for `float` fields and recast to `float`, per
        the PEP 484 numeric tower; `bool` and non-numeric values are
        rejected.
    """

    def test_client_config_recasts_int_to_float(self):
        """Test that ClientConfig recasts int timeouts to float."""
        config = ClientConfig(timeout=30, retry_delay=2)
        assert config.timeout == 30.0
        assert isinstance(config.timeout, float)
        assert config.retry_delay == 2.0
        assert isinstance(config.retry_delay, float)

    def test_optional_float_field_recasts_int(self):
        """Test that int values are recast for `float | None` fields."""
        opts = GenerationOptions(temperature=0, top_p=1)
        assert opts.temperature == 0.0
        assert isinstance(opts.temperature, float)
        assert opts.top_p == 1.0
        assert isinstance(opts.top_p, float)

    def test_bool_rejected_for_float_fields(self):
        """Test that bool is rejected for float fields."""
        with self.assertRaises(TypeError):
            ClientConfig(timeout=True)

        with self.assertRaises(TypeError):
            GenerationOptions(temperature=True)

    def test_non_numeric_still_rejected_for_float_fields(self):
        """Test that non-numeric values are still rejected."""
        with self.assertRaises(TypeError):
            ClientConfig(timeout="30")

        with self.assertRaises(TypeError):
            GenerationOptions(temperature="hot")


class TestContainerRecast(unittest.TestCase):
    """`int` elements/values inside `list` and `dict` fields are
        recast to `float` in place, per the PEP 484 numeric tower;
        invalid elements are still rejected.
    """

    def test_list_elements_recast(self):
        """Test that int elements of `list[float]` fields recast."""
        @validated_dataclass
        class Vector:
            xs: list[float]

        v = Vector(xs=[1, 2.5, 3])
        assert v.xs == [1.0, 2.5, 3.0]
        assert all(isinstance(x, float) for x in v.xs)

    def test_nested_list_elements_recast(self):
        """Test that recasting propagates into nested lists."""
        @validated_dataclass
        class Matrix:
            rows: list[list[float]]

        m = Matrix(rows=[[1], [2, 3]])
        assert m.rows == [[1.0], [2.0, 3.0]]
        assert all(isinstance(x, float) for row in m.rows for x in row)

    def test_dict_values_recast(self):
        """Test that int values of `dict[str, float]` fields recast."""
        @validated_dataclass
        class Settings:
            values: dict[str, float]

        s = Settings(values={"a": 1, "b": 2.5})
        assert s.values == {"a": 1.0, "b": 2.5}
        assert isinstance(s.values["a"], float)

    def test_optional_list_union_recasts(self):
        """Test recasting through the `list[float] | None` union path."""
        @validated_dataclass
        class OptionalVector:
            xs: list[float] | None = None

        ov = OptionalVector(xs=[1, 2])
        assert ov.xs == [1.0, 2.0]
        assert all(isinstance(x, float) for x in ov.xs)
        assert OptionalVector(xs=None).xs is None

    def test_container_bad_elements_rejected(self):
        """Test that invalid container elements still raise TypeError."""
        @validated_dataclass
        class Vector:
            xs: list[float]

        @validated_dataclass
        class Settings:
            values: dict[str, float]

        with self.assertRaises(TypeError):
            Vector(xs=[1, True])

        with self.assertRaises(TypeError):
            Settings(values={"a": "hot"})


if __name__ == "__main__":
    unittest.main()
