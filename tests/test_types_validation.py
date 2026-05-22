"""
Tests for type validation in dataclasses.
"""

import unittest
from hola.types import (
    ChatRequest,
    GenerationOptions,
    GenerateRequest,
    Message,
    Role,
)
from hola.ollama.types import (
    GenerateChunk,
    GenerateResponse,
)
from hola.tools.types import (
    ToolCall,
    ToolCallFunction,
    ToolDefinition,
)


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
            tool_calls=None,
            tool_name=None
        )
        assert msg.tool_calls is None
        assert msg.tool_name is None

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
        func = ToolCallFunction(name="inner", arguments={})
        inner_tc = ToolCall(function=func)
        outer_tc = ToolCall(function=func, nested_calls=[inner_tc])

        assert outer_tc.nested_calls is not None
        assert len(outer_tc.nested_calls) == 1
        assert outer_tc.nested_calls[0] == inner_tc

    def test_nested_tool_calls_wrong_type(self):
        """Test that nested tool_calls rejects wrong types."""
        func = ToolCallFunction(name="test", arguments={})
        with self.assertRaises(TypeError) as ctx:
            ToolCall(function=func, nested_calls=["not a tool call"])

        assert "ToolCall.nested_calls" in str(ctx.exception)
        assert "ToolCall" in str(ctx.exception)


if __name__ == "__main__":
    unittest.main()
