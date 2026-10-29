"""
Tests for OpenAI-compatible BaseClient request and response mapping.
"""

from hola.client import BaseClient, ClientConfig
from hola.tools.types import ToolCall, ToolCallFunction
from hola.types import (
    ChatRequest,
    GenerateRequest,
    GenerationOptions,
    Message,
    Role,
    ToolDefinition,
)
import json
import unittest


class TestOpenAIClient(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.client = BaseClient(
            ClientConfig(base_url="http://localhost:1")
        )

    async def asyncTearDown(self):
        await self.client.close()

    async def test_chat_request_serialization(self):
        tool_call = ToolCall(
            function=ToolCallFunction(name="lookup", arguments={"q": "x"}),
        )
        request = ChatRequest(
            model="test-model",
            messages=[
                Message(
                    role=Role.ASSISTANT,
                    content="",
                    tool_calls=[tool_call],
                ),
                Message(
                    role=Role.TOOL,
                    content="result",
                    call_id=tool_call.call_id,
                ),
            ],
            tools=[
                ToolDefinition(
                    name="lookup",
                    description="Look something up.",
                    parameters={"type": "object"},
                ),
            ],
            stream=False,
            options=GenerationOptions(
                temperature=0.25,
                top_p=0.9,
                top_k=5,
                num_predict=32,
                seed=7,
            ),
            keep_alive="1h",
        )
        data = self.client._serialize_chat_request(request)
        assert data["temperature"] == 0.25
        assert data["top_p"] == 0.9
        assert data["top_k"] == 5
        assert data["max_tokens"] == 32
        assert data["seed"] == 7
        assert "options" not in data
        assert "keep_alive" not in data
        assert data["tools"][0]["type"] == "function"
        serialized_call = data["messages"][0]["tool_calls"][0]
        assert serialized_call["id"] == tool_call.call_id
        assert serialized_call["type"] == "function"
        assert json.loads(
            serialized_call["function"]["arguments"]
        ) == {"q": "x"}
        assert data["messages"][1]["tool_call_id"] == tool_call.call_id

    async def test_nonstream_chat_parsing(self):
        data = {
            "model": "test-model",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "hello",
                        "reasoning_content": "thinking",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "lookup",
                                    "arguments": "{\"q\": \"x\"}",
                                },
                            },
                        ],
                    },
                },
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 6},
        }
        response = self.client._parse_chat_response(data, "fallback")
        assert response.model == "test-model"
        assert response.message.content == "hello"
        assert response.message.thinking == "thinking"
        assert response.message.tool_calls[0].call_id == "call_1"
        assert response.message.tool_calls[0].function.name == "lookup"
        assert response.message.tool_calls[0].function.arguments == {"q": "x"}
        assert response.prompt_tokens == 5
        assert response.output_tokens == 6

    async def test_stream_tool_call_accumulation(self):
        accumulator: dict[int, dict] = {}
        self.client._accumulate_tool_calls(
            accumulator,
            [
                {
                    "index": 0,
                    "id": "call_1",
                    "function": {
                        "name": "lookup",
                        "arguments": "{\"q\":",
                    },
                },
            ],
        )
        self.client._accumulate_tool_calls(
            accumulator,
            [{"index": 0, "function": {"arguments": " \"x\"}"}}],
        )
        calls = self.client._parse_tool_calls(list(accumulator.values()))
        assert calls[0].call_id == "call_1"
        assert calls[0].function.name == "lookup"
        assert calls[0].function.arguments == {"q": "x"}

    async def test_generate_request_and_response(self):
        request = GenerateRequest(
            model="test-model",
            prompt="hello",
            system="be nice",
            stream=False,
            options=GenerationOptions(num_predict=4, temperature=0.1),
        )
        data = self.client._serialize_generate_request(request)
        assert data["prompt"] == "be nice\n\nhello"
        assert data["max_tokens"] == 4
        assert data["temperature"] == 0.1
        assert "options" not in data
        response = self.client._parse_generate_response(
            {
                "model": "test-model",
                "choices": [{"text": "hi", "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 3},
            },
            "fallback",
        )
        assert response.response == "hi"
        assert response.done is True
        assert response.prompt_tokens == 2
        assert response.output_tokens == 3


if __name__ == "__main__":
    unittest.main()
