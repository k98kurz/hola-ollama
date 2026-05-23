"""
Tests for OllamaClient functionality.
"""

import asyncio
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, Mock, patch, MagicMock
import unittest
import logging
import httpx
from hola.ollama.client import OllamaClient
from hola.ollama.config import ClientConfig
from hola.ollama.exceptions import (
    OllamaConnectionError,
    OllamaRequestError,
    OllamaModelError,
)
from hola.ollama.types import (
    RunningModel,
)
from hola.types import (
    ChatRequest,
    GenerateRequest,
    GenerationOptions,
    LLMClient,
    Message,
    Role,
    ChatResponse,
    GenerateChunk,
    GenerateResponse,
    ModelInfo,
    ModelMetadata,
)

from hola.cache import AsyncLRUCache, model_info_cache


class TestClient(unittest.IsolatedAsyncioTestCase):
    """Test cases for OllamaClient."""
    @classmethod
    def setUpClass(cls):
        """Suppress error logs from hola.ollama.client during tests."""
        cls.client_logger = logging.getLogger('hola.ollama.client')
        cls.original_level = cls.client_logger.level
        cls.client_logger.setLevel(logging.CRITICAL)

    @classmethod
    def tearDownClass(cls):
        """Restore original logging level."""
        cls.client_logger.setLevel(cls.original_level)

    def setUp(self):
        """Set up test fixtures."""
        self.client = OllamaClient()

    async def asyncTearDown(self):
        """Clean up after tests."""
        await self.client.close()
        await model_info_cache.clear()

    async def test_client_initialization(self):
        """Test client initialization with defaults."""
        assert self.client._config.base_url == "http://localhost:11434", (
            self.client._config.base_url,
            "http://localhost:11434",
        )
        assert self.client._config.timeout == 120.0, (
            self.client._config.timeout,
            120.0,
        )

    async def test_client_initialization_custom_config(self):
        """Test client initialization with custom config."""
        custom_client = OllamaClient(
            base_url="http://localhost:11435",
            timeout=60.0,
        )
        assert custom_client._config.base_url == "http://localhost:11435", (
            custom_client._config.base_url,
            "http://localhost:11435",
        )
        assert custom_client._config.timeout == 60.0, (
            custom_client._config.timeout,
            60.0,
        )
        await custom_client.close()

    async def test_protocol_compatibility(self):
        """Test that OllamaClient implements LLMClient Protocol."""
        assert isinstance(self.client, LLMClient), (
            "OllamaClient should be compatible with LLMClient Protocol"
        )

        custom_client = OllamaClient(base_url="http://localhost:11435")
        assert isinstance(custom_client, LLMClient), (
            "Custom OllamaClient should be compatible with LLMClient Protocol"
        )
        await custom_client.close()

    async def test_streaming_operations(self):
        """Test streaming operations for both chat and generate."""
        for request_type, make_request in [
            ("chat", lambda: ChatRequest(
                model="qwen3:0.6b",
                messages=[Message(role=Role.USER, content="Say hello")],
            )),
            ("generate", lambda: GenerateRequest(
                model="qwen3:0.6b",
                prompt="Say hello",
            )),
        ]:
            request = make_request()
            chunks = []
            accumulated_thinking = []
            accumulated_content = []

            async for chunk in self.client.chat(request) if request_type == "chat" else self.client.generate(request):
                chunks.append(chunk)

                if request_type == "chat":
                    if chunk.message.thinking:
                        accumulated_thinking.append(chunk.message.thinking)
                    if chunk.message.content:
                        accumulated_content.append(chunk.message.content)
                else:
                    if chunk.response:
                        accumulated_content.append(chunk.response)

                if chunk.done:
                    break

            assert len(chunks) > 0, chunks

            if request_type == "chat":
                assert any(chunk.message.content for chunk in chunks), chunks
                complete_thinking = "".join(accumulated_thinking)
                assert len(accumulated_thinking) > 0, (
                    f"Expected thinking chunks but got none. Content: {''.join(accumulated_content)}"
                )
                assert len(complete_thinking) > 0, (
                    f"Expected non-empty thinking but got empty. Content: {''.join(accumulated_content)}"
                )
            else:
                assert any(chunk.response for chunk in chunks), chunks
                complete_content = "".join(accumulated_content)
                assert len(complete_content) > 0, (
                    f"Expected non-empty response but got empty."
                )

            assert chunks[-1].done, chunks[-1].done

    async def test_sync_operations(self):
        """Test sync operations for both chat and generate."""
        for request_type, request_class, mock_json in [
            ("chat", ChatRequest, {
                "model": "granite4.1:3b",
                "message": {
                    "role": "assistant",
                    "content": "Hello world",
                    "thinking": "I should greet the user."
                },
                "done": True,
                "total_duration": 1000000000,
            }),
            ("generate", GenerateRequest, {
                "model": "granite4.1:3b",
                "response": "Hello world",
                "done": True,
                "total_duration": 1000000000,
            }),
        ]:
            if request_type == "chat":
                request = ChatRequest(
                    model="granite4.1:3b",
                    messages=[Message(role=Role.USER, content="test")],
                )
            else:
                request = GenerateRequest(
                    model="granite4.1:3b",
                    prompt="test",
                )

            mock_response = MagicMock()
            mock_response.json.return_value = mock_json

            async def mock_post(*args, **kwargs):
                return mock_response

            client = OllamaClient()
            client._client.post = mock_post

            if request_type == "chat":
                def run_sync():
                    return client.chat_sync(request)

                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    run_sync,
                )

                assert response.model == "granite4.1:3b", (
                    response.model,
                    "granite4.1:3b",
                )
                assert response.message.content == "Hello world", (
                    response.message.content,
                    "Hello world",
                )
                assert response.message.thinking == "I should greet the user.", (
                    response.message.thinking,
                    "I should greet the user.",
                )
            else:
                def run_sync():
                    return client.generate_sync(request)

                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    run_sync,
                )

                assert response.model == "granite4.1:3b", (
                    response.model,
                    "granite4.1:3b",
                )
                assert response.response == "Hello world", (
                    response.response,
                    "Hello world",
                )

            await client.close()

    async def test_e2e_operations(self):
        """Test end-to-end operations for both chat and generate."""
        for request_type, make_request in [
            ("chat", lambda: ChatRequest(
                model="granite4.1:3b",
                messages=[Message(role=Role.USER, content="Say hello")],
                options=GenerationOptions(num_predict=10),
            )),
            ("generate", lambda: GenerateRequest(
                model="granite4.1:3b",
                prompt="Say hello",
                options=GenerationOptions(num_predict=10),
            )),
        ]:
            request = make_request()
            chunks = []
            async for chunk in self.client.chat(request) if request_type == "chat" else self.client.generate(request):
                chunks.append(chunk)
                if chunk.done:
                    break

            assert len(chunks) > 0, chunks
            if request_type == "chat":
                assert any(chunk.message.content for chunk in chunks), chunks
            else:
                assert any(chunk.response for chunk in chunks), chunks
            assert chunks[-1].done, chunks[-1].done

    async def test_load_model(self):
        """Test model loading."""
        mock_response = Mock()
        with patch.object(
            self.client._client,
            'post',
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_post:
            await self.client.load_model("granite4.1:3b")

            mock_post.assert_called_once()
            call_kwargs = mock_post.call_args.kwargs
            assert call_kwargs["json"]["model"] == "granite4.1:3b", (
                call_kwargs["json"]["model"],
                "granite4.1:3b",
            )

    async def test_unload_model(self):
        """Test model unloading."""
        mock_response = Mock()
        with patch.object(
            self.client._client,
            'post',
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_post:
            await self.client.unload_model("granite4.1:3b")

            mock_post.assert_called_once()
            call_kwargs = mock_post.call_args.kwargs
            assert call_kwargs["json"]["keep_alive"] == "0", (
                call_kwargs["json"]["keep_alive"],
                "0",
            )

    async def test_list_models(self):
        """Test listing available models."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "models": [
                {
                    "name": "granite4.1:3b:latest",
                    "modified_at": "2024-01-01T00:00:00Z",
                    "size": 4000000000,
                    "digest": "abc123",
                },
            ],
        }

        with patch.object(
            self.client._client,
            'get',
            new_callable=AsyncMock,
        ) as mock_get:
            mock_get.return_value = mock_response

            models = await self.client.list_models()

            assert len(models) == 1, len(models)
            assert models[0].name == "granite4.1:3b:latest", (
                models[0].name,
                "granite4.1:3b:latest",
            )

    async def test_list_running_models(self):
        """Test listing running models."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "models": [
                {
                    "name": "granite4.1:3b",
                    "expires_at": "2024-01-01T01:00:00Z",
                    "size": 4000000000,
                    "vm_peak": 8000000000,
                    "vram_total": 8000000000,
                },
            ],
        }

        with patch.object(
            self.client._client,
            'get',
            new_callable=AsyncMock,
        ) as mock_get:
            mock_get.return_value = mock_response

            models = await self.client.list_running_models()

            assert len(models) == 1, len(models)
            assert models[0].name == "granite4.1:3b", (
                models[0].name,
                "granite4.1:3b",
            )

    async def test_retry_logic(self):
        """Test retry logic with connection and timeout errors for both chat and generate."""
        for request_type, make_request in [
            ("chat", lambda: ChatRequest(
                model="granite4.1:3b",
                messages=[Message(role=Role.USER, content="test")],
            )),
            ("generate", lambda: GenerateRequest(
                model="granite4.1:3b",
                prompt="test",
            )),
        ]:
            request = make_request()
            call_count = [0]

            class MockStreamContext:
                async def __aenter__(self):
                    call_count[0] += 1
                    raise httpx.ConnectError("Connection refused")

                async def __aexit__(self, *args):
                    pass

            self.client._client.stream = Mock(return_value=MockStreamContext())

            with self.assertRaises(OllamaConnectionError):
                if request_type == "chat":
                    async for _ in self.client.chat(request):
                        pass
                else:
                    async for _ in self.client.generate(request):
                        pass

            assert call_count[0] == 3, (call_count[0], 3)

            client = OllamaClient()
            request = make_request()
            chunks = []
            if request_type == "chat":
                async for chunk in client.chat(request):
                    chunks.append(chunk)
            else:
                async for chunk in client.generate(request):
                    chunks.append(chunk)

            assert len(chunks) > 0, len(chunks)
            await client.close()

    async def test_http_status_error(self):
        """Test HTTP status error handling."""
        request = ChatRequest(
            model="granite4.1:3b",
            messages=[Message(role=Role.USER, content="test")],
        )

        class MockStreamContext:
            def __init__(self):
                self.status_code = 404

            async def __aenter__(self):
                raise httpx.HTTPStatusError(
                    "Not Found",
                    request=Mock(),
                    response=self,
                )

            async def __aexit__(self, *args):
                pass

        self.client._client.stream = Mock(return_value=MockStreamContext())

        with self.assertRaises(OllamaRequestError):
            async for _ in self.client.chat(request):
                pass

    async def test_model_error_load_model(self):
        """Test model error when loading model."""
        async def mock_post(*args, **kwargs):
            class MockResponse:
                status_code = 404

                def raise_for_status(self):
                    raise httpx.HTTPStatusError(
                        "Model not found",
                        request=Mock(),
                        response=self,
                    )

            return MockResponse()

        self.client._client.post = mock_post

        with self.assertRaises(OllamaModelError):
            await self.client.load_model("nonexistent")

    async def test_get_model_info_caching(self):
        """Test model info retrieval with caching."""
        info1 = await self.client.get_model_info("qwen3:0.6b")
        assert info1.context_length > 0, info1.context_length
        assert info1.architecture == "qwen3", info1.architecture
        assert info1.model_name == "qwen3:0.6b", info1.model_name

        info2 = await self.client.get_model_info("qwen3:0.6b")
        assert info1 == info2, (info1, info2)

    async def test_get_model_info_cache_clear(self):
        """Test cache clearing functionality across client instances."""
        client1 = OllamaClient()
        await client1.get_model_info("qwen3:0.6b")
        await client1.close()

        await model_info_cache.clear()

        client2 = OllamaClient()
        info = await client2.get_model_info("qwen3:0.6b")
        assert info.context_length > 0, info.context_length
        await client2.close()

    async def test_get_model_info_shared_cache(self):
        """Test that cache is shared between client instances."""
        client1 = OllamaClient()
        info1 = await client1.get_model_info("qwen3:0.6b")
        await client1.close()

        client2 = OllamaClient()
        info2 = await client2.get_model_info("qwen3:0.6b")
        await client2.close()

        assert info1 == info2, (info1, info2)
        assert len(model_info_cache._cache) > 0

    async def test_get_model_info_cache_override(self):
        """Test that custom cache instance is isolated."""
        custom_cache = AsyncLRUCache(maxsize=5)
        client = OllamaClient(cache=custom_cache)

        await client.get_model_info("qwen3:0.6b")

        assert len(custom_cache._cache) == 1
        assert len(model_info_cache._cache) == 0

        await client.close()

    async def test_get_model_info_error(self):
        """Test error handling for nonexistent model."""
        with self.assertRaises(OllamaModelError):
            await self.client.get_model_info("nonexistent-model-xyz123")

    async def test_context_manager(self):
        """Test async context manager."""
        async with OllamaClient() as client:
            assert client is not None, client

    async def test_e2e_model_management_workflow(self):
        """Test end-to-end model management (requires Ollama server)."""
        models = await self.client.list_models()
        assert len(models) > 0, models

        if models:
            model_name = models[0].name
            await self.client.load_model(model_name)

            running_models = await self.client.list_running_models()
            assert any(
                model_name in rm.name for rm in running_models
            ), running_models

            await self.client.unload_model(model_name)

    async def test_thinking_field_none_when_not_present(self):
        """Test that thinking field is None when not in response."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "model": "granite4.1:3b",
            "message": {
                "role": "assistant",
                "content": "Hello",
            },
            "done": True,
            "total_duration": 1000000000,
        }

        async def mock_post(*args, **kwargs):
            return mock_response

        client = OllamaClient()
        client._client.post = mock_post

        request = ChatRequest(
            model="granite4.1:3b",
            messages=[Message(role=Role.USER, content="test")],
        )

        def run_sync():
            return client.chat_sync(request)

        response = await asyncio.get_event_loop().run_in_executor(
            None,
            run_sync,
        )

        assert response.message.thinking is None, (
            response.message.thinking,
        )

        await client.close()


if __name__ == "__main__":
    unittest.main()
