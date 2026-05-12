"""
Tests for OllamaClient functionality.
"""

import asyncio
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, Mock, patch, MagicMock
import unittest
import httpx
from hola.ollama.client import OllamaClient
from hola.ollama.config import ClientConfig
from hola.ollama.exceptions import (
    OllamaConnectionError,
    OllamaRequestError,
    OllamaModelError,
)
from hola.ollama.types import (
    ChatRequest,
    ChatResponse,
    GenerateChunk,
    GenerateRequest,
    GenerateResponse,
    GenerationOptions,
    Message,
    ModelInfo,
    Role,
    RunningModel,
)


class TestClient(unittest.IsolatedAsyncioTestCase):
    """Test cases for OllamaClient."""

    def setUp(self):
        """Set up test fixtures."""
        self.client = OllamaClient()

    async def asyncTearDown(self):
        """Clean up after tests."""
        await self.client.close()

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

    async def test_chat_streaming(self):
        """Test streaming chat response."""
        request = ChatRequest(
            model="granite4.1:3b",
            messages=[Message(role=Role.USER, content="Say hello")],
            options=GenerationOptions(num_predict=10),
        )

        chunks = []
        async for chunk in self.client.chat(request):
            chunks.append(chunk)
            if chunk.done:
                break

        assert len(chunks) > 0, chunks
        assert any(chunk.message.content for chunk in chunks), chunks
        assert chunks[-1].done, chunks[-1].done

    async def test_chat_sync(self):
        """Test non-streaming chat response."""
        request = ChatRequest(
            model="granite4.1:3b",
            messages=[Message(role=Role.USER, content="test")],
        )

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "model": "granite4.1:3b",
            "message": {"role": "assistant", "content": "Hello world"},
            "done": True,
            "total_duration": 1000000000,
        }

        async def mock_post(*args, **kwargs):
            return mock_response

        client = OllamaClient()
        client._client.post = mock_post

        def run_chat_sync():
            return client.chat_sync(request)

        response = await asyncio.get_event_loop().run_in_executor(
            None,
            run_chat_sync,
        )

        assert response.model == "granite4.1:3b", (
            response.model,
            "granite4.1:3b",
        )
        assert response.message.content == "Hello world", (
            response.message.content,
            "Hello world",
        )

        await client.close()

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

    async def test_retry_logic_connection_error(self):
        """Test retry logic with connection errors."""
        request = ChatRequest(
            model="granite4.1:3b",
            messages=[Message(role=Role.USER, content="test")],
        )

        call_count = [0]

        class MockStreamContext:
            async def __aenter__(self):
                call_count[0] += 1
                raise httpx.ConnectError("Connection refused")

            async def __aexit__(self, *args):
                pass

        self.client._client.stream = Mock(return_value=MockStreamContext())

        with self.assertRaises(OllamaConnectionError):
            async for _ in self.client.chat(request):
                pass

        assert call_count[0] == 3, (call_count[0], 3)

    async def test_retry_logic_timeout_error(self):
        """Test retry logic with timeout errors."""
        request = ChatRequest(
            model="granite4.1:3b",
            messages=[Message(role=Role.USER, content="test")],
        )

        chunks = []
        async for chunk in self.client.chat(request):
            chunks.append(chunk)

        assert len(chunks) > 0, len(chunks)

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

    async def test_context_manager(self):
        """Test async context manager."""
        async with OllamaClient() as client:
            assert client is not None, client

    async def test_e2e_chat_workflow(self):
        """Test end-to-end chat workflow (requires Ollama server)."""
        request = ChatRequest(
            model="granite4.1:3b",
            messages=[Message(role=Role.USER, content="Say hello")],
            options=GenerationOptions(num_predict=10),
        )

        chunks = []
        async for chunk in self.client.chat(request):
            chunks.append(chunk)
            if chunk.done:
                break

        assert len(chunks) > 0, chunks
        assert any(chunk.message.content for chunk in chunks), chunks
        assert chunks[-1].done, chunks[-1].done

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

    async def test_generate_streaming(self):
        """Test streaming generate response."""
        request = GenerateRequest(
            model="granite4.1:3b",
            prompt="Say hello",
            options=GenerationOptions(num_predict=10),
        )

        chunks = []
        async for chunk in self.client.generate(request):
            chunks.append(chunk)
            if chunk.done:
                break

        assert len(chunks) > 0, chunks
        assert any(chunk.response for chunk in chunks), chunks
        assert chunks[-1].done, chunks[-1].done

    async def test_generate_sync(self):
        """Test non-streaming generate response."""
        request = GenerateRequest(
            model="granite4.1:3b",
            prompt="test",
        )

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "model": "granite4.1:3b",
            "response": "Hello world",
            "done": True,
            "total_duration": 1000000000,
        }

        async def mock_post(*args, **kwargs):
            return mock_response

        client = OllamaClient()
        client._client.post = mock_post

        def run_generate_sync():
            return client.generate_sync(request)

        response = await asyncio.get_event_loop().run_in_executor(
            None,
            run_generate_sync,
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

    async def test_generate_retry_logic_connection_error(self):
        """Test generate retry logic with connection errors."""
        request = GenerateRequest(
            model="granite4.1:3b",
            prompt="test",
        )

        call_count = [0]

        class MockStreamContext:
            async def __aenter__(self):
                call_count[0] += 1
                raise httpx.ConnectError("Connection refused")

            async def __aexit__(self, *args):
                pass

        self.client._client.stream = Mock(return_value=MockStreamContext())

        with self.assertRaises(OllamaConnectionError):
            async for _ in self.client.generate(request):
                pass

        assert call_count[0] == 3, (call_count[0], 3)

    async def test_e2e_generate_workflow(self):
        """Test end-to-end generate workflow (requires Ollama server)."""
        request = GenerateRequest(
            model="granite4.1:3b",
            prompt="Say hello",
            options=GenerationOptions(num_predict=10),
        )

        chunks = []
        async for chunk in self.client.generate(request):
            chunks.append(chunk)
            if chunk.done:
                break

        assert len(chunks) > 0, chunks
        assert any(chunk.response for chunk in chunks), chunks
        assert chunks[-1].done, chunks[-1].done


if __name__ == "__main__":
    unittest.main()
