"""
Tests for OllamaClient functionality.
"""

from collections.abc import AsyncGenerator
from hola.cache import AsyncLRUCache
from hola.ollama.client import OllamaClient, model_info_cache
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
import asyncio
import logging
import unittest
import httpx


class TestClient(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.client_logger = logging.getLogger('hola.ollama.client')
        cls.original_level = cls.client_logger.level
        cls.client_logger.setLevel(logging.CRITICAL)

    @classmethod
    def tearDownClass(cls):
        cls.client_logger.setLevel(cls.original_level)

        async def cleanup():
            client = OllamaClient()
            try:
                for model in ["qwen3:0.6b", "granite4.1:3b"]:
                    try:
                        await client.unload_model(model)
                    except Exception:
                        pass
            finally:
                await client.close()

        asyncio.run(cleanup())

    def setUp(self):
        self.client = OllamaClient()

    async def asyncTearDown(self):
        await self.client.close()
        await model_info_cache.clear()

    async def test_client_initialization(self):
        assert self.client._config.base_url == "http://localhost:11434", (
            self.client._config.base_url,
            "http://localhost:11434",
        )
        assert self.client._config.timeout == 120.0, (
            self.client._config.timeout,
            120.0,
        )

    async def test_client_initialization_custom_config(self):
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
        assert isinstance(self.client, LLMClient), (
            "OllamaClient should be compatible with LLMClient Protocol"
        )

        custom_client = OllamaClient(base_url="http://localhost:11435")
        assert isinstance(custom_client, LLMClient), (
            "Custom OllamaClient should be compatible with LLMClient Protocol"
        )
        await custom_client.close()

    async def test_streaming_operations(self):
        for request_type, request in [
                ("chat", ChatRequest(
                    model="qwen3:0.6b",
                    messages=[Message(role=Role.USER, content="Say hello")],
                )),
                ("generate", GenerateRequest(
                    model="qwen3:0.6b",
                    prompt="Say hello",
                )),
            ]:
            chunks = []
            accumulated_thinking = []
            accumulated_content = []

            async for chunk in await (
                    self.client.chat(request)
                    if request_type == "chat"
                    else self.client.generate(request)
                ):
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
                    "Expected thinking chunks but got none. Content: "
                    f"{''.join(accumulated_content)}"
                )
                assert len(complete_thinking) > 0, (
                    "Expected non-empty thinking but got empty. Content: "
                    f"{''.join(accumulated_content)}"
                )
            else:
                assert any(chunk.response for chunk in chunks), chunks
                complete_content = "".join(accumulated_content)
                assert len(complete_content) > 0, (
                    f"Expected non-empty response but got empty."
                )

            assert chunks[-1].done, chunks[-1].done

    async def test_list_models(self):
        models = await self.client.list_models()

        assert len(models), len(models)
        assert all([type(m) is ModelInfo for m in models]), models

    async def test_load_model_list_running_models_unload_model_e2e(self):
        await self.client.load_model("qwen3:0.6b")
        models = await self.client.list_running_models()
        assert len(models) > 0, len(models)
        assert all([type(m) is RunningModel for m in models]), models
        assert any([m.name == "qwen3:0.6b" for m in models]), models
        await self.client.unload_model("qwen3:0.6b")
        await asyncio.sleep(1)
        models = await self.client.list_running_models()
        assert not any([m.name == "qwen3:0.6b" for m in models]), models

    async def test_http_status_error(self):
        request = ChatRequest(
            model="nonexistentmodel",
            messages=[Message(role=Role.USER, content="test")],
        )

        with self.assertRaises(OllamaRequestError):
            async for _ in await self.client.chat(request):
                pass

    async def test_model_error_load_model(self):
        with self.assertRaises(httpx.HTTPStatusError):
            await self.client.load_model("nonexistent")

    async def test_get_model_info_caching(self):
        assert not await model_info_cache.get("qwen3:0.6b")
        info1 = await self.client.get_model_info("qwen3:0.6b")
        assert info1.context_length > 0, info1.context_length
        assert info1.architecture == "qwen3", info1.architecture
        assert info1.model_name == "qwen3:0.6b", info1.model_name
        assert await model_info_cache.get("qwen3:0.6b")

    async def test_get_model_info_cache_override(self):
        custom_cache = AsyncLRUCache(maxsize=5)
        client = OllamaClient(cache=custom_cache)

        await client.get_model_info("qwen3:0.6b")

        assert len(custom_cache._cache) == 1
        assert len(model_info_cache._cache) == 0

        await client.close()

    async def test_get_model_info_error(self):
        with self.assertRaises(httpx.HTTPStatusError):
            await self.client.get_model_info("nonexistent-model-xyz123")

    async def test_context_manager(self):
        async with OllamaClient() as client:
            assert client is not None, client

    async def test_e2e_model_management_workflow(self):
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
        client = OllamaClient()

        request = ChatRequest(
            model="granite4.1:3b",
            messages=[Message(role=Role.USER, content="test")],
            stream=False,
        )

        response = await client.chat(request)

        assert response.message.thinking is None, (
            response.message.thinking,
        )

        await client.close()

    async def test_chat_with_stream_false(self):
        request = ChatRequest(
            model="granite4.1:3b",
            messages=[Message(role=Role.USER, content="Say hello")],
            stream=False,
        )

        response = await self.client.chat(request)

        assert isinstance(response, ChatResponse), type(response)
        assert response.done is True, response.done
        assert response.message.content is not None, response.message.content

    async def test_generate_with_stream_false(self):
        request = GenerateRequest(
            model="granite4.1:3b",
            prompt="Say hello",
            stream=False,
        )

        response = await self.client.generate(request)

        assert isinstance(response, GenerateResponse), type(response)
        assert response.done is True, response.done
        assert response.response is not None, response.response


if __name__ == "__main__":
    unittest.main()
