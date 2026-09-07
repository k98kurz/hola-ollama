"""
Ollama HTTP client for API interactions.
"""

from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass
from typing import Self, TypeVar
from hola.cache import AsyncLRUCache
from hola.errors import type_assert, value_assert
from hola.ollama.config import ClientConfig
from hola.ollama.exceptions import (
    OllamaClientError,
    OllamaConnectionError,
    OllamaModelError,
    OllamaRequestError,
)
from hola.ollama.types import RunningModel
from hola.tools.types import ToolCall, ToolCallFunction
from hola.types import (
    ChatRequest,
    GenerateRequest,
    GenerationOptions,
    LLMClient,
    Message,
    Role,
    ChatResponse,
    ChatResponseChunk,
    GenerateChunk,
    GenerateResponse,
    ModelInfo,
    ModelMetadata,
)
import asyncio
import json
import logging
import httpx


logger = logging.getLogger(__name__)
model_info_cache: AsyncLRUCache = AsyncLRUCache(maxsize=20)
T = TypeVar("T")


class OllamaClient:
    """HTTP client for Ollama API interactions."""
    def __init__(
            self,
            base_url: str = "http://localhost:11434",
            timeout: float = 120.0,
            config: ClientConfig | None = None,
            cache: AsyncLRUCache | None = None,
        ) -> None:
        """Initialize client with httpx.AsyncClient."""
        if config is None:
            config = ClientConfig(base_url=base_url, timeout=timeout)

        self._config = config
        self._client = httpx.AsyncClient(
            base_url=config.base_url,
            timeout=httpx.Timeout(config.timeout),
        )
        self._model_info_cache = cache if cache is not None else model_info_cache

    async def _with_retries(
            self, func: Callable[[], Awaitable[T]], *,
            retries: int | None = None,
        ) -> T:
        """Run `func` with retries on connection errors.
            `httpx.ConnectError` and `httpx.TimeoutException` are
            retried up to `retries` (or `config.max_retries` when
            `retries` is `None`) times; other `httpx` errors are mapped
            to the corresponding `Ollama*` exception and raised
            immediately. `retries` must be `>= 0`. Passing
            `retries=0` disables retry/backoff while retaining
            exception mapping.
        """
        if retries is None:
            retries = self._config.max_retries
        type_assert(type(retries) is int, 'retries must be int')
        value_assert(retries >= 0, "retries must be >= 0")

        delay = self._config.retry_delay
        type_assert(type(delay) in (float, int), "config delay must be int|float")

        for attempt in range(retries + 1):
            try:
                return await func()
            except (httpx.ConnectError, httpx.TimeoutException) as e:
                if attempt == retries:
                    logger.error(
                        "Request failed after %d retries: %s",
                        retries,
                        e,
                    )
                    raise OllamaConnectionError(
                        f"Request failed after {retries} retries: {e}"
                    ) from e

                logger.debug(
                    "Connection error, retrying in %.1fs (retry %d/%d): %s",
                    delay,
                    attempt + 1,
                    retries,
                    e,
                )
                await asyncio.sleep(delay)
                delay *= 2

            except httpx.HTTPStatusError as e:
                logger.error("HTTP status error %s: %s", e.response.status_code, e)
                raise OllamaRequestError(
                    f"HTTP error {e.response.status_code}: {e}"
                ) from e

            except httpx.HTTPError as e:
                logger.error("HTTP error: %s", e)
                raise OllamaClientError(f"HTTP error: {e}") from e

    async def _post(self, url: str, payload: dict) -> httpx.Response:
        """POST `payload` to `url` and raise for status errors."""
        response = await self._client.post(url, json=payload)
        response.raise_for_status()
        return response

    async def _get(self, url: str) -> httpx.Response:
        """GET `url` and raise for status errors."""
        response = await self._client.get(url)
        response.raise_for_status()
        return response

    async def _send_stream(self, url: str, payload: dict) -> httpx.Response:
        """Open a streaming POST to `url`. The caller must `aclose()`
            the returned response once finished.
        """
        request = self._client.build_request("POST", url, json=payload)
        response = await self._client.send(request, stream=True)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError:
            await response.aclose()
            raise
        return response

    async def chat(
            self, request: ChatRequest
        ) -> AsyncGenerator[ChatResponseChunk, None] | ChatResponse:
        """Supports both streaming and non-streaming."""
        if request.stream:
            return self._chat_stream(request)
        else:
            return await self._chat_nonstream(request)

    async def _chat_stream(
            self, request: ChatRequest
        ) -> AsyncGenerator[ChatResponseChunk, None]:
        logger.debug("Starting chat request for model %s", request.model)
        payload = self._serialize_chat_request(request)
        response = await self._with_retries(
            lambda: self._send_stream("/api/chat", payload)
        )

        try:
            async for chunk in self._parse_sse_stream(response):
                yield chunk
        except httpx.HTTPError as e:
            raise OllamaConnectionError(f"Stream failed: {e}") from e
        finally:
            await response.aclose()

    async def _chat_nonstream(self, request: ChatRequest) -> ChatResponse:
        request_dict = self._serialize_chat_request(request)
        request_dict["stream"] = False
        response = await self._with_retries(
            lambda: self._post("/api/chat", request_dict)
        )
        data = response.json()

        return ChatResponse(
                model=data["model"],
                message=Message(
                    role=Role(data["message"]["role"]),
                    content=data["message"].get("content"),
                    tool_calls=(
                        self._parse_tool_calls(data["message"].get("tool_calls"))
                        if "tool_calls" in data["message"]
                        else None
                    ),
                    thinking=data["message"].get("thinking"),
                ),
                done=data["done"],
                total_duration=data.get("total_duration"),
                load_duration=data.get("load_duration"),
                prompt_tokens=data.get("prompt_eval_count"),
                prompt_duration=data.get("prompt_eval_duration"),
                output_tokens=data.get("eval_count"),
                output_duration=data.get("eval_duration"),
            )

    async def generate(
            self, request: GenerateRequest
        ) -> AsyncGenerator[GenerateChunk, None] | GenerateResponse:
        """Generate text using /api/generate endpoint."""
        if request.stream:
            return self._generate_stream(request)
        else:
            return await self._generate_nonstream(request)

    async def _generate_stream(
            self, request: GenerateRequest
        ) -> AsyncGenerator[GenerateChunk, None]:
        logger.debug("Starting generate request for model %s", request.model)
        payload = self._serialize_generate_request(request)
        response = await self._with_retries(
            lambda: self._send_stream("/api/generate", payload)
        )

        try:
            async for chunk in self._parse_generate_stream(response):
                yield chunk
        except httpx.HTTPError as e:
            raise OllamaConnectionError(f"Stream failed: {e}") from e
        finally:
            await response.aclose()

    async def _generate_nonstream(
            self, request: GenerateRequest
        ) -> GenerateResponse:
        request_dict = self._serialize_generate_request(request)
        request_dict["stream"] = False
        response = await self._with_retries(
            lambda: self._post("/api/generate", request_dict)
        )
        data = response.json()

        return GenerateResponse(
                model=data["model"],
                response=data["response"],
                done=data["done"],
                total_duration=data.get("total_duration"),
                load_duration=data.get("load_duration"),
                prompt_tokens=data.get("prompt_eval_count"),
                prompt_duration=data.get("prompt_eval_duration"),
                output_tokens=data.get("eval_count"),
                output_duration=data.get("eval_duration"),
            )

    async def load_model(self, model: str, keep_alive: str = "24h") -> None:
        """Explicit model load."""
        logger.debug("Loading model %s (keep_alive: %s)", model, keep_alive)
        request_dict = {"model": model, "messages": [], "keep_alive": keep_alive}

        await self._with_retries(
            lambda: self._post("/api/chat", request_dict),
            retries=1,
        )
        logger.info("Model %s loaded successfully", model)

    async def unload_model(self, model: str) -> None:
        """Explicit model unload."""
        logger.debug("Unloading model %s", model)
        request_dict = {"model": model, "messages": [], "keep_alive": "0"}
        await self._with_retries(
            lambda: self._post("/api/chat", request_dict),
            retries=1,
        )
        logger.info("Model %s unloaded successfully", model)

    async def list_models(self) -> list[ModelInfo]:
        """List available models."""
        logger.debug("Listing available models")
        response = await self._with_retries(lambda: self._get("/api/tags"))
        data = response.json()

        models = [
            ModelInfo(
                name=model["name"],
                modified_at=model["modified_at"],
                size=model["size"],
                digest=model["digest"],
            )
            for model in data["models"]
        ]
        logger.info("Listed %d available models", len(models))
        return models

    async def list_running_models(self) -> list[RunningModel]:
        """List currently loaded models."""
        logger.debug("Listing running models")
        response = await self._with_retries(lambda: self._get("/api/ps"))
        data = response.json()

        models = [
            RunningModel(
                name=model["name"],
                expires_at=model["expires_at"],
                size=model["size"],
                vm_peak=model.get("vm_peak"),
                vram_total=model.get("vram_total"),
            )
            for model in data["models"]
        ]
        logger.info("Listed %d running models", len(models))
        return models

    async def get_model_info(self, model: str) -> ModelMetadata:
        """Get metadata about a specific model with caching."""
        cached = await self._model_info_cache.get(model)
        if cached is not None:
            return cached

        response = await self._with_retries(
            lambda: self._post("/api/show", {"name": model}),
        )
        data = response.json()

        model_info = data.get("model_info", {})
        details = data.get("details", {})

        context_length = None
        for key, value in model_info.items():
            if key.endswith(".context_length"):
                context_length = value
                break

        if context_length is None:
            raise OllamaModelError(
                f"Could not determine context length for model: {model}"
            )

        metadata = ModelMetadata(
            model_name=model,
            architecture=model_info.get("general.architecture", "unknown"),
            context_length=context_length,
            parameter_count=model_info.get("general.parameter_count"),
            quantization_level=details.get("quantization_level"),
            family=details.get("family"),
        )

        await self._model_info_cache.set(model, metadata)
        return metadata

    async def close(self) -> None:
        """Cleanup httpx client."""
        await self._client.aclose()

    async def __aenter__(self) -> Self:
        """Async context manager entry."""
        return self

    async def __aexit__(
            self, exc_type: type[BaseException] | None,
            exc_val: BaseException | None, exc_tb
        ) -> None:
        """Async context manager exit."""
        await self._client.aclose()

    def _serialize_chat_request(self, request: ChatRequest) -> dict:
        """Convert ChatRequest to JSON-serializable dict."""
        result = {
            "model": request.model,
            "messages": [self._serialize_message(msg) for msg in request.messages],
            "stream": request.stream,
        }

        if request.tools is not None:
            result["tools"] = [self._serialize_tool(tool) for tool in request.tools]

        if request.options is not None:
            result["options"] = self._serialize_generation_options(request.options)

        if request.keep_alive is not None:
            result["keep_alive"] = request.keep_alive

        return result

    def _serialize_message(self, message: Message) -> dict:
        """Convert Message to dict."""
        result = {
            "role": message.role.value,
            "content": message.content,
        }

        if message.tool_calls is not None:
            result["tool_calls"] = [
                self._serialize_tool_call(tc) for tc in message.tool_calls
            ]

        return result

    def _serialize_tool(self, tool) -> dict:
        """Serialize ToolDefinition to Ollama API format."""
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }

    def _serialize_tool_call(self, tool_call) -> dict:
        """Serialize ToolCall to dict."""
        return {
            "function": {
                "name": tool_call.function.name,
                "arguments": tool_call.function.arguments,
            },
        }

    def _serialize_generation_options(self, options) -> dict:
        """Convert GenerationOptions to dict."""
        result = {}

        if options.temperature is not None:
            result["temperature"] = options.temperature

        if options.top_p is not None:
            result["top_p"] = options.top_p

        if options.top_k is not None:
            result["top_k"] = options.top_k

        if options.num_predict is not None:
            result["num_predict"] = options.num_predict

        if options.seed is not None:
            result["seed"] = options.seed

        return result

    async def _parse_sse_stream(
            self, response,
        ) -> AsyncGenerator[ChatResponseChunk, None]:
        """Parse SSE streaming response."""
        async for line in response.aiter_lines():
            line = line.strip()

            if not line:
                continue

            if line.startswith("data: "):
                data = line[6:]
            else:
                data = line

            try:
                chunk_data = json.loads(data)
            except json.JSONDecodeError:
                continue

            yield self._parse_chat_chunk(chunk_data)

    def _parse_chat_chunk(self, data: dict) -> ChatResponseChunk:
        """Parse JSON chunk to ChatResponseChunk."""
        message_data = data.get("message", {})

        tool_calls = None
        if "tool_calls" in message_data:
            tool_calls = self._parse_tool_calls(message_data["tool_calls"])

        return ChatResponseChunk(
            model=data["model"],
            message=Message(
                role=Role(message_data.get("role", "assistant")),
                content=message_data.get("content"),
                tool_calls=tool_calls,
                thinking=message_data.get("thinking"),
            ),
            done=data["done"],
            done_reason=data.get("done_reason"),
            total_duration=data.get("total_duration"),
            load_duration=data.get("load_duration"),
            prompt_tokens=data.get("prompt_eval_count"),
            prompt_duration=data.get("prompt_eval_duration"),
            output_tokens=data.get("eval_count"),
            output_duration=data.get("eval_duration"),
        )

    def _parse_tool_calls(
            self, tool_calls_data: list[dict] | None
        ) -> list[ToolCall] | None:
        """Parse tool calls from response data."""
        if tool_calls_data is None:
            return None

        return [
            ToolCall(
                function=ToolCallFunction(
                    name=tc["function"]["name"],
                    arguments=tc["function"]["arguments"],
                ),
            )
            for tc in tool_calls_data
        ]

    def _serialize_generate_request(self, request: GenerateRequest) -> dict:
        """Convert GenerateRequest to JSON-serializable dict."""
        result = {
            "model": request.model,
            "prompt": request.prompt,
            "stream": request.stream,
        }

        if request.system is not None:
            result["system"] = request.system

        if request.options is not None:
            result["options"] = self._serialize_generation_options(request.options)

        if request.keep_alive is not None:
            result["keep_alive"] = request.keep_alive

        return result

    async def _parse_generate_stream(
            self, response,
        ) -> AsyncGenerator[GenerateChunk, None]:
        """Parse SSE streaming response for generate endpoint."""
        async for line in response.aiter_lines():
            line = line.strip()

            if not line:
                continue

            if line.startswith("data: "):
                data = line[6:]
            else:
                data = line

            try:
                chunk_data = json.loads(data)
            except json.JSONDecodeError:
                continue

            yield self._parse_generate_chunk(chunk_data)

    def _parse_generate_chunk(self, data: dict) -> GenerateChunk:
        """Parse JSON chunk to GenerateChunk."""
        return GenerateChunk(
            model=data["model"],
            response=data.get("response", ""),
            done=data["done"],
            done_reason=data.get("done_reason"),
            total_duration=data.get("total_duration"),
            load_duration=data.get("load_duration"),
            prompt_tokens=data.get("prompt_eval_count"),
            prompt_duration=data.get("prompt_eval_duration"),
            output_tokens=data.get("eval_count"),
            output_duration=data.get("eval_duration"),
        )
