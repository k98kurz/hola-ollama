"""
Ollama HTTP client for API interactions.
"""

from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Self
import asyncio
from hola.ollama.config import ClientConfig
from hola.ollama.exceptions import (
    OllamaClientError,
    OllamaConnectionError,
    OllamaModelError,
    OllamaRequestError,
)
from hola.ollama.types import (
    ChatRequest,
    ChatResponse,
    ChatResponseChunk,
    GenerateChunk,
    GenerateRequest,
    GenerateResponse,
    Message,
    ModelInfo,
    Role,
    RunningModel,
    ToolCall,
    ToolCallFunction,
)
import json
import logging
import httpx


logger = logging.getLogger(__name__)


@dataclass
class _ChatResponseStats:
    """Statistics from final chat response chunk."""
    total_duration: int | None = None
    load_duration: int | None = None
    prompt_eval_count: int | None = None
    prompt_eval_duration: int | None = None
    eval_count: int | None = None
    eval_duration: int | None = None


class OllamaClient:
    """HTTP client for Ollama API interactions."""

    def __init__(
            self,
            base_url: str = "http://localhost:11434",
            timeout: float = 120.0,
            config: ClientConfig | None = None,
        ) -> None:
        """Initialize client with httpx.AsyncClient."""
        if config is None:
            config = ClientConfig(base_url=base_url, timeout=timeout)

        self._config = config
        self._client = httpx.AsyncClient(
            base_url=config.base_url,
            timeout=httpx.Timeout(config.timeout),
        )

    async def chat(
            self,
            request: ChatRequest,
        ) -> AsyncGenerator[ChatResponseChunk, None]:
        """Primary streaming API - yields response chunks."""
        logger.debug("Starting chat request for model %s", request.model)

        for attempt in range(self._config.max_retries):
            try:
                async with self._client.stream(
                        "POST",
                        "/api/chat",
                        json=self._serialize_chat_request(request),
                    ) as response:
                    response.raise_for_status()

                    async for chunk in self._parse_sse_stream(response):
                        yield chunk

                    return

            except httpx.ConnectError as e:
                if attempt == self._config.max_retries - 1:
                    logger.error(
                        "Connection failed after %d attempts for model %s",
                        self._config.max_retries,
                        request.model,
                    )
                    raise OllamaConnectionError(
                        f"Failed to connect to Ollama after "
                        f"{self._config.max_retries} attempts"
                    ) from e

                delay = self._config.retry_delay * (2 ** attempt)
                logger.debug(
                    "Connection error, retrying in %.1fs (attempt %d/%d): %s",
                    delay,
                    attempt + 1,
                    self._config.max_retries,
                    e,
                )
                await asyncio.sleep(delay)

            except httpx.TimeoutException as e:
                if attempt == self._config.max_retries - 1:
                    logger.error(
                        "Request timed out after %d attempts for model %s",
                        self._config.max_retries,
                        request.model,
                    )
                    raise OllamaConnectionError(
                        f"Timeout after {self._config.max_retries} attempts"
                    ) from e

                delay = self._config.retry_delay * (2 ** attempt)
                logger.debug(
                    "Timeout, retrying in %.1fs (attempt %d/%d): %s",
                    delay,
                    attempt + 1,
                    self._config.max_retries,
                    e,
                )
                await asyncio.sleep(delay)

            except httpx.HTTPStatusError as e:
                logger.error(
                    "HTTP error %s for model %s: %s",
                    e.response.status_code,
                    request.model,
                    e,
                )
                raise OllamaRequestError(
                    f"HTTP error {e.response.status_code}: {e}"
                ) from e

            except httpx.HTTPError as e:
                logger.error(
                    "HTTP error for model %s: %s",
                    request.model,
                    e,
                )
                raise OllamaClientError(
                    f"HTTP error: {e}"
                ) from e

    def chat_sync(
            self,
            request: ChatRequest,
        ) -> ChatResponse:
        """Non-streaming convenience wrapper."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None:
            raise RuntimeError(
                "chat_sync cannot be called from within an "
                "async context. Use the async chat() method instead."
            )

        return asyncio.run(self._chat_sync_async(request))

    async def _chat_sync_async(
            self,
            request: ChatRequest,
        ) -> ChatResponse:
        """Async implementation of non-streaming chat."""
        logger.debug("Starting non-streaming chat for model %s", request.model)
        request_dict = self._serialize_chat_request(request)
        request_dict["stream"] = False

        for attempt in range(self._config.max_retries):
            try:
                response = await self._client.post(
                    "/api/chat",
                    json=request_dict,
                )
                response.raise_for_status()
                data = response.json()

                return ChatResponse(
                    model=data["model"],
                    message=Message(
                        role=data["message"]["role"],
                        content=data["message"].get("content"),
                        tool_calls=(
                            self._parse_tool_calls(data["message"].get("tool_calls"))
                            if "tool_calls" in data["message"]
                            else None
                        ),
                        tool_name=data["message"].get("tool_name"),
                    ),
                    done=data["done"],
                    total_duration=data.get("total_duration"),
                    load_duration=data.get("load_duration"),
                    prompt_eval_count=data.get("prompt_eval_count"),
                    prompt_eval_duration=data.get("prompt_eval_duration"),
                    eval_count=data.get("eval_count"),
                    eval_duration=data.get("eval_duration"),
                )

            except (httpx.ConnectError, httpx.TimeoutException) as e:
                if attempt == self._config.max_retries - 1:
                    logger.error(
                        "Connection failed after %d attempts for model %s",
                        self._config.max_retries,
                        request.model,
                    )
                    raise OllamaConnectionError(
                        f"Failed to connect to Ollama after "
                        f"{self._config.max_retries} attempts"
                    ) from e

                delay = self._config.retry_delay * (2 ** attempt)
                logger.debug(
                    "Connection error, retrying in %.1fs (attempt %d/%d): %s",
                    delay,
                    attempt + 1,
                    self._config.max_retries,
                    e,
                )
                await asyncio.sleep(delay)

            except httpx.HTTPStatusError as e:
                logger.error(
                    "HTTP error %s for model %s: %s",
                    e.response.status_code,
                    request.model,
                    e,
                )
                raise OllamaRequestError(
                    f"HTTP error {e.response.status_code}: {e}"
                ) from e

            except httpx.HTTPError as e:
                logger.error(
                    "HTTP error for model %s: %s",
                    request.model,
                    e,
                )
                raise OllamaClientError(
                    f"HTTP error: {e}"
                ) from e

    async def generate(
            self,
            request: GenerateRequest,
        ) -> AsyncGenerator[GenerateChunk, None]:
        """Generate text using /api/generate endpoint for non-tool-calling models."""
        logger.debug("Starting generate request for model %s", request.model)

        for attempt in range(self._config.max_retries):
            try:
                async with self._client.stream(
                        "POST",
                        "/api/generate",
                        json=self._serialize_generate_request(request),
                    ) as response:
                    response.raise_for_status()

                    async for chunk in self._parse_generate_stream(response):
                        yield chunk

                    return

            except httpx.ConnectError as e:
                if attempt == self._config.max_retries - 1:
                    logger.error(
                        "Connection failed after %d attempts for model %s",
                        self._config.max_retries,
                        request.model,
                    )
                    raise OllamaConnectionError(
                        f"Failed to connect to Ollama after "
                        f"{self._config.max_retries} attempts"
                    ) from e

                delay = self._config.retry_delay * (2 ** attempt)
                logger.debug(
                    "Connection error, retrying in %.1fs (attempt %d/%d): %s",
                    delay,
                    attempt + 1,
                    self._config.max_retries,
                    e,
                )
                await asyncio.sleep(delay)

            except httpx.TimeoutException as e:
                if attempt == self._config.max_retries - 1:
                    logger.error(
                        "Request timed out after %d attempts for model %s",
                        self._config.max_retries,
                        request.model,
                    )
                    raise OllamaConnectionError(
                        f"Timeout after {self._config.max_retries} attempts"
                    ) from e

                delay = self._config.retry_delay * (2 ** attempt)
                logger.debug(
                    "Timeout, retrying in %.1fs (attempt %d/%d): %s",
                    delay,
                    attempt + 1,
                    self._config.max_retries,
                    e,
                )
                await asyncio.sleep(delay)

            except httpx.HTTPStatusError as e:
                logger.error(
                    "HTTP error %s for model %s: %s",
                    e.response.status_code,
                    request.model,
                    e,
                )
                raise OllamaRequestError(
                    f"HTTP error {e.response.status_code}: {e}"
                ) from e

            except httpx.HTTPError as e:
                logger.error(
                    "HTTP error for model %s: %s",
                    request.model,
                    e,
                )
                raise OllamaClientError(
                    f"HTTP error: {e}"
                ) from e

    def generate_sync(
            self,
            request: GenerateRequest,
        ) -> GenerateResponse:
        """Non-streaming generate convenience wrapper."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None:
            raise RuntimeError(
                "generate_sync cannot be called from within an "
                "async context. Use the async generate() method instead."
            )

        return asyncio.run(self._generate_sync_async(request))

    async def _generate_sync_async(
            self,
            request: GenerateRequest,
        ) -> GenerateResponse:
        """Async implementation of non-streaming generate."""
        logger.debug("Starting non-streaming generate for model %s", request.model)
        request_dict = self._serialize_generate_request(request)
        request_dict["stream"] = False

        for attempt in range(self._config.max_retries):
            try:
                response = await self._client.post(
                    "/api/generate",
                    json=request_dict,
                )
                response.raise_for_status()
                data = response.json()

                return GenerateResponse(
                    model=data["model"],
                    response=data["response"],
                    done=data["done"],
                    total_duration=data.get("total_duration"),
                    load_duration=data.get("load_duration"),
                    prompt_eval_count=data.get("prompt_eval_count"),
                    prompt_eval_duration=data.get("prompt_eval_duration"),
                    eval_count=data.get("eval_count"),
                    eval_duration=data.get("eval_duration"),
                )

            except (httpx.ConnectError, httpx.TimeoutException) as e:
                if attempt == self._config.max_retries - 1:
                    logger.error(
                        "Connection failed after %d attempts for model %s",
                        self._config.max_retries,
                        request.model,
                    )
                    raise OllamaConnectionError(
                        f"Failed to connect to Ollama after "
                        f"{self._config.max_retries} attempts"
                    ) from e

                delay = self._config.retry_delay * (2 ** attempt)
                logger.debug(
                    "Connection error, retrying in %.1fs (attempt %d/%d): %s",
                    delay,
                    attempt + 1,
                    self._config.max_retries,
                    e,
                )
                await asyncio.sleep(delay)

            except httpx.HTTPStatusError as e:
                logger.error(
                    "HTTP error %s for model %s: %s",
                    e.response.status_code,
                    request.model,
                    e,
                )
                raise OllamaRequestError(
                    f"HTTP error {e.response.status_code}: {e}"
                ) from e

            except httpx.HTTPError as e:
                logger.error(
                    "HTTP error for model %s: %s",
                    request.model,
                    e,
                )
                raise OllamaClientError(
                    f"HTTP error: {e}"
                ) from e

    async def load_model(
            self,
            model: str,
            keep_alive: str = "24h",
        ) -> None:
        """Explicit model load."""
        logger.debug("Loading model %s (keep_alive: %s)", model, keep_alive)
        request_dict = {"model": model, "messages": [], "keep_alive": keep_alive}

        for attempt in range(self._config.max_retries):
            try:
                response = await self._client.post(
                    "/api/chat",
                    json=request_dict,
                )
                response.raise_for_status()
                logger.info("Model %s loaded successfully", model)
                return

            except (httpx.ConnectError, httpx.TimeoutException) as e:
                if attempt == self._config.max_retries - 1:
                    logger.error(
                        "Failed to load model %s after %d attempts",
                        model,
                        self._config.max_retries,
                    )
                    raise OllamaConnectionError(
                        f"Failed to connect to Ollama after "
                        f"{self._config.max_retries} attempts"
                    ) from e

                delay = self._config.retry_delay * (2 ** attempt)
                logger.debug(
                    "Connection error, retrying in %.1fs (attempt %d/%d): %s",
                    delay,
                    attempt + 1,
                    self._config.max_retries,
                    e,
                )
                await asyncio.sleep(delay)

            except httpx.HTTPStatusError as e:
                logger.error(
                    "Failed to load model %s: HTTP error %s: %s",
                    model,
                    e.response.status_code,
                    e,
                )
                raise OllamaModelError(
                    f"Failed to load model {model}: HTTP error "
                    f"{e.response.status_code}"
                ) from e

            except httpx.HTTPError as e:
                logger.error(
                    "Failed to load model %s: %s",
                    model,
                    e,
                )
                raise OllamaModelError(
                    f"Failed to load model {model}: {e}"
                ) from e

    async def unload_model(
            self,
            model: str,
        ) -> None:
        """Explicit model unload."""
        logger.debug("Unloading model %s", model)
        request_dict = {"model": model, "messages": [], "keep_alive": "0"}

        for attempt in range(self._config.max_retries):
            try:
                response = await self._client.post(
                    "/api/chat",
                    json=request_dict,
                )
                response.raise_for_status()
                logger.info("Model %s unloaded successfully", model)
                return

            except (httpx.ConnectError, httpx.TimeoutException) as e:
                if attempt == self._config.max_retries - 1:
                    logger.error(
                        "Failed to unload model %s after %d attempts",
                        model,
                        self._config.max_retries,
                    )
                    raise OllamaConnectionError(
                        f"Failed to connect to Ollama after "
                        f"{self._config.max_retries} attempts"
                    ) from e

                delay = self._config.retry_delay * (2 ** attempt)
                logger.debug(
                    "Connection error, retrying in %.1fs (attempt %d/%d): %s",
                    delay,
                    attempt + 1,
                    self._config.max_retries,
                    e,
                )
                await asyncio.sleep(delay)

            except httpx.HTTPStatusError as e:
                logger.error(
                    "Failed to unload model %s: HTTP error %s: %s",
                    model,
                    e.response.status_code,
                    e,
                )
                raise OllamaModelError(
                    f"Failed to unload model {model}: HTTP error "
                    f"{e.response.status_code}"
                ) from e

            except httpx.HTTPError as e:
                logger.error(
                    "Failed to unload model %s: %s",
                    model,
                    e,
                )
                raise OllamaModelError(
                    f"Failed to unload model {model}: {e}"
                ) from e

    async def list_models(
            self,
        ) -> list[ModelInfo]:
        """List available models."""
        logger.debug("Listing available models")
        for attempt in range(self._config.max_retries):
            try:
                response = await self._client.get("/api/tags")
                response.raise_for_status()
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

            except (httpx.ConnectError, httpx.TimeoutException) as e:
                if attempt == self._config.max_retries - 1:
                    logger.error(
                        "Failed to list models after %d attempts",
                        self._config.max_retries,
                    )
                    raise OllamaConnectionError(
                        f"Failed to connect to Ollama after "
                        f"{self._config.max_retries} attempts"
                    ) from e

                delay = self._config.retry_delay * (2 ** attempt)
                logger.debug(
                    "Connection error, retrying in %.1fs (attempt %d/%d): %s",
                    delay,
                    attempt + 1,
                    self._config.max_retries,
                    e,
                )
                await asyncio.sleep(delay)

            except httpx.HTTPStatusError as e:
                logger.error(
                    "Failed to list models: HTTP error %s: %s",
                    e.response.status_code,
                    e,
                )
                raise OllamaRequestError(
                    f"Failed to list models: HTTP error {e.response.status_code}"
                ) from e

            except httpx.HTTPError as e:
                logger.error(
                    "Failed to list models: %s",
                    e,
                )
                raise OllamaClientError(
                    f"Failed to list models: {e}"
                ) from e

    async def list_running_models(
            self,
        ) -> list[RunningModel]:
        """List currently loaded models."""
        logger.debug("Listing running models")
        for attempt in range(self._config.max_retries):
            try:
                response = await self._client.get("/api/ps")
                response.raise_for_status()
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

            except (httpx.ConnectError, httpx.TimeoutException) as e:
                if attempt == self._config.max_retries - 1:
                    logger.error(
                        "Failed to list running models after %d attempts",
                        self._config.max_retries,
                    )
                    raise OllamaConnectionError(
                        f"Failed to connect to Ollama after "
                        f"{self._config.max_retries} attempts"
                    ) from e

                delay = self._config.retry_delay * (2 ** attempt)
                logger.debug(
                    "Connection error, retrying in %.1fs (attempt %d/%d): %s",
                    delay,
                    attempt + 1,
                    self._config.max_retries,
                    e,
                )
                await asyncio.sleep(delay)

            except httpx.HTTPStatusError as e:
                logger.error(
                    "Failed to list running models: HTTP error %s: %s",
                    e.response.status_code,
                    e,
                )
                raise OllamaRequestError(
                    f"Failed to list running models: HTTP error "
                    f"{e.response.status_code}"
                ) from e

            except httpx.HTTPError as e:
                logger.error(
                    "Failed to list running models: %s",
                    e,
                )
                raise OllamaClientError(
                    f"Failed to list running models: {e}"
                ) from e

    async def close(self) -> None:
        """Cleanup httpx client."""
        await self._client.aclose()

    async def __aenter__(self) -> Self:
        """Async context manager entry."""
        return self

    async def __aexit__(
            self,
            exc_type: type[BaseException] | None,
            exc_val: BaseException | None,
            exc_tb,
        ) -> None:
        """Async context manager exit."""
        await self._client.aclose()

    def _serialize_chat_request(
            self,
            request: ChatRequest,
        ) -> dict:
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

    def _serialize_message(
            self,
            message: Message,
        ) -> dict:
        """Convert Message to dict."""
        result = {
            "role": message.role.value,
            "content": message.content,
        }

        if message.tool_calls is not None:
            result["tool_calls"] = [
                self._serialize_tool_call(tc) for tc in message.tool_calls
            ]

        if message.tool_name is not None:
            result["tool_name"] = message.tool_name

        return result

    def _serialize_tool(
            self,
            tool,
        ) -> dict:
        """Serialize ToolDefinition to dict."""
        return {
            "type": tool.type,
            "function": {
                "name": tool.function.name,
                "description": tool.function.description,
                "parameters": tool.function.parameters,
            },
        }

    def _serialize_tool_call(
            self,
            tool_call,
        ) -> dict:
        """Serialize ToolCall to dict."""
        return {
            "function": {
                "name": tool_call.function.name,
                "arguments": tool_call.function.arguments,
            },
        }

    def _serialize_generation_options(
            self,
            options,
        ) -> dict:
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
            self,
            response,
        ) -> ChatResponseChunk:
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

    def _parse_chat_chunk(
            self,
            data: dict,
        ) -> ChatResponseChunk:
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
                tool_name=message_data.get("tool_name"),
            ),
            done=data["done"],
            done_reason=data.get("done_reason"),
            total_duration=data.get("total_duration"),
            load_duration=data.get("load_duration"),
            prompt_eval_count=data.get("prompt_eval_count"),
            prompt_eval_duration=data.get("prompt_eval_duration"),
            eval_count=data.get("eval_count"),
            eval_duration=data.get("eval_duration"),
        )

    def _parse_tool_calls(
            self,
            tool_calls_data: list[dict] | None,
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

    def _serialize_generate_request(
            self,
            request: GenerateRequest,
        ) -> dict:
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
            self,
            response,
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

    def _parse_generate_chunk(
            self,
            data: dict,
        ) -> GenerateChunk:
        """Parse JSON chunk to GenerateChunk."""
        return GenerateChunk(
            model=data["model"],
            response=data.get("response", ""),
            done=data["done"],
            done_reason=data.get("done_reason"),
            total_duration=data.get("total_duration"),
            load_duration=data.get("load_duration"),
            prompt_eval_count=data.get("prompt_eval_count"),
            prompt_eval_duration=data.get("prompt_eval_duration"),
            eval_count=data.get("eval_count"),
            eval_duration=data.get("eval_duration"),
        )
