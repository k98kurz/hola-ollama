"""
Ollama HTTP client for API interactions.
"""

from collections.abc import AsyncGenerator
from hola.cache import AsyncLRUCache
from hola.client import BaseClient, ClientConfig
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
    ChatResponse,
    ChatResponseChunk,
    GenerateChunk,
    GenerateRequest,
    GenerateResponse,
    GenerationOptions,
    Message,
    ModelInfo,
    ModelMetadata,
    Role,
)
import logging
import httpx


logger = logging.getLogger(__name__)
model_info_cache: AsyncLRUCache = AsyncLRUCache(maxsize=20)


class OllamaClient(BaseClient):
    """HTTP client for Ollama API interactions."""
    chat_endpoint = "/api/chat"
    generate_endpoint = "/api/generate"
    connection_error = OllamaConnectionError
    request_error = OllamaRequestError
    client_error = OllamaClientError

    def __init__(
            self,
            base_url: str = "http://localhost:11434",
            timeout: float = 120.0,
            config: ClientConfig | None = None,
            cache: AsyncLRUCache | None = None, *,
            chat_endpoint: str | None = None,
            generate_endpoint: str | None = None,
        ) -> None:
        """Initialize the Ollama client."""
        if config is None:
            config = ClientConfig(base_url=base_url, timeout=timeout)
        super().__init__(
            config,
            chat_endpoint=chat_endpoint,
            generate_endpoint=generate_endpoint,
        )
        self._model_info_cache = (
            cache if cache is not None else model_info_cache
        )

    async def load_model(self, model: str, keep_alive: str = "24h") -> None:
        """Explicit model load."""
        logger.debug("Loading model %s (keep_alive: %s)", model, keep_alive)
        request_dict = {
            "model": model,
            "messages": [],
            "keep_alive": keep_alive,
        }

        await self._with_retries(
            lambda: self._post(self.chat_endpoint, request_dict),
            retries=1,
        )
        logger.info("Model %s loaded successfully", model)

    async def unload_model(self, model: str) -> None:
        """Explicit model unload."""
        logger.debug("Unloading model %s", model)
        request_dict = {"model": model, "messages": [], "keep_alive": "0"}
        await self._with_retries(
            lambda: self._post(self.chat_endpoint, request_dict),
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

    def _serialize_chat_request(self, request: ChatRequest) -> dict:
        """Convert ChatRequest to an Ollama-compatible request dict."""
        result = {
            "model": request.model,
            "messages": [
                self._serialize_message(msg) for msg in request.messages
            ],
            "stream": request.stream,
        }

        if request.tools is not None:
            result["tools"] = [
                self._serialize_tool(tool) for tool in request.tools
            ]

        if request.options is not None:
            result["options"] = self._serialize_generation_options(
                request.options
            )

        if request.keep_alive is not None:
            result["keep_alive"] = request.keep_alive

        return result

    def _serialize_message(self, message: Message) -> dict:
        """Convert Message to an Ollama-compatible message dict."""
        result = {
            "role": message.role.value,
            "content": message.content,
        }

        if message.tool_calls is not None:
            result["tool_calls"] = [
                self._serialize_tool_call(tc) for tc in message.tool_calls
            ]

        return result

    def _serialize_tool_call(self, tool_call: ToolCall) -> dict:
        """Serialize ToolCall to Ollama API format."""
        return {
            "function": {
                "name": tool_call.function.name,
                "arguments": tool_call.function.arguments,
            },
        }

    def _serialize_generation_options(
            self, options: GenerationOptions
        ) -> dict:
        """Convert GenerationOptions to an Ollama options dict."""
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

    def _serialize_generate_request(self, request: GenerateRequest) -> dict:
        """Convert GenerateRequest to an Ollama-compatible request dict."""
        result = {
            "model": request.model,
            "prompt": request.prompt,
            "stream": request.stream,
        }

        if request.system is not None:
            result["system"] = request.system

        if request.options is not None:
            result["options"] = self._serialize_generation_options(
                request.options
            )

        if request.keep_alive is not None:
            result["keep_alive"] = request.keep_alive

        return result

    def _parse_chat_response(
            self, data: dict, model: str
        ) -> ChatResponse:
        """Parse an Ollama-compatible non-streaming chat response."""
        message = data["message"]

        return ChatResponse(
            model=data.get("model", model),
            message=Message(
                role=Role(message["role"]),
                content=message.get("content") or "",
                tool_calls=(
                    self._parse_tool_calls(message.get("tool_calls"))
                    if "tool_calls" in message
                    else None
                ),
                thinking=message.get("thinking"),
            ),
            done=data["done"],
            total_duration=data.get("total_duration"),
            load_duration=data.get("load_duration"),
            prompt_tokens=data.get("prompt_eval_count"),
            prompt_duration=data.get("prompt_eval_duration"),
            output_tokens=data.get("eval_count"),
            output_duration=data.get("eval_duration"),
        )

    def _parse_generate_response(
            self, data: dict, model: str
        ) -> GenerateResponse:
        """Parse an Ollama-compatible non-streaming completion."""
        return GenerateResponse(
            model=data.get("model", model),
            response=data["response"],
            done=data["done"],
            thinking=data.get("thinking"),
            total_duration=data.get("total_duration"),
            load_duration=data.get("load_duration"),
            prompt_tokens=data.get("prompt_eval_count"),
            prompt_duration=data.get("prompt_eval_duration"),
            output_tokens=data.get("eval_count"),
            output_duration=data.get("eval_duration"),
        )

    async def _parse_sse_stream(
            self, response: httpx.Response, model: str = "",
        ) -> AsyncGenerator[ChatResponseChunk, None]:
        """Parse an Ollama-compatible SSE chat stream."""
        async for data in self._iter_sse_data(response):
            yield self._parse_chat_chunk(data, model)

    def _parse_chat_chunk(
            self, data: dict, model: str = "",
        ) -> ChatResponseChunk:
        """Parse an Ollama JSON chunk to ChatResponseChunk."""
        message_data = data.get("message", {})

        tool_calls = None
        if "tool_calls" in message_data:
            tool_calls = self._parse_tool_calls(message_data["tool_calls"])

        return ChatResponseChunk(
            model=data.get("model", model),
            message=Message(
                role=Role(message_data.get("role") or "assistant"),
                content=message_data.get("content") or "",
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
        """Parse Ollama tool calls from response data."""
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

    async def _parse_generate_stream(
            self, response: httpx.Response, model: str = "",
        ) -> AsyncGenerator[GenerateChunk, None]:
        """Parse an Ollama-compatible SSE completion stream."""
        async for data in self._iter_sse_data(response):
            yield self._parse_generate_chunk(data, model)

    def _parse_generate_chunk(
            self, data: dict, model: str = "",
        ) -> GenerateChunk:
        """Parse an Ollama JSON chunk to GenerateChunk."""
        return GenerateChunk(
            model=data.get("model", model),
            response=data.get("response") or "",
            done=data["done"],
            thinking=data.get("thinking"),
            done_reason=data.get("done_reason"),
            total_duration=data.get("total_duration"),
            load_duration=data.get("load_duration"),
            prompt_tokens=data.get("prompt_eval_count"),
            prompt_duration=data.get("prompt_eval_duration"),
            output_tokens=data.get("eval_count"),
            output_duration=data.get("eval_duration"),
        )
