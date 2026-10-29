"""
Common HTTP client base for LLM inference endpoints.
"""

from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import Self, TypeVar
from hola.client.config import ClientConfig
from hola.client.exceptions import (
    ClientConnectionError,
    ClientError,
    ClientRequestError,
)
from hola.errors import type_assert, value_assert
from hola.tools.types import ToolCall, ToolCallFunction, ToolDefinition
from hola.types import (
    ChatRequest,
    GenerateRequest,
    GenerationOptions,
    Message,
    Role,
    ChatResponse,
    ChatResponseChunk,
    GenerateChunk,
    GenerateResponse,
)
import asyncio
import json
import logging
import httpx


logger = logging.getLogger(__name__)
T = TypeVar("T")


class BaseClient:
    """HTTP client for OpenAI-compatible LLM inference APIs."""
    chat_endpoint: str = "/v1/chat/completions"
    generate_endpoint: str = "/v1/completions"
    connection_error: type[ClientError] = ClientConnectionError
    request_error: type[ClientError] = ClientRequestError
    client_error: type[ClientError] = ClientError

    def __init__(
            self, config: ClientConfig, *,
            chat_endpoint: str | None = None,
            generate_endpoint: str | None = None,
        ) -> None:
        """Initialize the client with an `httpx.AsyncClient`."""
        self._config = config
        self._client = httpx.AsyncClient(
            base_url=config.base_url,
            timeout=httpx.Timeout(config.timeout),
        )

        if config.api_key is not None:
            self._client.headers.update(
                {"Authorization": f"Bearer {config.api_key}"}
            )

        if chat_endpoint is not None:
            self.chat_endpoint = chat_endpoint
        if generate_endpoint is not None:
            self.generate_endpoint = generate_endpoint

    async def _with_retries(
            self, func: Callable[[], Awaitable[T]], *,
            retries: int | None = None,
        ) -> T:
        """Run `func` with retries on connection errors.
            `httpx.ConnectError` and `httpx.TimeoutException` are
            retried up to `retries` (or `config.max_retries` when
            `retries` is `None`) times; other `httpx` errors are mapped
            to the configured client exception and raised immediately.
            `retries` must be `>= 0`. Passing `retries=0` disables
            retry/backoff while retaining exception mapping.
        """
        if retries is None:
            retries = self._config.max_retries
        type_assert(type(retries) is int, 'retries must be int')
        value_assert(retries >= 0, "retries must be >= 0")

        delay = self._config.retry_delay
        type_assert(
            type(delay) in (float, int),
            "config delay must be int|float",
        )

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
                    raise self.connection_error(
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
                logger.error(
                    "HTTP status error %s: %s",
                    e.response.status_code,
                    e,
                )
                raise self.request_error(
                    f"HTTP error {e.response.status_code}: {e}"
                ) from e

            except httpx.HTTPError as e:
                logger.error("HTTP error: %s", e)
                raise self.client_error(f"HTTP error: {e}") from e

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

    async def _iter_sse_data(
            self, response: httpx.Response,
        ) -> AsyncGenerator[dict, None]:
        """Yield decoded JSON objects from an SSE response, skipping
            blank lines, `[DONE]`, and malformed JSON.
        """
        async for line in response.aiter_lines():
            line = line.strip()

            if not line:
                continue

            if line.startswith("data: "):
                data = line[6:]
            else:
                data = line

            if data == "[DONE]":
                continue

            try:
                yield json.loads(data)
            except json.JSONDecodeError:
                continue

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
            lambda: self._send_stream(self.chat_endpoint, payload)
        )

        try:
            async for chunk in self._parse_sse_stream(
                    response, request.model
                ):
                yield chunk
        except httpx.HTTPError as e:
            raise self.connection_error(f"Stream failed: {e}") from e
        finally:
            await response.aclose()

    async def _chat_nonstream(self, request: ChatRequest) -> ChatResponse:
        request_dict = self._serialize_chat_request(request)
        request_dict["stream"] = False
        response = await self._with_retries(
            lambda: self._post(self.chat_endpoint, request_dict)
        )
        return self._parse_chat_response(response.json(), request.model)

    def _parse_chat_response(
            self, data: dict, model: str
        ) -> ChatResponse:
        """Parse an OpenAI-compatible non-streaming chat response."""
        choice = data["choices"][0]
        message = choice["message"]
        usage = data.get("usage") or {}

        return ChatResponse(
            model=data.get("model", model),
            message=Message(
                role=Role(message.get("role") or "assistant"),
                content=message.get("content") or "",
                tool_calls=self._parse_tool_calls(
                    message.get("tool_calls")
                ),
                thinking=(
                    message.get("reasoning_content")
                    or message.get("reasoning")
                    or message.get("thinking")
                ),
            ),
            done=True,
            prompt_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
        )

    async def generate(
            self, request: GenerateRequest
        ) -> AsyncGenerator[GenerateChunk, None] | GenerateResponse:
        """Generate text using the configured generate endpoint."""
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
            lambda: self._send_stream(self.generate_endpoint, payload)
        )

        try:
            async for chunk in self._parse_generate_stream(
                    response, request.model
                ):
                yield chunk
        except httpx.HTTPError as e:
            raise self.connection_error(f"Stream failed: {e}") from e
        finally:
            await response.aclose()

    async def _generate_nonstream(
            self, request: GenerateRequest
        ) -> GenerateResponse:
        request_dict = self._serialize_generate_request(request)
        request_dict["stream"] = False
        response = await self._with_retries(
            lambda: self._post(self.generate_endpoint, request_dict)
        )
        return self._parse_generate_response(response.json(), request.model)

    def _parse_generate_response(
            self, data: dict, model: str
        ) -> GenerateResponse:
        """Parse an OpenAI-compatible non-streaming completion."""
        choice = data["choices"][0]
        usage = data.get("usage") or {}

        return GenerateResponse(
            model=data.get("model", model),
            response=choice.get("text") or "",
            done=True,
            prompt_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
        )

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
        await self.close()

    def _serialize_chat_request(self, request: ChatRequest) -> dict:
        """Convert ChatRequest to an OpenAI-compatible request dict."""
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
            result.update(
                self._serialize_generation_options(request.options)
            )

        return result

    def _serialize_message(self, message: Message) -> dict:
        """Convert Message to an OpenAI-compatible message dict."""
        result = {
            "role": message.role.value,
            "content": message.content,
        }

        if message.tool_calls is not None:
            result["tool_calls"] = [
                self._serialize_tool_call(tc) for tc in message.tool_calls
            ]

        if message.role is Role.TOOL and message.call_id is not None:
            result["tool_call_id"] = message.call_id

        return result

    def _serialize_tool(self, tool: ToolDefinition) -> dict:
        """Serialize ToolDefinition to API format."""
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }

    def _serialize_tool_call(self, tool_call: ToolCall) -> dict:
        """Serialize ToolCall to OpenAI API format."""
        return {
            "id": tool_call.call_id,
            "type": "function",
            "function": {
                "name": tool_call.function.name,
                "arguments": json.dumps(tool_call.function.arguments),
            },
        }

    def _serialize_generation_options(
            self, options: GenerationOptions
        ) -> dict:
        """Convert GenerationOptions to top-level API parameters."""
        result = {}

        if options.temperature is not None:
            result["temperature"] = options.temperature

        if options.top_p is not None:
            result["top_p"] = options.top_p

        if options.top_k is not None:
            result["top_k"] = options.top_k

        if options.num_predict is not None:
            result["max_tokens"] = options.num_predict

        if options.seed is not None:
            result["seed"] = options.seed

        return result

    async def _parse_sse_stream(
            self, response: httpx.Response, model: str = "",
        ) -> AsyncGenerator[ChatResponseChunk, None]:
        """Parse an OpenAI-compatible SSE chat stream."""
        tool_call_acc: dict[int, dict] = {}
        finished = False

        async for data in self._iter_sse_data(response):
            choices = data.get("choices") or []

            if not choices:
                continue

            choice = choices[0]
            delta = choice.get("delta") or {}
            finish_reason = choice.get("finish_reason")
            usage = data.get("usage") or {}
            self._accumulate_tool_calls(
                tool_call_acc, delta.get("tool_calls")
            )

            parsed_tool_calls = None
            if finish_reason is not None and tool_call_acc:
                parsed_tool_calls = self._parse_tool_calls(
                    list(tool_call_acc.values())
                )

            done = finish_reason is not None
            finished = finished or done

            yield ChatResponseChunk(
                model=data.get("model", model),
                message=Message(
                    role=Role(delta.get("role") or "assistant"),
                    content=delta.get("content") or "",
                    tool_calls=parsed_tool_calls,
                    thinking=(
                        delta.get("reasoning_content")
                        or delta.get("reasoning")
                        or delta.get("thinking")
                    ),
                ),
                done=done,
                done_reason=finish_reason,
                prompt_tokens=usage.get("prompt_tokens"),
                output_tokens=usage.get("completion_tokens"),
            )

        if not finished:
            parsed_tool_calls = (
                self._parse_tool_calls(list(tool_call_acc.values()))
                if tool_call_acc else None
            )
            yield ChatResponseChunk(
                model=model,
                message=Message(
                    role=Role.ASSISTANT,
                    content="",
                    tool_calls=parsed_tool_calls,
                ),
                done=True,
            )

    def _accumulate_tool_calls(
            self, accumulator: dict[int, dict],
            fragments: list[dict] | None,
        ) -> None:
        """Accumulate fragmented streaming tool calls by index."""
        if not fragments:
            return

        for fragment in fragments:
            index = fragment.get("index", 0)
            current = accumulator.setdefault(
                index,
                {
                    "id": None,
                    "type": "function",
                    "function": {"name": "", "arguments": ""},
                },
            )
            call_id = fragment.get("id")

            if call_id is not None:
                current["id"] = call_id

            function = fragment.get("function") or {}
            name = function.get("name")

            if name:
                current["function"]["name"] = name

            arguments = function.get("arguments")

            if isinstance(arguments, str):
                current["function"]["arguments"] += arguments
            elif isinstance(arguments, dict):
                current["function"]["arguments"] = json.dumps(arguments)

    def _parse_tool_calls(
            self, tool_calls_data: list[dict] | None
        ) -> list[ToolCall] | None:
        """Parse OpenAI tool calls from response data."""
        if tool_calls_data is None:
            return None

        result = []

        for tool_call_data in tool_calls_data:
            function = tool_call_data.get("function") or {}
            arguments = function.get("arguments") or {}

            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments) if arguments else {}
                except json.JSONDecodeError:
                    logger.warning(
                        "Could not decode tool call arguments: %s",
                        arguments,
                    )
                    arguments = {}

            tool_call = ToolCall(
                function=ToolCallFunction(
                    name=function.get("name", ""),
                    arguments=arguments,
                ),
            )
            call_id = tool_call_data.get("id")

            if call_id is not None:
                tool_call.call_id = call_id

            result.append(tool_call)

        return result

    def _serialize_generate_request(self, request: GenerateRequest) -> dict:
        """Convert GenerateRequest to an OpenAI-compatible request dict."""
        prompt = request.prompt

        if request.system is not None:
            prompt = f"{request.system}\n\n{prompt}"

        result = {
            "model": request.model,
            "prompt": prompt,
            "stream": request.stream,
        }

        if request.options is not None:
            result.update(
                self._serialize_generation_options(request.options)
            )

        return result

    async def _parse_generate_stream(
            self, response: httpx.Response, model: str = "",
        ) -> AsyncGenerator[GenerateChunk, None]:
        """Parse an OpenAI-compatible SSE completion stream."""
        finished = False

        async for data in self._iter_sse_data(response):
            choices = data.get("choices") or []

            if not choices:
                continue

            chunk = self._parse_generate_chunk(data, model)
            finished = finished or chunk.done
            yield chunk

        if not finished:
            yield GenerateChunk(
                model=model,
                response="",
                done=True,
            )

    def _parse_generate_chunk(
            self, data: dict, model: str = "",
        ) -> GenerateChunk:
        """Parse an OpenAI-compatible streaming completion chunk."""
        choice = data["choices"][0]
        finish_reason = choice.get("finish_reason")
        usage = data.get("usage") or {}

        return GenerateChunk(
            model=data.get("model", model),
            response=choice.get("text") or "",
            done=finish_reason is not None,
            done_reason=finish_reason,
            prompt_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
        )
