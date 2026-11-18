from .ainput import BlockRequest, Console
from .response_fsm import ResponseFSM, ResponseEvent, ResponseState
from collections.abc import AsyncGenerator
from typing import Awaitable, Callable
from hola.client import BaseClient
from hola.errors import type_assert, value_assert
from hola.prompts.adapters import tool_enabled_chat_adapter
from hola.tools.core import execute_tool_call
from hola.tools.types import ToolCall, ToolDefinition
from hola.types import ChatResponse, ChatResponseChunk, Message, Role
import asyncio
import contextlib
import json


class ChatHarness:
    """Sequential chat loop fed by an asynchronous input queue."""
    def __init__(
            self, client: BaseClient, model: str, console: Console,
            system: str = "", show_thinking: bool = False,
            tools: list[ToolDefinition] = [], max_tool_calls: int = 999
        ) -> None:
        """Initialize the harness. An optional `system` message is
            placed at the start of the conversation.
        """
        self.client = client
        self.model = model
        self.console = console
        self.history: list[Message] = []
        self.queue: asyncio.Queue[Message] = asyncio.Queue()
        self._worker_task: asyncio.Task | None = None
        self.system = system
        self.reset_history()
        self.show_thinking = show_thinking
        type_assert(isinstance(tools, list), "tools must be list[ToolDefinition]")
        type_assert(
            all(isinstance(t, ToolDefinition) for t in tools),
            "tools must be list[ToolDefinition]"
        )
        self.tools = tools
        type_assert(isinstance(max_tool_calls, int), "max_tool_calls must be int")
        value_assert(max_tool_calls > 0, "max_tool_calls must be positive")
        self.max_tool_calls = max_tool_calls
        self.verbose = False
        self.response_fsm = ResponseFSM()
        self._register_response_hooks()
        self.stream_response = True
        self.slash_commands = {}
        self._register_slash_commands()

    def slash(self, command: str, description: str = ''):
        """Decorator for registering a slash command. Calls
            `register_slash_command` under the hood.
        """
        def wrap(fn: Callable):
            self.register_slash_command(command, fn, description)
            return fn
        return wrap

    def register_slash_command(
            self, command: str,
            fn: Callable[[str], bool | None | Awaitable[bool | None]],
            description: str = ''
        ) -> None:
        """Register an additional slash command. `command` is the name
            without the leading slash; `fn` receives the full command
            line text.
        """
        self.slash_commands[f"/{command}"] = (fn, description)

    def _register_slash_commands(self):
        @self.slash("quit", "quit the harness")
        def quit(_: str):
            return True

        @self.slash("cancel", "cancel the current request cycle")
        async def cancel(_: str):
            self._worker_task.cancel()
            self._worker_task = asyncio.create_task(self._worker())

        @self.slash(
            "settings", "show settings with `/settings [key]` or `/settings`"
        )
        async def settings(text: str):
            settings = {
                "stream": self.stream_response,
                "thinking": self.show_thinking,
                "verbose": self.verbose,
            }
            parts = text.split()
            if len(parts) == 2:
                flag = parts[1]
                if flag in settings:
                    await self.console.write_line(f"{flag}: {settings[flag]}")
                return
            for k,v in settings.items():
                await self.console.write_line(f"{k}: {v}")

        @self.slash("stream", "toggle response streaming")
        async def toggle_streaming(_: str):
            if self.stream_response:
                self.stream_response = False
                await self.console.write_line("Response streaming disabled")
            else:
                self.stream_response = True
                await self.console.write_line("Response streaming enabled")

        @self.slash("thinking", "toggle show thinking")
        async def toggle_thinking(_: str):
            if self.show_thinking:
                self.show_thinking = False
                await self.console.write_line("Thinking hidden")
            else:
                self.show_thinking = True
                await self.console.write_line("Thinking shown")

        @self.slash("clear", "start new session")
        @self.slash("new", "start new session")
        async def clear(_: str):
            self.reset_history()
            await self.console.write_line("Session history cleared")

        @self.slash("block", "enter block input mode")
        async def toggle_block_input(_: str):
            await self.console.write_line("<Block input mode active>")
            block_text = await self.console.ainput_block()
            if block_text is None:
                return
            if block_text.strip():
                self.submit(block_text)

        @self.slash("verbose", "toggle verbose output (tool call results)")
        async def toggle_verbose(_: str):
            self.verbose = not self.verbose
            await self.console.write_line(
                "Verbose enabled" if self.verbose else "Verbose disabled"
            )

        @self.slash("help", "show slash command help")
        @self.slash("?", "show slash command help")
        async def help(text: str):
            commands = {
                k: v[1] or v[0].__doc__ or v[0].__name__
                for k, v in self.slash_commands.items()
            }
            parts = text.strip().split()
            if len(parts) == 2:
                cmd = parts[1]
                cmd = f"/{cmd}" if cmd[0] != "/" else cmd
                if cmd in commands:
                    await self.console.write_line(f"{cmd}: {commands[cmd]}")
                return
            for k,v in commands.items():
                await self.console.write_line(f"{k}: {v}")

    async def handle_slash_command(self, text: str) -> bool|None:
        command = text.split(" ")[0]
        if command not in self.slash_commands:
            await self.console.write_line(f"unrecognized slash command: {command}\n")
            return
        res = self.slash_commands[command][0](text)
        if asyncio.iscoroutine(res):
            res = await res
        await self.console.write_line("")
        return res

    def reset_history(self):
        self.history.clear()
        if self.system:
            self.history.append(Message(role=Role.SYSTEM, content=self.system))

    def _register_response_hooks(self) -> None:
        """Register transition hooks on the response FSM for tag and
            prefix side effects.
        """
        for transition in self.response_fsm.rules:
            if transition.on_event == ResponseEvent.WRITE_THOUGHT:
                self.response_fsm.add_transition_hook(
                    transition, self._open_thought_hook
                )
            if transition.from_state is ResponseState.THINKING:
                self.response_fsm.add_transition_hook(
                    transition, self._close_thought_hook
                )
            if transition.on_event is ResponseEvent.WRITE_CONTENT:
                self.response_fsm.add_transition_hook(
                    transition, self._write_prefix_hook
                )

    async def _open_thought_hook(
            self, transition, context: dict, data: object
        ) -> None:
        if context.get("stream_thinking"):
            await context["console"].write_stream("<thought>\n")

    async def _close_thought_hook(
            self, transition, context: dict, data: object
        ) -> None:
        if context.get("stream_thinking"):
            await context["console"].write_stream("\n</thought>\n\n")

    async def _write_prefix_hook(
            self, transition, context: dict, data: object
        ) -> None:
        await context["console"].write_stream("assistant> ")

    def submit(self, text: str) -> None:
        """Queue a user message for the worker."""
        self.queue.put_nowait(Message(role=Role.USER, content=text))

    async def run(self) -> None:
        """Read input until `/quit` or EOF. The worker is cancelled
            immediately when input ends.
        """
        self._worker_task = asyncio.create_task(self._worker())

        try:
            while True:
                line = await self.console.ainput()

                if line is None:
                    break

                if isinstance(line, BlockRequest):
                    await self.console.write_line("<Block input mode active>")
                    block_text = await self.console.ainput_block()
                    if block_text is None:
                        continue
                    if block_text.strip():
                        self.submit(block_text)
                    continue

                line = line.strip()

                if len(line) and line[0] == "/":
                    if await self.handle_slash_command(line):
                        break
                    continue

                if line:
                    self.submit(line)
        finally:
            self._worker_task.cancel()

            with contextlib.suppress(asyncio.CancelledError):
                await self._worker_task

            self._worker_task = None

    async def _worker(self) -> None:
        """Process queued messages one at a time so that history stays
            strictly ordered, streaming assistant output.
        """
        while True:
            message = await self.queue.get()
            self.history.append(message)
            stream_response = self.stream_response

            i = 0
            while i < self.max_tool_calls:
                request = tool_enabled_chat_adapter(
                    phase_instructions=self.system or "",
                    messages=list(self.history),
                    tools=self.tools,
                    model_name=self.model,
                )
                request.stream = stream_response

                try:
                    response = await self.client.chat(request)
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    self.history.pop()
                    await self.console.write_line(f"[error] {error}")
                    break

                if stream_response:
                    assistant = await self._stream_response(response)
                else:
                    assistant = await self._print_response(response)

                self.history.append(assistant)

                if not assistant.tool_calls:
                    break

                for tc in assistant.tool_calls:
                    i += 1
                    try:
                        result = await execute_tool_call(tc)
                    except Exception as error:
                        result = f"Tool execution error: {error}"
                    if self.verbose:
                        await self.console.write_line(
                            f"<tool name={tc.function.name} callid={tc.call_id}>"
                            f"{result}</tool>"
                        )
                    self.history.append(Message(
                        role=Role.TOOL,
                        content=str(result),
                        call_id=tc.call_id,
                    ))

            if i >= self.max_tool_calls:
                await self.console.write_line("[loop]: max_tool_calls reached or exceeded")

    async def _stream_response(
            self, stream: AsyncGenerator[ChatResponseChunk, None],
        ) -> Message:
        """Stream a chat response to the console, returning the
            assistant message. Thinking arrives before content; the
            first content chunk closes thinking and opens the assistant
            prefix.
        """
        await self.console.write_line("")
        stream_thinking = self.show_thinking
        thinking_parts: list[str] = []
        content_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        fsm = self.response_fsm
        fsm.context["console"] = self.console
        fsm.context["stream_thinking"] = stream_thinking
        fsm.reset()

        try:
            async for chunk in stream:
                thinking = chunk.message.thinking or ""
                content = chunk.message.content or ""

                if stream_thinking and thinking:
                    await fsm.input(ResponseEvent.WRITE_THOUGHT)
                    if fsm.current is ResponseState.THINKING:
                        await self.console.write_stream(thinking)
                        thinking_parts.append(thinking)

                if chunk.message.tool_calls:
                    await fsm.input(ResponseEvent.CALL_TOOL)
                    tool_calls = chunk.message.tool_calls

                    tc_lines = []
                    for tc in tool_calls:
                        tc_lines.append(
                            f"<call id={tc.call_id} "
                            f"name={tc.function.name} "
                            f"args={json.dumps(tc.function.arguments)} />"
                        )
                    if fsm.current is ResponseState.CALLING_TOOLS:
                        await self.console.write_stream("\n".join(tc_lines) + "\n")

                if content:
                    await fsm.input(ResponseEvent.WRITE_CONTENT)
                    if fsm.current is ResponseState.WRITING_CONTENT:
                        await self.console.write_stream(content)
                        content_parts.append(content)

                if chunk.done:
                    break
        except Exception as e:
            await self.console.write_line(f"An error was encountered: {e}")
            raise
        finally:
            await fsm.input(ResponseEvent.END_RESPONSE)
            await self.console.write_stream("", done=True)

        thinking = "".join(thinking_parts) or None
        return Message(
            role=Role.ASSISTANT,
            content="".join(content_parts),
            thinking=thinking,
            tool_calls=tool_calls,
        )

    async def _print_response(self, response: ChatResponse) -> Message:
        """Print a non-streaming chat response to the console, returning
            the assistant message.
        """
        thinking = response.message.thinking or ""
        content = response.message.content or ""
        tool_calls = response.message.tool_calls

        lines = [""]
        if thinking and self.show_thinking:
            lines.append(f"<thought>\n{thinking}\n</thought>")
            lines.append("")

        if content:
            lines.append(f"assistant> {content}")

        for tc in tool_calls:
            lines.append(
                f"<call id={tc.call_id} "
                f"name={tc.function.name} "
                f"args={json.dumps(tc.function.arguments)} />"
            )

        await self.console.write_line("\n".join(lines))

        return Message(
            role=Role.ASSISTANT,
            content=content or None,
            thinking=thinking or None,
            tool_calls=tool_calls,
        )
