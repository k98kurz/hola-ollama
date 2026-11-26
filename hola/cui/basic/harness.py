from .ainput import (
    BlockRequest, Console, RED, MAGENTA, BLUE, GREEN, CYAN, RESET, YELLOW
)
from .response_fsm import ResponseFSM, ResponseEvent, ResponseState
from collections.abc import AsyncGenerator
from dataclasses import fields
from typing import Awaitable, Callable
from hola.client import BaseClient
from hola.errors import type_assert, value_assert
from hola.stats import stats_from_response
from hola.prompts.adapters import (
    tool_enabled_chat_adapter, xml_workaround_chat_adapter
)
from hola.tools.compat import process_assistant_message
from hola.tools.core import execute_tool_call
from hola.tools.types import ToolCall, ToolCallFunction, ToolDefinition
from hola.types import (
    ChatResponse, ChatResponseChunk, Message, MessageStats, Role, SessionProtocol
)
import asyncio
import contextlib
import json
import traceback


def _param_schema_type(v: dict) -> str:
    if 'type' in v:
        return v['type']
    if 'anyOf' in v:
        return 'anyOf'
    if 'items' in v:
        return 'array'
    if 'properties' in v:
        return 'object'
    return 'any'


class ChatHarness:
    """Sequential chat loop fed by an asynchronous input queue."""
    def __init__(
            self, client: BaseClient, model: str, console: Console,
            system: str = "", show_thinking: bool = False,
            tools: list[ToolDefinition] | None = None, max_tool_calls: int = 999,
            session: SessionProtocol | None = None, use_xml: bool = False
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
        tools = tools or []
        type_assert(isinstance(tools, list), "tools must be list[ToolDefinition]")
        type_assert(
            all(isinstance(t, ToolDefinition) for t in tools),
            "tools must be list[ToolDefinition]"
        )
        self.tools = tools
        type_assert(isinstance(max_tool_calls, int), "max_tool_calls must be int")
        value_assert(max_tool_calls > 0, "max_tool_calls must be positive")
        self.max_tool_calls = max_tool_calls
        self.session = session
        self.use_xml = use_xml
        self.verbose = False
        self.response_fsm = ResponseFSM()
        self._register_response_hooks()
        self.stream_response = True
        self.slash_commands = {}
        self._register_slash_commands()
        self.effort = None
        self.show_stats = True

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
                "system": self.system,
                "effort": self.effort,
                "stats": self.show_stats,
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
            self.stream_response = not self.stream_response
            await self.console.write_line(
                "Response streaming enabled" if self.stream_response else
                "Response streaming disabled"
            )

        @self.slash("thinking", "toggle show thinking")
        async def toggle_thinking(_: str):
            self.show_thinking = not self.show_thinking
            await self.console.write_line(
                "Thinking shown" if self.show_thinking else "Thinking hidden"
            )

        @self.slash("system", "change or view the system prompt")
        async def system(text: str):
            text = text[7:].strip()
            if text:
                self.system = text
            await self.console.write_line(f"System prompt: {self.system}")

        @self.slash("effort", "view or change the reasoning effort level")
        async def effort(text: str):
            text = text[7:].strip()
            if text:
                self.effort = text
            await self.console.write_line(f"Reasoning effort: {self.effort}")

        @self.slash("noeffort", "unset the reasoning effort")
        def noeffort(_: str):
            self.effort = None

        @self.slash("stats", "toggle show stats")
        async def stats(_: str):
            self.show_stats = not self.show_stats
            await self.console.write_line(
                "Stats shown" if self.show_stats else "Stats hidden"
            )

        @self.slash("rollback", "rollback the message history one turn")
        async def rollback(_: str):
            if not self.history:
                await self.console.write_line("nothing to roll back")
                return
            user_messages = []
            for i, m in enumerate(self.history):
                if m.role is Role.USER:
                    user_messages.append(i)
            if len(user_messages) < 2:
                self.history.clear()
                await self.console.write_line("roll back cleared history")
                return
            rollback_to = user_messages[-1]
            self.history = [
                m for i, m in enumerate(self.history)
                if i < rollback_to
            ]
            await self.console.write_line("rolled back one turn")

        @self.slash("reprint", "reprint message history")
        async def reprint(_: str):
            await self.reprint()

        @self.slash("tool", "attempt a tool call")
        async def tool(text: str):
            text = text[5:].strip()
            if text:
                parts = text.split()
                name = parts[0]
                args_str = " ".join(parts[1:])
                try:
                    arguments = json.loads(args_str) if args_str else {}
                except json.JSONDecodeError:
                    arguments = {}
                    for p in parts[1:]:
                        if "=" in p:
                            k, v = p.split("=", 1)
                            arguments[k] = v
                        else:
                            arguments[f"arg{len(arguments)}"] = p
                try:
                    tool = ToolCallFunction(name, arguments)
                    tc = ToolCall(tool)
                    result = await execute_tool_call(
                        tc, self.session, {t.name for t in self.tools}
                    )
                except Exception as e:
                    result = e
                await self.console.write_line(f"{result}")
                return

            lines = []
            for t in self.tools:
                params = {
                    k: _param_schema_type(v)
                    for k, v in t.parameters['properties'].items()
                }
                lines.append(f"{t.name}: {json.dumps(params, indent=1)}")
            await self.console.write_line("\n".join(lines))

        @self.slash("clear", "start new session")
        @self.slash("new", "start new session")
        async def clear(_: str):
            self.reset_history()
            await self.console.write_line("Session history cleared")

        @self.slash("block", "enter block input mode")
        async def toggle_block_input(_: str):
            return await self._handle_block_request()

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

    async def handle_slash_command(self, text: str) -> bool | None:
        command = text.split()[0]
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
            if  (   transition.from_state is not ResponseState.CALLING_TOOLS
                    and transition.on_event is ResponseEvent.CALL_TOOL
                ):
                self.response_fsm.add_transition_hook(
                    transition, lambda *_: self.console.write_line("")
                )

    async def _open_thought_hook(
            self, transition, context: dict, data: object
        ) -> None:
        if context.get("stream_thinking"):
            await context["console"].write_stream(f"{GREEN}<thought>{RESET}\n")

    async def _close_thought_hook(
            self, transition, context: dict, data: object
        ) -> None:
        if context.get("stream_thinking"):
            await context["console"].write_stream(f"\n{GREEN}</thought>{RESET}\n\n")

    async def _write_prefix_hook(
            self, transition, context: dict, data: object
        ) -> None:
        await context["console"].write_stream(f"{GREEN}assistant>{RESET} ")

    async def submit(self, text: str) -> None:
        """Queue a user message for the worker."""
        self.queue.put_nowait(Message(role=Role.USER, content=text))
        await self.console.write_line("\n")

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
                    if await self._handle_block_request():
                        break
                    continue

                line = line.strip()

                if len(line) and line[0] == "/":
                    if await self.handle_slash_command(line):
                        break
                    continue

                if line:
                    await self.submit(line)
        finally:
            self._worker_task.cancel()

            with contextlib.suppress(asyncio.CancelledError):
                await self._worker_task

            self._worker_task = None

    async def _handle_block_request(self) -> bool | None:
        await self.console.write_line(
            "<Block input mode active; use Ctrl-D to submit>"
        )
        block_text = await self.console.ainput_block()
        if block_text is None:
            return
        block_text = block_text.strip()
        if not block_text: return
        if block_text[0] == "/":
            return await self.handle_slash_command(block_text)
        else:
            await self.submit(block_text)

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
                if self.use_xml:
                    request = xml_workaround_chat_adapter(
                        phase_instructions=self.system or "",
                        messages=list(self.history),
                        tools=self.tools,
                        model_name=self.model,
                    )
                else:
                    request = tool_enabled_chat_adapter(
                        phase_instructions=self.system or "",
                        messages=list(self.history),
                        tools=self.tools,
                        model_name=self.model,
                    )
                request.stream = stream_response
                request.options.reasoning_effort = self.effort

                try:
                    response = await self.client.chat(request)

                    if stream_response:
                        assistant = await self._stream_response(response)
                    else:
                        assistant = await self._print_response(response)

                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    self.history.pop()
                    await self.console.write_line(f"{RED}[error] {error}{RESET}")
                    if self.verbose:
                        await self.console.write_line(
                            f"{MAGENTA}{traceback.format_exc()}{RESET}"
                        )
                    break

                self.history.append(assistant)

                if not assistant.tool_calls:
                    break

                for tc in assistant.tool_calls:
                    i += 1
                    try:
                        result = await execute_tool_call(
                            tc, self.session, {t.name for t in self.tools}
                        )
                    except Exception as error:
                        result = f"Tool execution error: {error}"
                    if self.verbose:
                        await self.console.write_line(
                            f"{YELLOW}"
                            f"<tool name={tc.function.name} callid={tc.call_id}>"
                            f"{result}</tool>{RESET}"
                        )
                    self.history.append(Message(
                        role=Role.TOOL,
                        content=str(result),
                        call_id=tc.call_id,
                    ))

            if i >= self.max_tool_calls:
                await self.console.write_line(
                    "[loop]: max_tool_calls reached or exceeded"
                )

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
        last_chunk: ChatResponseChunk | None = None
        fsm = self.response_fsm
        fsm.context["console"] = self.console
        fsm.context["stream_thinking"] = stream_thinking
        fsm.reset()
        stat_fields = [f.name for f in fields(MessageStats)]

        try:
            stats = MessageStats(*[0 for _ in range(len(stat_fields))])
            async for chunk in stream:
                thinking = chunk.message.thinking or None
                content = chunk.message.content or ""
                for f in stat_fields:
                    if getattr(chunk, f, None):
                        setattr(stats, f, getattr(stats, f) + getattr(chunk, f))

                if thinking:
                    thinking_parts.append(thinking)
                if stream_thinking and thinking:
                    await fsm.input(ResponseEvent.WRITE_THOUGHT)
                    if fsm.current is ResponseState.THINKING:
                        await self.console.write_stream(thinking)

                if chunk.message.tool_calls and not self.use_xml:
                    await fsm.input(ResponseEvent.CALL_TOOL)
                    tool_calls = chunk.message.tool_calls

                    tc_lines = []
                    for tc in tool_calls:
                        tc_lines.append(
                            f"{YELLOW}<call id={tc.call_id} "
                            f"name={tc.function.name} "
                            f"args={json.dumps(tc.function.arguments)} />{RESET}"
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
            await self.console.write_line(f"{RED}[error] {e}{RESET}")
            if self.verbose:
                await self.console.write_line(
                    f"{MAGENTA}{traceback.format_exc()}{RESET}"
                )
        finally:
            await fsm.input(ResponseEvent.END_RESPONSE)
            await self.console.write_stream("", done=True)

        stats.total_duration /= 1_000_000_000
        stats.load_duration /= 1_000_000_000
        stats.prompt_duration /= 1_000_000_000
        stats.output_duration /= 1_000_000_000
        if stats.prompt_tokens and stats.prompt_duration:
            stats.prompt_tps = stats.prompt_tokens / stats.prompt_duration
        if stats.output_tokens and stats.output_duration:
            stats.output_tps = stats.output_tokens / stats.output_duration

        thinking = "".join(thinking_parts) or None
        content = "".join(content_parts)
        if self.use_xml:
            content, tool_calls = process_assistant_message(content, self.tools)
        if self.show_stats and stats:
            await self.console.write_line(f"[stats] {stats}\n")

        return Message(
            role=Role.ASSISTANT,
            content=content,
            thinking=thinking,
            tool_calls=tool_calls,
            stats=stats,
        )

    async def _print_response(self, response: ChatResponse) -> Message:
        """Print a non-streaming chat response to the console, returning
            the assistant message.
        """
        thinking = response.message.thinking or None
        content = response.message.content or ""
        if self.use_xml:
            content, tool_calls = process_assistant_message(content, self.tools)
        else:
            tool_calls = response.message.tool_calls or []

        lines = [""]
        if thinking and self.show_thinking:
            lines.append(
                f"{GREEN}<thought>{RESET}\n{thinking}\n{GREEN}</thought>{RESET}")
            lines.append("")

        if content:
            lines.append(f"{GREEN}assistant>{RESET} {content}")

        for tc in tool_calls:
            lines.append(
                f"{YELLOW}<call id={tc.call_id} "
                f"name={tc.function.name} "
                f"args={json.dumps(tc.function.arguments)} />{RESET}"
            )

        stats = stats_from_response(response)
        if self.show_stats:
            lines.append(f"[stats] {stats}")

        await self.console.write_line("\n".join(lines))

        return Message(
            role=Role.ASSISTANT,
            content=content,
            thinking=thinking,
            tool_calls=tool_calls,
            stats=stats,
        )

    async def reprint(self):
        if self.system:
            await self.console.write_line(f"{BLUE}[system]{RESET} {self.system}\n")
        for m in self.history:
            if m.role is Role.SYSTEM:
                await self.console.write_line(f"{BLUE}[system]{RESET} {m.content}\n")
            elif m.role is Role.USER:
                await self.console.write_line(f"{CYAN}[user]{RESET} {m.content}\n")
            elif m.role is Role.ASSISTANT:
                if m.thinking and self.show_thinking:
                    await self.console.write_line(
                        f"{GREEN}<thought>{RESET}\n{m.thinking}\n"
                        f"{GREEN}</thought>{RESET}\n"
                    )
                if m.content:
                    await self.console.write_line(
                        f"{GREEN}[assistant]{RESET} {m.content}\n"
                    )
                for tc in m.tool_calls or []:
                    await self.console.write_line(
                        f"{YELLOW}<call id={tc.call_id} name={tc.function.name} "
                        f"args={json.dumps(tc.function.arguments)} />{RESET}"
                    )
                if m.stats and self.show_stats:
                    await self.console.write_line(f"[stats] {m.stats}\n")
            elif m.role is Role.TOOL and self.verbose:
                await self.console.write_line(f"{YELLOW}[tool]{RESET} {m.content}\n")
