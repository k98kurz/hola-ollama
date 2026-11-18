from collections.abc import AsyncGenerator
from typing import Callable
from .ainput import BlockRequest, Console
from .response_fsm import ResponseFSM, ResponseEvent, ResponseState
from hola.client import BaseClient
from hola.types import ChatRequest, ChatResponse, ChatResponseChunk, Message, Role
import asyncio
import contextlib


class ChatHarness:
    """Sequential chat loop fed by an asynchronous input queue."""
    def __init__(
            self, client: BaseClient, model: str, console: Console,
            system: str | None = None, show_thinking: bool = False
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
        self.response_fsm = ResponseFSM()
        self._register_response_hooks()
        self.stream_response = True
        self.slash_commands = {}
        self._register_slash_commands()

    def _register_slash_commands(self):
        def slash(command: str):
            def wrap(fn: Callable):
                self.slash_commands[f"/{command}"] = fn
                return fn
            return wrap
        
        @slash("quit")
        def quit(_: str):
            return True

        @slash("stream")
        def toggle_streaming(_: str):
            if self.stream_response:
                self.stream_response = False
                self.console.write_line("Response streaming disabled")
            else:
                self.stream_response = True
                self.console.write_line("Response streaming enabled")

        @slash("thinking")
        def toggle_thinking(_: str):
            if self.show_thinking:
                self.show_thinking = False
                self.console.write_line("Thinking hidden")
            else:
                self.show_thinking = True
                self.console.write_line("Thinking shown")

        @slash("clear")
        def clear(_: str):
            self.reset_history()
            self.console.write_line("Session history cleared")

        @slash("block")
        async def toggle_block_input(_: str):
            self.console.write_line("<Block input mode active>")
            block_text = await self.console.ainput_block()
            if block_text is None:
                return
            if block_text.strip():
                self.submit(block_text)

    async def handle_slash_command(self, text: str):
        command = text.split(" ")[0]
        if command not in self.slash_commands:
            self.console.write_line(f"unrecognized slash command: {command}")
            return
        res = self.slash_commands[command](text)
        if asyncio.iscoroutine(res):
            res = await res
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
            if (transition.from_state == ResponseState.INIT
                    and transition.on_event == ResponseEvent.WRITE_THOUGHT):
                self.response_fsm.add_transition_hook(
                    transition, self._open_thought_hook
                )
            if transition.on_event == ResponseEvent.WRITE_CONTENT:
                if (transition.from_state
                        == ResponseState.WRITING_THOUGHT):
                    self.response_fsm.add_transition_hook(
                        transition, self._close_thought_hook
                    )
                self.response_fsm.add_transition_hook(
                    transition, self._write_prefix_hook
                )

    def _open_thought_hook(
            self, transition, context: dict, data: object
        ) -> None:
        if context.get("stream_thinking"):
            context["console"].write_stream("<thought>\n")

    def _close_thought_hook(
            self, transition, context: dict, data: object
        ) -> None:
        if context.get("stream_thinking"):
            context["console"].write_stream("\n</thought>\n\n")

    def _write_prefix_hook(
            self, transition, context: dict, data: object
        ) -> None:
        context["console"].write_stream("assistant> ")

    def submit(self, text: str) -> None:
        """Queue a user message for the worker."""
        self.queue.put_nowait(Message(role=Role.USER, content=text))

    async def run(self) -> None:
        """Read input until `/quit` or EOF. The worker is cancelled
            immediately when input ends.
        """
        worker = asyncio.create_task(self._worker())
        self._worker_task = worker

        try:
            while True:
                line = await self.console.ainput()

                if line is None:
                    break

                if isinstance(line, BlockRequest):
                    self.console.write_line("<Block input mode active>")
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
            worker.cancel()

            with contextlib.suppress(asyncio.CancelledError):
                await worker

            self._worker_task = None

    async def _worker(self) -> None:
        """Process queued messages one at a time so that history stays
            strictly ordered, streaming assistant output.
        """
        while True:
            message = await self.queue.get()
            self.history.append(message)
            stream_response = self.stream_response

            try:
                request = ChatRequest(
                    model=self.model,
                    messages=list(self.history),
                    stream=stream_response,
                )
                response = await self.client.chat(request)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                self.history.pop()
                self.console.write_line(f"[error] {error}")
                continue

            if stream_response:
                assistant = await self._stream_response(response)
            else:
                assistant = await self._print_response(response)

            self.history.append(assistant)

    async def _stream_response(
            self, stream: AsyncGenerator[ChatResponseChunk, None],
        ) -> Message:
        """Stream a chat response to the console, returning the
            assistant message. Thinking arrives before content; the
            first content chunk closes thinking and opens the assistant
            prefix.
        """
        self.console.write_line("")
        stream_thinking = self.show_thinking
        thinking_parts: list[str] = []
        content_parts: list[str] = []
        fsm = self.response_fsm
        fsm.context["console"] = self.console
        fsm.context["stream_thinking"] = stream_thinking
        fsm.reset()

        try:
            async for chunk in stream:
                thinking = chunk.message.thinking or ""
                content = chunk.message.content or ""

                if stream_thinking and thinking:
                    if fsm.current == ResponseState.INIT:
                        fsm.input(ResponseEvent.WRITE_THOUGHT)
                    if fsm.current == ResponseState.WRITING_THOUGHT:
                        self.console.write_stream(thinking)
                        thinking_parts.append(thinking)

                if content:
                    if fsm.current in (
                            ResponseState.INIT,
                            ResponseState.WRITING_THOUGHT
                        ):
                        fsm.input(ResponseEvent.WRITE_CONTENT)
                    if fsm.current == ResponseState.WRITING_CONTENT:
                        self.console.write_stream(content)
                        content_parts.append(content)

                if chunk.done:
                    break
        finally:
            if fsm.current == ResponseState.WRITING_THOUGHT:
                fsm.input(ResponseEvent.WRITE_CONTENT)
            if fsm.current == ResponseState.INIT:
                fsm.input(ResponseEvent.WRITE_CONTENT)
            if fsm.current == ResponseState.WRITING_CONTENT:
                fsm.input(ResponseEvent.END_RESPONSE)
            self.console.write_stream("", done=True)

        thinking = "".join(thinking_parts) or None
        return Message(
            role=Role.ASSISTANT,
            content="".join(content_parts),
            thinking=thinking,
        )

    async def _print_response(self, response: ChatResponse) -> Message:
        """Print a non-streaming chat response to the console, returning
            the assistant message.
        """
        thinking = response.message.thinking or ""
        content = response.message.content or ""

        lines = [""]
        if thinking and self.show_thinking:
            lines.append(f"<thought>\n{thinking}\n</thought>")
            lines.append("")

        if content:
            lines.append(f"assistant> {content}")

        self.console.write_line("\n".join(lines))

        return Message(
            role=Role.ASSISTANT,
            content=content or None,
            thinking=thinking or None,
        )
