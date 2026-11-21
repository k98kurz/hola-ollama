"""
A minimal asynchronous chat CUI built on `BaseClient` against
OpenAI-compatible endpoints. Supports optional system prompts,
API keys, tool selection, and a thinking-mode toggle.
"""

from . import ChatHarness, Console
from hola.client import BaseClient, ClientConfig
from hola.session import Session
from hola.tools.types import ToolDefinition


DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODEL = "qwen3.5:2b"
DEFAULT_PROMPT = "you> "
DEFAULT_SESSION = "hola-cli"


async def run_chat(
        host: str, model: str, system: str | None = None,
        api_key: str | None = None, show_thinking: bool = True,
        tools: list[ToolDefinition] | None = None,
    ) -> None:
    """Run the chat harness until input ends."""
    config = ClientConfig(
        base_url=host, timeout=300.0, api_key=api_key,
    )
    client = BaseClient(config)
    console = Console(DEFAULT_PROMPT)
    harness = ChatHarness(
        client, model, console, system, show_thinking, max_tool_calls=999,
        tools=tools,
        session=Session(DEFAULT_SESSION, True),
    )
    try:
        await harness.run()
    finally:
        console.close()
        await client.close()
