"""
A minimal asynchronous chat CUI built on `BaseClient` against
OpenAI-compatible endpoints. Supports optional system prompts,
API keys, tool selection, and a thinking-mode toggle.
"""

from . import ChatHarness, Console
from .ainput import CYAN, RESET
from hola.client import BaseClient, ClientConfig
from hola.ollama import OllamaClient
from hola.session import Session
from hola.tools.types import ToolDefinition


DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODEL = "gemma4"
DEFAULT_PROMPT = f"{CYAN}you>{RESET} "
DEFAULT_SESSION = "hola-cli"


async def run_chat(
        host: str, model: str, system: str | None = None,
        api_key: str | None = None, show_thinking: bool = True,
        tools: list[ToolDefinition] | None = None,
        use_ollama: bool = False, use_xml: bool = False,
    ) -> None:
    """Run the chat harness until input ends."""
    config = ClientConfig(
        base_url=host, timeout=300.0, api_key=api_key,
    )
    client = OllamaClient(host, 120.0, config) if use_ollama else BaseClient(config)
    console = Console(DEFAULT_PROMPT)
    harness = ChatHarness(
        client, model, console, system, show_thinking, max_tool_calls=999,
        tools=tools, session=Session(DEFAULT_SESSION, True), use_xml=use_xml,
    )
    try:
        await harness.run()
    finally:
        console.close()
        await client.close()
