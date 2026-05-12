"""
Ollama integration for Hola.
"""

from hola.ollama.client import OllamaClient
from hola.ollama.exceptions import (
    OllamaClientError,
    OllamaConnectionError,
    OllamaModelError,
    OllamaRequestError,
    ToolExecutionError,
)
from hola.ollama.config import (
    ClientConfig,
    WarmerConfig,
)
from hola.ollama.types import (
    ChatRequest,
    ChatResponse,
    ChatResponseChunk,
    FunctionDefinition,
    GenerationOptions,
    GenerateChunk,
    GenerateRequest,
    GenerateResponse,
    Message,
    ModelInfo,
    Role,
    RunningModel,
    ToolCall,
    ToolCallFunction,
    ToolDefinition,
)
from hola.ollama.tools import (
    ToolRegistry,
    execute_tool_call,
    get_tool,
    list_tools,
    register_tool,
    tool,
)
from hola.ollama.warmer import ModelWarmer

__all__ = [
    "OllamaClient",
    "OllamaClientError",
    "OllamaConnectionError",
    "OllamaModelError",
    "OllamaRequestError",
    "ToolExecutionError",
    "ClientConfig",
    "WarmerConfig",
    "ChatRequest",
    "ChatResponse",
    "ChatResponseChunk",
    "FunctionDefinition",
    "GenerationOptions",
    "GenerateChunk",
    "GenerateRequest",
    "GenerateResponse",
    "Message",
    "ModelInfo",
    "Role",
    "RunningModel",
    "ToolCall",
    "ToolCallFunction",
    "ToolDefinition",
    "ToolRegistry",
    "execute_tool_call",
    "get_tool",
    "list_tools",
    "register_tool",
    "tool",
    "ModelWarmer",
]
