"""
Ollama integration for Hola.
"""

from hola.ollama.client import OllamaClient
from hola.ollama.exceptions import (
    OllamaClientError,
    OllamaConnectionError,
    OllamaModelError,
    OllamaRequestError,
)
from hola.ollama.config import (
    ClientConfig,
    WarmerConfig,
)
from hola.ollama.types import (
    ChatResponse,
    ChatResponseChunk,
    GenerateChunk,
    GenerateResponse,
    ModelInfo,
    ModelMetadata,
    RunningModel,
)
from hola.types import (
    ChatRequest,
    GenerateRequest,
    GenerationOptions,
    LLMClient,
    Message,
    Role,
)
from hola.tools import (
    ToolCall,
    ToolCallFunction,
    ToolDefinition,
    ToolExecutionError,
    ToolRegistry,
    execute_tool_call,
    get_tool,
    list_tools,
    register_tool,
    tool,
)
from hola.ollama.warmer import ModelWarmer

