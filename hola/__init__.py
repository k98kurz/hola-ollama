"""
Hola - An extensible harness for Ollama for local AI agent development.
"""

from hola.ollama import (
    OllamaClient,
    ModelWarmer,
    tool,
    register_tool,
    get_tool,
    list_tools,
    execute_tool_call,
    Message,
    ToolDefinition,
    ToolCall,
    GenerationOptions,
    ChatRequest,
    ChatResponseChunk,
    ChatResponse,
    ModelInfo,
    RunningModel,
    ClientConfig,
    WarmerConfig,
    ModelPriority,
    OllamaClientError,
    OllamaConnectionError,
    OllamaRequestError,
    OllamaModelError,
    ToolExecutionError,
)
