"""
Hola - An extensible harness for Ollama for local AI agent development.
"""

from hola.errors import (
    HolaError,
    ToolExecutionError,
    PermissionDeniedError,
    VerificationException,
)
from hola.ollama import (
    OllamaClient,
    ModelWarmer,
    RunningModel,
    ClientConfig,
    WarmerConfig,
    OllamaClientError,
    OllamaConnectionError,
    OllamaRequestError,
    OllamaModelError,
)
from hola.types import (
    Role,
    Message,
    GenerationOptions,
    ChatRequest,
    GenerateRequest,
    LLMClient,
    ChatResponseChunk,
    ChatResponse,
    GenerateChunk,
    GenerateResponse,
    ModelInfo,
)
from hola.tools import (
    tool,
    register_tool,
    get_tool,
    list_tools,
    execute_tool_call,
    ToolDefinition,
    ToolCall,
)
from hola.session import (
    Session,
    SessionAlreadyExistsError,
    SessionDeleteError,
    SessionError,
    SessionLoadError,
    SessionNotFoundError,
    SessionSaveError,
    create_session,
    delete_session,
    get_session,
    list_sessions,
    session_exists,
)
import logging


logging.getLogger(__name__).addHandler(logging.NullHandler())
