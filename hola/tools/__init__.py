"""
Tool system for function registration and execution.
"""

from hola.tools.core import (
    ToolRegistry,
    execute_tool_call,
    get_tool,
    list_tools,
    register_tool,
    tool,
)
from hola.tools.types import (
    ToolCall,
    ToolCallFunction,
    ToolDefinition,
)
from hola.ollama.exceptions import ToolExecutionError
