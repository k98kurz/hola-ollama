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
from hola.tools.compat import (
    tools_to_xml,
    parse_tool_calls_from_xml,
    tool_calls_to_xml,
)
from hola.tools.bundled import *  # Registers all bundled tools
from hola.tools.python import python  # Registers python tool
from hola.tools.calculate import calculate  # Registers calculate tool
from hola.tools import orchestration  # Registers orchestration tools
