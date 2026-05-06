"""
Custom exceptions for Ollama client operations.
"""


class OllamaClientError(Exception):
    """Base exception for all Ollama client errors."""


class OllamaConnectionError(OllamaClientError):
    """Raised for connection/timeout failures."""


class OllamaRequestError(OllamaClientError):
    """Raised for 4xx/5xx HTTP status errors."""


class OllamaModelError(OllamaClientError):
    """Raised for model load/unload errors."""


class ToolExecutionError(OllamaClientError):
    """Raised for tool execution failures."""
