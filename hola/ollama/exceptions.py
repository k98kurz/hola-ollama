"""
Custom exceptions for Ollama client operations.
"""

from hola.client.exceptions import (
    ClientConnectionError,
    ClientError,
    ClientModelError,
    ClientRequestError,
)


class OllamaClientError(ClientError):
    """Base exception for all Ollama client errors."""


class OllamaConnectionError(OllamaClientError, ClientConnectionError):
    """Raised for connection/timeout failures."""


class OllamaRequestError(OllamaClientError, ClientRequestError):
    """Raised for 4xx/5xx HTTP status errors."""


class OllamaModelError(OllamaClientError, ClientModelError):
    """Raised for model load/unload errors."""
