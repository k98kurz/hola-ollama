"""
Provider-neutral exceptions for LLM clients.
"""

from hola.errors import HolaError


class ClientError(HolaError):
    """Base exception for all LLM client errors."""


class ClientConnectionError(ClientError):
    """Raised for connection and timeout failures."""


class ClientRequestError(ClientError):
    """Raised for 4xx/5xx HTTP status errors."""


class ClientModelError(ClientError):
    """Raised for model-specific failures."""
