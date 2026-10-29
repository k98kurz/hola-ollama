"""
Configuration classes for LLM clients.
"""

from hola.errors import validated_dataclass, value_assert


@validated_dataclass
class ClientConfig:
    """Configuration for an LLM HTTP client."""
    base_url: str
    timeout: float = 120.0
    max_retries: int = 2
    retry_delay: float = 1.0
    api_key: str | None = None

    def __post_init__(self) -> None:
        """Validate field values."""
        value_assert(self.max_retries >= 0, "max_retries must be >= 0")
        value_assert(self.retry_delay >= 0, "retry_delay must be >= 0")
