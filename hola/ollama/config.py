"""
Configuration classes for the Ollama client and model warmer.
"""

from hola.errors import validated_dataclass, value_assert


@validated_dataclass
class ClientConfig:
    base_url: str = "http://localhost:11434"
    timeout: float = 120.0
    max_retries: int = 3
    retry_delay: float = 1.0

    def __post_init__(self) -> None:
        """Validate field values."""
        value_assert(self.max_retries >= 1, "max_retries must be >= 1")
        value_assert(self.retry_delay >= 0, "retry_delay must be >= 0")


@validated_dataclass
class WarmerConfig:
    default_keep_alive: str = "4h"
    ping_interval_seconds: int = 120
    check_interval_seconds: int = 1
    enable_background_ping: bool = True
