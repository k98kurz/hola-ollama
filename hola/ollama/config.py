"""
Configuration classes for the Ollama model warmer.
"""

from hola.errors import validated_dataclass


@validated_dataclass
class WarmerConfig:
    default_keep_alive: str = "4h"
    ping_interval_seconds: int = 120
    check_interval_seconds: int = 1
    enable_background_ping: bool = True
