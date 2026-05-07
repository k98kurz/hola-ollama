"""
Configuration classes for the Ollama client and model warmer.
"""

from dataclasses import dataclass


@dataclass
class ClientConfig:
    base_url: str = "http://localhost:11434"
    timeout: float = 120.0
    max_retries: int = 3
    retry_delay: float = 1.0


@dataclass
class WarmerConfig:
    default_keep_alive: str = "4h"
    ping_interval_seconds: int = 120
    check_interval_seconds: int = 1
    enable_background_ping: bool = True
