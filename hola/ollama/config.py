"""
Configuration classes for the Ollama client and model warmer.
"""

from dataclasses import dataclass, field
from enum import Enum


class ModelPriority(Enum):
    CRITICAL = "CRITICAL"
    NORMAL = "NORMAL"
    LOW = "LOW"


@dataclass
class ClientConfig:
    base_url: str = "http://localhost:11434"
    timeout: float = 120.0
    max_retries: int = 3
    retry_delay: float = 1.0


@dataclass
class WarmerConfig:
    default_keep_alive: str = "4h"
    default_ping_interval: float = 300.0
    ping_intervals: dict[str, float] = field(
        default_factory=lambda: {
            "CRITICAL": 120.0,
            "NORMAL": 300.0,
            "LOW": 600.0,
        }
    )
    enable_background_ping: bool = True
