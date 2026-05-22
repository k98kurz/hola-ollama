"""
Ollama-specific API types.
"""

from __future__ import annotations
from hola.errors import validated_dataclass


@validated_dataclass
class RunningModel:
    name: str
    expires_at: str
    size: int
    vm_peak: int | None = None
    vram_total: int | None = None
