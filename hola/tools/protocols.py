"""
Protocol definitions for tool dependencies.
"""

from typing import Protocol
from pathlib import Path
from hola.permissions import Permission


class SessionProtocol(Protocol):
    """Minimal session interface for tool execution."""

    @property
    def cwd(self) -> str:
        """Current working directory."""
        ...

    def get_file_permission(self, path: Path | str) -> Permission:
        """Get effective permission for a path."""
        ...

    def memorize(self, key: str, value: object) -> None:
        """Store a key-value pair in session memory."""
        ...

    def recall(self, key: str) -> object:
        """Retrieve a value from session memory by key."""
        ...