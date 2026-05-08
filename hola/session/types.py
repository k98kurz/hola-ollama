"""
Session types for Hola session management.

Note: SessionState is deprecated and only maintained for backward compatibility.
The session system now uses a SQLite database with ORM models.
"""

import warnings
from dataclasses import dataclass
from typing import Any


@dataclass
class SessionState:
    """Deprecated: In-memory session data (use database models instead).

    This class is maintained for backward compatibility only.
    All session data is now persisted via the database models in
    hola.session.models.
    """

    def __init__(self) -> None:
        warnings.warn(
            "SessionState is deprecated. "
            "Use database models in hola.session.models instead.",
            DeprecationWarning,
            stacklevel=2
        )
        self.conversation_history = []
        self.tool_outputs = {}
        self.fsm_state = {}
        self.metadata = {}
        self.memory = {}
