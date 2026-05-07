"""
Session exceptions for Hola session management.
"""


class SessionError(Exception):
    """Base exception for session-related errors."""
    pass


class SessionNotFoundError(SessionError):
    """Raised when a session cannot be found."""
    pass


class SessionAlreadyExistsError(SessionError):
    """Raised when attempting to create a session that already exists."""
    pass


class SessionLoadError(SessionError):
    """Raised when loading a session state fails."""
    pass


class SessionSaveError(SessionError):
    """Raised when saving a session state fails."""
    pass


class SessionDeleteError(SessionError):
    """Raised when deleting a session fails."""
    pass
