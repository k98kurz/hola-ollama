"""
Session exceptions for Hola session management.
"""

from hola.errors import HolaError


class SessionError(HolaError):
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


class TempFileNotFoundError(SessionError):
    """Raised when a temporary file cannot be found."""
    pass


class TempFileError(SessionError):
    """Raised when a temporary file operation fails."""
    pass
