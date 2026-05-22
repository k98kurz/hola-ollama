"""
Database model for storing workflow tasks with phase tracking.
"""

from enum import Enum
from sqloquent import SqlModel, Default, RelatedModel


class TaskStatus(Enum):
    """Status values for workflow tasks."""
    PENDING = 'PENDING'
    COMPLETED = 'COMPLETED'
    ACCEPTED = 'ACCEPTED'
    REJECTED = 'REJECTED'


class Task(SqlModel):
    """Database model for storing workflow tasks with phase tracking."""
    connection_info: str = 'hola.db'
    table: str = 'tasks'
    id_column: str = 'id'
    columns: tuple[str,] = (
        'id', 'session_id', 'sequence', 'status', 'phase', 'description',
        'memory_key'
    )
    id: str
    session_id: str
    sequence: int
    status: str | Default['PENDING']
    phase: str
    description: str
    memory_key: str | None
    session: RelatedModel

    @property
    def status(self) -> TaskStatus:
        """Return the task status as a `TaskStatus` enum. Can be set
            using the setter to change the task status.
        """
        return TaskStatus(self.data.get('status', 'PENDING').upper())

    @status.setter
    def status(self, val: TaskStatus | str) -> None:
        if type(val) is str:
            val = TaskStatus(val.upper())
        self.data['status'] = val.value
