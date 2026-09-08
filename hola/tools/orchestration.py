"""
Orchestration tools for task management in workflow phases.
"""

from hola.session.session import MAX_TASK_DESCRIPTION_LENGTH
from hola.tools.core import tool
from hola.types import SessionProtocol


@tool(
    name="view_tasks",
    description="View all tasks for the session, optionally filtered by phase/status"
)
def view_tasks(
        session: SessionProtocol, phase: str | None = None,
        status: str | None = None
    ) -> list[dict]:
    """View all tasks for the session, optionally filtered by phase/status."""
    tasks = session.list_tasks(phase, status)
    return [
        {
            'phase': task.phase,
            'sequence': task.sequence,
            'status': task.status.value,
            'description': task.description,
            'memory_key': task.memory_key,
        }
        for task in tasks
    ]


@tool(
    name="add_task",
    description="Create a new task with pending status. Description must be "
    f"{MAX_TASK_DESCRIPTION_LENGTH} characters or fewer; put extra detail in "
    "notes. Returns task dict"
)
def add_task(
        session: SessionProtocol, phase: str, description: str,
        notes: str | None = None
    ) -> dict:
    """Create a new task with pending status. Returns task dict."""
    task = session.add_task(phase, description, notes)
    return {
        'phase': task.phase,
        'sequence': task.sequence,
        'status': task.status.value,
        'description': task.description,
        'memory_key': task.memory_key,
    }


@tool(
    name="move_task",
    description="Move a task to a new sequence, renumbering subsequent tasks"
)
def move_task(
        session: SessionProtocol, phase: str, old_sequence: int,
        new_sequence: int
    ) -> list[str]:
    """Move a task to a new sequence, renumbering affected tasks."""
    return session.move_task(phase, old_sequence, new_sequence)


@tool(name="delete_task", description="Delete a task by phase and sequence number")
def delete_task(
        session: SessionProtocol, phase: str, sequence: int
    ) -> None:
    """Delete a task by phase and sequence number."""
    session.delete_task(phase, sequence)


@tool(
    name="complete_task",
    description="Mark a task as completed, optionally storing result in memory"
)
def complete_task(
        session: SessionProtocol, phase: str, sequence: int,
        result: str | None = None
    ) -> None:
    """Mark a task as completed, optionally storing result in memory."""
    session.complete_task(phase, sequence, result)


@tool(name="accept_task", description="Mark a task as accepted")
def accept_task(
        session: SessionProtocol, phase: str, sequence: int
    ) -> None:
    """Mark a task as accepted."""
    session.accept_task(phase, sequence)


@tool(
    name="reject_task",
    description="Mark a task as rejected and store reason in memory"
)
def reject_task(
        session: SessionProtocol, phase: str, sequence: int, reason: str
    ) -> None:
    """Mark a task as rejected and store reason in memory."""
    session.reject_task(phase, sequence, reason)


@tool(name="accept", description="Mark the work done as acceptable")
def accept(session: SessionProtocol) -> None:
    """Mark the workflow as done and the work as acceptable."""
    session.accept()


@tool(name="reject", description="Reject the work done with actionable feedback")
def reject(session: SessionProtocol, reason: str) -> None:
    """Mark the workflow as rejected with actionable feedback by the
        review agent.
    """
    session.reject(reason)
