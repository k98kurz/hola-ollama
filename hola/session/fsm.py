"""
General-purpose workflow FSM for session management.
"""

from enum import Enum, auto
from flying_state_machines import AsyncFSM, AsyncTransition


class WorkflowState(Enum):
    """States for the workflow FSM."""
    INIT = auto()
    PLANNING = auto()
    EXECUTING = auto()
    REVIEWING_TASK = auto()
    REVIEWING = auto()
    DONE = auto()


class WorkflowEvent(Enum):
    """Events for the workflow FSM."""
    START = auto()
    CONTINUE = auto()
    EXECUTE = auto()
    REVIEW_TASK = auto()
    REVIEW = auto()
    REJECT = auto()
    ACCEPT = auto()


class WorkflowFSM(AsyncFSM):
    """General-purpose workflow FSM with planning, execution, and
        review phases.
    """
    initial_state = WorkflowState.INIT
    rules = set([
        AsyncTransition(
            WorkflowState.INIT, WorkflowEvent.START, WorkflowState.PLANNING
        ),
        AsyncTransition(
            WorkflowState.PLANNING, WorkflowEvent.CONTINUE, WorkflowState.PLANNING
        ),
        AsyncTransition(
            WorkflowState.PLANNING, WorkflowEvent.EXECUTE, WorkflowState.EXECUTING
        ),
        AsyncTransition(
            WorkflowState.EXECUTING, WorkflowEvent.CONTINUE, WorkflowState.EXECUTING
        ),
        AsyncTransition(
            WorkflowState.EXECUTING,
            WorkflowEvent.REVIEW_TASK,
            WorkflowState.REVIEWING_TASK
        ),
        AsyncTransition(
            WorkflowState.REVIEWING_TASK,
            WorkflowEvent.REJECT,
            WorkflowState.PLANNING
        ),
        AsyncTransition(
            WorkflowState.REVIEWING_TASK,
            WorkflowEvent.ACCEPT,
            WorkflowState.EXECUTING
        ),
        AsyncTransition(
            WorkflowState.EXECUTING, WorkflowEvent.REVIEW, WorkflowState.REVIEWING
        ),
        AsyncTransition(
            WorkflowState.REVIEWING, WorkflowEvent.CONTINUE, WorkflowState.REVIEWING
        ),
        AsyncTransition(
            WorkflowState.REVIEWING, WorkflowEvent.REJECT, WorkflowState.PLANNING
        ),
        AsyncTransition(
            WorkflowState.REVIEWING, WorkflowEvent.ACCEPT, WorkflowState.DONE
        ),
    ])
