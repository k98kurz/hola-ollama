"""
General-purpose workflow FSM for session management.
"""

from enum import Enum, auto
from flying_state_machines import FSM, Transition


class WorkflowState(Enum):
    """States for the workflow FSM."""
    INIT = auto()
    PLANNING = auto()
    EXECUTING = auto()
    REVIEWING = auto()
    DONE = auto()


class WorkflowEvent(Enum):
    """Events for the workflow FSM."""
    START = auto()
    EXECUTE = auto()
    REVIEW = auto()
    REJECTED = auto()
    ACCEPTED = auto()


class WorkflowFSM(FSM):
    """General-purpose workflow FSM with planning, execution, and review phases."""
    
    initial_state = WorkflowState.INIT
    rules = set([
        Transition(WorkflowState.INIT, WorkflowEvent.START, WorkflowState.PLANNING),
        Transition(WorkflowState.PLANNING, WorkflowEvent.EXECUTE, WorkflowState.EXECUTING),
        Transition(WorkflowState.EXECUTING, WorkflowEvent.REVIEW, WorkflowState.REVIEWING),
        Transition(WorkflowState.REVIEWING, WorkflowEvent.REJECTED, WorkflowState.PLANNING),
        Transition(WorkflowState.REVIEWING, WorkflowEvent.ACCEPTED, WorkflowState.DONE),
    ])
