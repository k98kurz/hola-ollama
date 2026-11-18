from enum import Enum, auto
from flying_state_machines import AsyncFSM as FSM, AsyncTransition as Transition


class ResponseState(Enum):
    INIT = auto()
    WRITING_THOUGHT = auto()
    CALLING_TOOLS = auto()
    WRITING_CONTENT = auto()
    DONE = auto()


class ResponseEvent(Enum):
    WRITE_THOUGHT = auto()
    CALL_TOOL = auto()
    WRITE_CONTENT = auto()
    END_RESPONSE = auto()


class ResponseFSM(FSM):
    initial_state = ResponseState.INIT
    rules = {
        Transition(
            ResponseState.INIT,
            ResponseEvent.WRITE_THOUGHT,
            ResponseState.WRITING_THOUGHT
        ),
        Transition(
            ResponseState.WRITING_THOUGHT,
            ResponseEvent.WRITE_CONTENT,
            ResponseState.WRITING_CONTENT
        ),
        Transition(
            ResponseState.WRITING_THOUGHT,
            ResponseEvent.CALL_TOOL,
            ResponseState.CALLING_TOOLS
        ),
        Transition(
            ResponseState.INIT,
            ResponseEvent.WRITE_CONTENT,
            ResponseState.WRITING_CONTENT
        ),
        Transition(
            ResponseState.WRITING_CONTENT,
            ResponseEvent.END_RESPONSE,
            ResponseState.DONE
        ),
        Transition(
            ResponseState.INIT,
            ResponseEvent.CALL_TOOL,
            ResponseState.CALLING_TOOLS
        ),
        Transition(
            ResponseState.CALLING_TOOLS,
            ResponseEvent.END_RESPONSE,
            ResponseState.DONE
        ),
    }

    def reset(self):
        self.current = self.initial_state
        self.previous, self.next = None, None
