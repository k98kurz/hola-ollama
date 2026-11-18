from enum import Enum, auto
from flying_state_machines import AsyncFSM as FSM, AsyncTransition as Transition


class ResponseState(Enum):
    INIT = auto()
    THINKING = auto()
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
            ResponseState.THINKING
        ),
        Transition(
            ResponseState.INIT,
            ResponseEvent.WRITE_CONTENT,
            ResponseState.WRITING_CONTENT
        ),
        Transition(
            ResponseState.INIT,
            ResponseEvent.CALL_TOOL,
            ResponseState.CALLING_TOOLS
        ),
        Transition(
            ResponseState.THINKING,
            ResponseEvent.WRITE_CONTENT,
            ResponseState.WRITING_CONTENT
        ),
        Transition(
            ResponseState.THINKING,
            ResponseEvent.CALL_TOOL,
            ResponseState.CALLING_TOOLS
        ),
        Transition(
            ResponseState.THINKING,
            ResponseEvent.END_RESPONSE,
            ResponseState.DONE
        ),
        Transition(
            ResponseState.WRITING_CONTENT,
            ResponseEvent.WRITE_THOUGHT,
            ResponseState.THINKING
        ),
        Transition(
            ResponseState.WRITING_CONTENT,
            ResponseEvent.CALL_TOOL,
            ResponseState.CALLING_TOOLS
        ),
        Transition(
            ResponseState.WRITING_CONTENT,
            ResponseEvent.END_RESPONSE,
            ResponseState.DONE
        ),
        Transition(
            ResponseState.CALLING_TOOLS,
            ResponseEvent.WRITE_THOUGHT,
            ResponseState.THINKING
        ),
        Transition(
            ResponseState.CALLING_TOOLS,
            ResponseEvent.WRITE_CONTENT,
            ResponseState.WRITING_CONTENT
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
