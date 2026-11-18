from enum import Enum, auto
from flying_state_machines import FSM, Transition


class ResponseState(Enum):
    INIT = auto()
    WRITING_THOUGHT = auto()
    WRITING_CONTENT = auto()
    DONE = auto()


class ResponseEvent(Enum):
    WRITE_THOUGHT = auto()
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
            ResponseState.INIT,
            ResponseEvent.WRITE_CONTENT,
            ResponseState.WRITING_CONTENT
        ),
        Transition(
            ResponseState.WRITING_CONTENT,
            ResponseEvent.END_RESPONSE,
            ResponseState.DONE
        )
    }

    def reset(self):
        self.current = self.initial_state
        self.previous, self.next = None, None
