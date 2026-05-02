from enum import Enum, auto


class AssistantState(Enum):
    IDLE = auto()
    LISTENING = auto()
    TRANSCRIBING = auto()
    PROCESSING = auto()
    EXECUTING_TOOL = auto()
    SPEAKING = auto()
    ERROR = auto()
