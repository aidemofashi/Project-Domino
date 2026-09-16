from dataclasses import dataclass, field
from enum import Enum, auto
import time


class RequestType(Enum):
    VOICE_INPUT = auto()
    TIMER_TRIGGER = auto()


class Priority(Enum):
    HIGH = 0
    NORMAL = 1


@dataclass
class Request:
    type: RequestType
    payload: dict
    priority: Priority = Priority.NORMAL
    timestamp: float = field(default_factory=time.time)
