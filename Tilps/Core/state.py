import time
import threading
from enum import Enum


class AppState(Enum):
    IDLE = "idle"
    PROCESSING = "processing"
    RECORDING = "recording"


class StateManager:
    def __init__(self):
        self._state = AppState.IDLE
        self._lock = threading.Lock()
        self._interrupt_event = threading.Event()

        self.last_activity_time = time.time()
        self.last_trigger_time = time.time()

    def get_state(self):
        with self._lock:
            return self._state

    def set_state(self, state):
        with self._lock:
            self._state = state

    def request_interrupt(self):
        self._interrupt_event.set()

    def consume_interrupt(self):
        if self._interrupt_event.is_set():
            self._interrupt_event.clear()
            return True
        return False

    def mark_activity(self):
        self.last_activity_time = time.time()
        self.last_trigger_time = time.time()

    def mark_trigger(self):
        self.last_trigger_time = time.time()

    def should_trigger(self, silence_timeout):
        now = time.time()
        return (now - self.last_activity_time > silence_timeout and
                now - self.last_trigger_time > silence_timeout)
