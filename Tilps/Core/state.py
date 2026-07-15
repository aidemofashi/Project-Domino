import time
import threading
from enum import Enum


class AppState(Enum):
    IDLE = "idle"
    PROCESSING = "processing"
    RECORDING = "recording"


class TTSState(Enum):
    IDLE = "idle"
    SYNTHESIZING = "synthesizing"
    PLAYING = "playing"


class StateManager:
    def __init__(self):
        self._state = AppState.IDLE
        self._lock = threading.Lock()
        self._interrupt_event = threading.Event()

        self.last_activity_time = time.time()
        self.last_trigger_time = time.time()
        self._auto_trigger_paused = False
        self._auto_trigger_count = 0

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
        self._auto_trigger_count = 0

    def pause_auto_trigger(self):
        self._auto_trigger_paused = True

    def resume_auto_trigger(self):
        if self._auto_trigger_paused:
            self._auto_trigger_paused = False

    def mark_trigger(self):
        self.last_trigger_time = time.time()
        self._auto_trigger_count += 1

    def should_trigger(self, silence_timeout, max_triggers=2):
        if self._auto_trigger_paused:
            return False
        if self._auto_trigger_count >= max_triggers:
            return False
        now = time.time()
        return (now - self.last_activity_time > silence_timeout and
                now - self.last_trigger_time > silence_timeout)
