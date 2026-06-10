import queue
import threading
import time
import keyboard

from Tilps.Core.request import Request, RequestType, Priority
from Tilps.Core.state import AppState, StateManager
from Tilps.Core.pipeline import Pipeline
from Tilps.VAD.vad_vosk import AudioInput


class RequestCore:
    def __init__(self):
        self.request_queue = queue.Queue()
        self.state = StateManager()
        self.modules = {}
        self.silence_timeout = 60
        self.make_memory = 16
        self._running = False
    
    #注册功能
    def register(self, name, module):
        self.modules[name] = module

    def emit(self, request):
        if request.priority == Priority.HIGH:
            self.state.set_state(AppState.RECORDING)
            self.state.request_interrupt()
            if "tts" in self.modules:
                self.modules["tts"].stop()
        self.request_queue.put(request)

    def _handle_vad_interrupt(self):
        self.state.set_state(AppState.RECORDING)
        self.state.request_interrupt()
        if "tts" in self.modules:
            self.modules["tts"].stop()

    def _vad_worker(self):
        while self._running:
            audio_data = AudioInput.record()
            if audio_data is not None and len(audio_data) > 0:
                self.emit(
                    Request(
                        type=RequestType.VOICE_INPUT,
                        payload={"audio_data": audio_data},
                        priority=Priority.HIGH,
                    )
                )

    def run(self):
        AudioInput.interrupt_callback = self._handle_vad_interrupt
        self._running = True

        pipeline = Pipeline(self.state, self.modules, self.make_memory)

        initial_memory = self.modules.get("memory")
        if initial_memory:
            pipeline.load_history(initial_memory)

        t = threading.Thread(target=self._vad_worker, daemon=True)
        t.start()

        print("\n" + "=" * 30)
        print("双向流式模式已就绪 (LLM Stream + TTS Stream)")
        print("提示：按 [空格] 键开始，按 [Esc] 退出")
        print("=" * 30)

        keyboard.wait("space")
        print("\n>>> 系统启动！")

        while self._running:
            try:
                if self.state.get_state() == AppState.IDLE:
                    if self.state.should_trigger(self.silence_timeout):
                        self.emit(
                            Request(
                                type=RequestType.TIMER_TRIGGER,
                                payload={},
                                priority=Priority.NORMAL,
                            )
                        )

                try:
                    request = self.request_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                current_state = self.state.get_state()
                if current_state in (AppState.IDLE, AppState.RECORDING):
                    self.state.consume_interrupt()
                    self.state.set_state(AppState.PROCESSING)
                    try:
                        pipeline.execute(request)
                    finally:
                        if self.state.get_state() == AppState.PROCESSING:
                            self.state.set_state(AppState.IDLE)

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"\n[运行错误]: {e}")
                time.sleep(1)

    def shutdown(self):
        self._running = False
