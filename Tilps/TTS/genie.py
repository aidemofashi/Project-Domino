from . import genie_tts as genie
import re
import threading
import queue
from pathlib import Path

try:
    from .genie_tts.ModelManager import model_manager
    model_manager.providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
except Exception:
    pass


class AudioOutput:
    def __init__(self, max_workers=2):
        self.script_dir = Path(__file__).parent.absolute()
        self.model_dir = self.script_dir / "models" / "feibi"
        self.ref_wav = self.script_dir / "reference" / "feibi.wav"
        self.character = "feibi"

        self.task_queue = queue.Queue()
        self._is_running = True
        self.stop_event = threading.Event()

        self._init_genie()

        for _ in range(max_workers):
            threading.Thread(target=self._worker, daemon=True).start()

    def _init_genie(self):
        try:
            if self.model_dir.exists():
                genie.load_character(self.character, str(self.model_dir), 'Chinese')
                if self.ref_wav.exists():
                    genie.set_reference_audio(self.character, str(self.ref_wav), "在此之前，请您务必享受旅居拉古娜的时光")
                print(f">>> [TTS] {self.character} 加载成功")
        except Exception as e:
            print(f">>> [TTS] 初始化失败: {e}")

    def _worker(self):
        while self._is_running:
            try:
                item = self.task_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:
                break
            text, interrupt = item
            if self.stop_event.is_set():
                continue
            try:
                genie.tts(
                    character_name=self.character,
                    text=text,
                    play=True,
                    split_sentence=True,
                )
            except Exception as e:
                print(f">>> [TTS] 合成失败: {e}")

    def speak(self, text, interrupt=False):
        if interrupt:
            self.stop()
        cleaned = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，。！？]', '', text).strip()
        if not cleaned:
            return
        self.task_queue.put((cleaned, interrupt))

    text_to_speech = speak

    def stop(self):
        self.stop_event.set()
        genie.stop()
        while not self.task_queue.empty():
            try:
                self.task_queue.get_nowait()
            except queue.Empty:
                break
        self.stop_event.clear()

    def shutdown(self):
        self._is_running = False
        self.stop()
        self.task_queue.put(None)
