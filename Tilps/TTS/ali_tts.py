import dashscope
from dashscope.audio.tts_v2 import SpeechSynthesizer, AudioFormat
import sounddevice as sd
import numpy as np
import re
import threading
import queue
import time


class TTSState:
    IDLE = "idle"
    SYNTHESIZING = "synthesizing"
    PLAYING = "playing"


class AudioOutput:
    _api_key = None

    @classmethod
    def input_api(cls, api_key: str):
        cls._api_key = api_key
        dashscope.api_key = api_key

    def __init__(self, max_workers=2):
        print(f">>> 阿里云TTS, workers={max_workers}")
        self.model = "cosyvoice-v1"
        self.voice = "longmiao"
        self.sample_rate = 24000
        self.max_workers = max_workers

        self._state = TTSState.IDLE
        self._state_lock = threading.Lock()

        self._sentence_queue = queue.Queue()
        self._current_sentence_event = threading.Event()
        self._current_sentence_event.set()

        self._stop_event = threading.Event()
        self._synthesis_stop = threading.Event()

        self._generation = 0
        self._gen_lock = threading.Lock()

        self._play_thread = threading.Thread(target=self._play_sequencer, daemon=True)
        self._play_thread.start()

        self._workers = []
        for i in range(self.max_workers):
            t = threading.Thread(target=self._synthesis_worker, daemon=True)
            t.start()
            self._workers.append(t)

    @property
    def state(self):
        with self._state_lock:
            return self._state

    def _set_state(self, s):
        with self._state_lock:
            self._state = s

    def _next_gen(self):
        with self._gen_lock:
            self._generation += 1
            return self._generation

    def _current_gen(self):
        with self._gen_lock:
            return self._generation

    def _synthesis_worker(self):
        while not self._synthesis_stop.is_set():
            try:
                seq, text, gen = self._sentence_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if text is None:
                break

            if self._stop_event.is_set():
                continue

            audio_data = self._synthesize(text)
            if audio_data is not None and not self._stop_event.is_set():
                if gen != self._current_gen():
                    continue
                self._current_sentence_event.wait()
                if not self._stop_event.is_set() and gen == self._current_gen():
                    self._current_sentence_event.clear()
                    self._set_state(TTSState.PLAYING)
                    sd.play(audio_data, samplerate=self.sample_rate)
                    sd.wait()
                    self._current_sentence_event.set()

    def _synthesize(self, text):
        text = re.sub(r'[\(\uff08].*?[\)\uff09]', '', text)
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，。！？,.!?]', '', text)
        if not text.strip():
            return None
        if not self.__class__._api_key:
            print(">>> 错误：未设置阿里云 API Key")
            return None
        try:
            synthesizer = SpeechSynthesizer(
                model=self.model,
                voice=self.voice,
                format=AudioFormat.PCM_24000HZ_MONO_16BIT
            )
            audio_bytes = synthesizer.call(text)
            if self._stop_event.is_set():
                return None
            audio_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            return audio_data
        except Exception as e:
            print(f"[TTS] 阿里云合成失败: {e}")
            return None

    def _play_sequencer(self):
        while not self._synthesis_stop.is_set():
            self._current_sentence_event.wait()
            if self._sentence_queue.empty() and not self._stop_event.is_set():
                self._set_state(TTSState.IDLE)
            time.sleep(0.1)

    def speak(self, text, interrupt=False):
        if interrupt:
            self.stop()
            gen = self._next_gen()
        else:
            gen = self._current_gen()
        self._set_state(TTSState.SYNTHESIZING)

        sentences = re.split(r'(?<=[。！？\n])', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        for sentence in sentences:
            if self._stop_event.is_set():
                break
            self._sentence_queue.put((id(sentence), sentence, gen))

    def stop(self):
        self._stop_event.set()
        sd.stop()
        self._current_sentence_event.set()
        while not self._sentence_queue.empty():
            try:
                self._sentence_queue.get_nowait()
            except queue.Empty:
                break
        time.sleep(0.05)
        self._stop_event.clear()
        self._set_state(TTSState.IDLE)

    def shutdown(self):
        self._synthesis_stop.set()
        self.stop()
        for _ in self._workers:
            self._sentence_queue.put((None, None, 0))
