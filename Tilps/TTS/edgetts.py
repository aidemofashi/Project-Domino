import edge_tts
import asyncio
import sounddevice as sd
import numpy as np
import re
import threading
import queue
import time
import miniaudio


class TTSState:
    IDLE = "idle"
    SYNTHESIZING = "synthesizing"
    PLAYING = "playing"


class AudioOutput:
    def __init__(self, max_workers=2):
        print(f">>> Edge-TTS, workers={max_workers}")
        self.voice = "zh-CN-XiaoxiaoNeural"
        self.rate = "+15%"
        self.max_workers = max_workers

        self._state = TTSState.IDLE
        self._state_lock = threading.Lock()

        self._sentence_queue = queue.Queue()
        self._current_sentence_event = threading.Event()
        self._current_sentence_event.set()

        self._stop_event = threading.Event()
        self._synthesis_stop = threading.Event()

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

    def _synthesis_worker(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        while not self._synthesis_stop.is_set():
            try:
                seq, text = self._sentence_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if text is None:
                break

            if self._stop_event.is_set():
                continue

            audio_data = self._synthesize(text, loop)
            if audio_data is not None and not self._stop_event.is_set():
                self._current_sentence_event.wait()
                if not self._stop_event.is_set():
                    self._current_sentence_event.clear()
                    self._set_state(TTSState.PLAYING)
                    sd.play(audio_data, samplerate=24000)
                    sd.wait()
                    self._current_sentence_event.set()

    def _synthesize(self, text, loop):
        text = re.sub(r'[\(\uff08].*?[\)\uff09]', '', text)
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，。！？,.!?\n]', '', text)
        if not text.strip():
            return None

        mp3_chunks = []
        communicate = edge_tts.Communicate(text, self.voice, rate=self.rate)
        async def run():
            async for chunk in communicate.stream():
                if self._stop_event.is_set():
                    break
                if chunk["type"] == "audio":
                    mp3_chunks.append(chunk["data"])
        try:
            loop.run_until_complete(run())
        except Exception as e:
            print(f"[TTS] Synthesis error: {e}")
            return None

        if not mp3_chunks:
            return None

        mp3_data = b"".join(mp3_chunks)
        decoded = miniaudio.decode(
            mp3_data, output_format=miniaudio.SampleFormat.FLOAT32,
            nchannels=1, sample_rate=24000
        )
        return np.frombuffer(decoded.samples, dtype=np.float32)

    def _play_sequencer(self):
        while not self._synthesis_stop.is_set():
            self._current_sentence_event.wait()
            if self._sentence_queue.empty() and not self._stop_event.is_set():
                self._set_state(TTSState.IDLE)
            time.sleep(0.1)

    def speak(self, text, interrupt=False):
        if interrupt:
            self.stop()

        self._set_state(TTSState.SYNTHESIZING)

        sentences = re.split(r'(?<=[。！？\n])', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        for sentence in sentences:
            if self._stop_event.is_set():
                break
            self._sentence_queue.put((id(sentence), sentence))

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
            self._sentence_queue.put((None, None))
